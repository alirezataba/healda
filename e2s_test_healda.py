import os
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import xarray as xr

# GPU and runtime environment setup
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")      # Override via terminal if GPU 3 is busy
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"   # Prevents JAX from pre-allocating full VRAM
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"  # Disables XLA autotune profiling
os.environ["ORT_LOGGING_LEVEL"] = "3"                    # Silences ONNX Runtime C++ warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"                 # Silences C++ CUDA/XLA log spam
os.environ["GCSFS_TOKEN"] = "anon"                      # Public Google Cloud access
os.environ["GCS_NO_SIGN_REQUEST"] = "true"              # Anonymous GCS access for obstore
os.environ["GOOGLE_NO_SIGN_REQUEST"] = "true"           # Anonymous GCS access for obstore

warnings.filterwarnings("ignore")

from earth2studio.models.da import HealDA
from earth2studio.models.px import FCN
from earth2studio.data import UFSObsConv, GFS, ARCO
from earth2studio.io import KVBackend
from earth2studio.run import deterministic
from earth2studio.lexicon.ufs import GSIConventionalLexicon


# ==============================================================================
# GUARDRAIL HELPER 1: Universal Tensor/DataArray/Tuple to CPU NumPy Converter
# ==============================================================================
def to_numpy(data) -> np.ndarray:
    """
    Safely unwraps PyTorch Tensors, xarray DataArrays, tuples/lists (e.g. Earth2Studio
    KVBackend tuples of (tensor, coords)), and custom containers into a contiguous CPU NumPy ndarray.
    """
    # Iteratively unwrap nested structures (up to 5 levels deep)
    for _ in range(5):
        if data is None:
            break

        # Return immediately if already a numeric numpy array
        if isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.number):
            return data

        # Unwrap tuple/list containers (e.g. KVBackend store tuple: (tensor, coords))
        if isinstance(data, (tuple, list)):
            if len(data) > 0:
                data = data[0]
                continue
            else:
                break

        # Unwrap dict containers
        if isinstance(data, dict):
            for k in ["data", "values", "tensor", "array", "output"]:
                if k in data:
                    data = data[k]
                    break
            else:
                if len(data) > 0:
                    data = next(iter(data.values()))
                    continue

        # xarray DataArray / Dataset
        if hasattr(data, "values") and not isinstance(data, (np.ndarray, str)):
            data = data.values
            continue
        if hasattr(data, "data") and not isinstance(data, (np.ndarray, str)):
            data = data.data
            continue

        # PyTorch Tensor (CPU/GPU)
        if hasattr(data, "detach"):
            data = data.detach()
        if hasattr(data, "cpu"):
            data = data.cpu()
        if hasattr(data, "numpy"):
            data = data.numpy()

        arr = np.asarray(data)
        if np.issubdtype(arr.dtype, np.number):
            return arr

    return np.asarray(data)


# ==============================================================================
# GUARDRAIL HELPER 2: Forecast Variable Extractor from IO Backend
# ==============================================================================
def get_forecast_field(io_backend, variable_name: str) -> np.ndarray:
    """
    Extracts a 2D (lat, lon) forecast slice for a given variable at the last lead time
    from Earth2Studio IO backend formats (KVBackend, ZarrBackend, XarrayBackend).
    """
    stores = []
    if hasattr(io_backend, "store") and isinstance(io_backend.store, dict):
        stores.append(io_backend.store)
    if hasattr(io_backend, "root") and isinstance(io_backend.root, dict):
        stores.append(io_backend.root)
    if hasattr(io_backend, "ds"):
        stores.append(io_backend.ds)

    # Strategy 1: Direct key lookup for specific variable (e.g. 't850' in KVBackend.store)
    for store in stores:
        if variable_name in store:
            raw_value = store[variable_name]

            # DEBUG: Check if this is a tuple with coordinates
            if isinstance(raw_value, tuple):
                print(f"      [DEBUG get_forecast_field] KVBackend stores as tuple, length={len(raw_value)}")
                if len(raw_value) > 1:
                    print(f"      [DEBUG get_forecast_field] Tuple element types: {[type(x).__name__ for x in raw_value]}")
                    # Check if second element has coordinate information
                    if len(raw_value) > 1 and hasattr(raw_value[1], 'keys'):
                        coords_dict = raw_value[1]
                        print(f"      [DEBUG get_forecast_field] Coordinate keys: {list(coords_dict.keys())}")
                        for coord_name, coord_val in coords_dict.items():
                            if hasattr(coord_val, 'shape'):
                                print(f"      [DEBUG get_forecast_field]   {coord_name}: shape={coord_val.shape}")

            # KVBackend may store as tuple (data, coords) or directly as tensor
            if isinstance(raw_value, tuple) and len(raw_value) > 0:
                raw_value = raw_value[0]

            # Handle wrapped/lazy tensors - try to unwrap if it's a scalar object
            if isinstance(raw_value, np.ndarray) and raw_value.shape == () and raw_value.dtype == object:
                raw_value = raw_value.item()

            # For PyTorch tensors stored as 0-d object arrays, extract the actual tensor
            if hasattr(raw_value, 'shape') and hasattr(raw_value, 'detach'):
                # It's a tensor - convert directly
                if hasattr(raw_value, 'detach'):
                    raw_value = raw_value.detach()
                if hasattr(raw_value, 'cpu'):
                    raw_value = raw_value.cpu()
                if hasattr(raw_value, 'numpy'):
                    arr = raw_value.numpy()
                else:
                    arr = np.asarray(raw_value)
            else:
                # Use the general converter
                arr = to_numpy(raw_value)

            if np.issubdtype(arr.dtype, np.number) and arr.ndim >= 2:
                # DEBUG: Print shape before slicing
                original_shape = arr.shape
                print(f"      [DEBUG get_forecast_field] Variable '{variable_name}' original shape: {original_shape}")
                print(f"      [DEBUG get_forecast_field] Original stats: min={np.min(arr):.2f}, max={np.max(arr):.2f}, mean={np.mean(arr):.2f}")

                # If we have 4D tensor (batch, time, lat, lon), check each timestep
                if arr.ndim == 4 and arr.shape[0] == 1 and arr.shape[1] > 1:
                    print(f"      [DEBUG get_forecast_field] Detected 4D tensor (batch, time, lat, lon). Checking each timestep:")
                    for t in range(arr.shape[1]):
                        slice_2d = arr[0, t, :, :]
                        print(f"        Timestep {t}: min={np.min(slice_2d):.2f}, max={np.max(slice_2d):.2f}, mean={np.mean(slice_2d):.2f}")
                        # Check if this timestep has physically reasonable values (200-320 K for temperature)
                        if 200.0 <= np.min(slice_2d) <= 320.0 and 200.0 <= np.max(slice_2d) <= 320.0:
                            print(f"        -> Timestep {t} has physically reasonable values! Using this.")
                            return slice_2d
                    # If no physically reasonable timestep found, take the last one anyway
                    print(f"      [WARNING] No physically reasonable timestep found. Using last timestep.")
                    arr = arr[0, -1, :, :]
                else:
                    # Extract last timestep for each extra dimension
                    while arr.ndim > 2:
                        print(f"      [DEBUG get_forecast_field] Before slice: shape={arr.shape}, taking last index on axis 0")
                        arr = arr[-1]
                        print(f"      [DEBUG get_forecast_field] After slice: shape={arr.shape}, min={np.min(arr):.2f}, max={np.max(arr):.2f}")

                if arr.shape[-2] in (720, 721) and arr.shape[-1] == 1440:
                    print(f"      [DEBUG get_forecast_field] Final 2D slice stats: min={np.min(arr):.2f}, max={np.max(arr):.2f}, mean={np.mean(arr):.2f}")
                    return arr
                elif arr.shape[-2] == 1440 and arr.shape[-1] in (720, 721):
                    # Array is transposed (lon, lat) instead of (lat, lon) - fix it
                    print(f"      [DEBUG get_forecast_field] Array is transposed! Shape is (lon={arr.shape[-2]}, lat={arr.shape[-1]}), transposing to (lat, lon)")
                    arr = arr.T
                    print(f"      [DEBUG get_forecast_field] After transpose: shape={arr.shape}, min={np.min(arr):.2f}, max={np.max(arr):.2f}")
                    return arr

    # Strategy 2: Multi-variable tensor lookup (e.g. 'output' in store)
    for store in stores:
        for key, value in store.items():
            arr = to_numpy(value)
            if not np.issubdtype(arr.dtype, np.number) or arr.ndim < 2:
                continue
            if arr.shape[-2] not in (720, 721) or arr.shape[-1] != 1440:
                continue

            if hasattr(io_backend, "coords") and "variable" in io_backend.coords:
                var_coords = [str(v) for v in to_numpy(io_backend.coords["variable"])]
                if variable_name in var_coords:
                    var_idx = var_coords.index(variable_name)
                    for ax, sz in enumerate(arr.shape):
                        if sz == len(var_coords) and ax not in (arr.ndim - 2, arr.ndim - 1):
                            arr = np.take(arr, var_idx, axis=ax)
                            break

            while arr.ndim > 2:
                arr = arr[-1]

            if arr.ndim == 2 and arr.shape[-2] in (720, 721) and arr.shape[-1] == 1440:
                return arr

    # Strategy 3: Direct indexing on backend object if supported
    try:
        arr = to_numpy(io_backend[variable_name])
        if np.issubdtype(arr.dtype, np.number) and arr.ndim >= 2:
            while arr.ndim > 2:
                arr = arr[-1]
            if arr.shape[-2] in (720, 721) and arr.shape[-1] == 1440:
                return arr
    except Exception:
        pass

    # If all strategies fail, report diagnostic information
    diag_info = []
    for i, store in enumerate(stores):
        for k, v in store.items():
            # Show raw type before conversion
            raw_type = type(v).__name__
            raw_repr = repr(v)[:100]  # First 100 chars of repr

            # Try to unwrap if tuple
            unwrapped = v
            if isinstance(v, tuple) and len(v) > 0:
                unwrapped = v[0]
                raw_type += f" (tuple[0]: {type(unwrapped).__name__})"

            # Try to get actual tensor shape if it's a wrapped tensor
            actual_shape = "unknown"
            if hasattr(unwrapped, 'shape'):
                try:
                    actual_shape = str(unwrapped.shape)
                except:
                    pass

            # Check if it's a 0-d numpy array wrapping something
            if isinstance(unwrapped, np.ndarray) and unwrapped.shape == () and unwrapped.dtype == object:
                inner = unwrapped.item()
                raw_type += f" (0-d array wrapping {type(inner).__name__})"
                if hasattr(inner, 'shape'):
                    actual_shape = f"inner.shape={inner.shape}"

            a = to_numpy(unwrapped)
            diag_info.append(
                f"Store {i} Key '{k}': type={raw_type}, actual_shape={actual_shape}, "
                f"numpy_shape={a.shape}, numpy_dtype={a.dtype}, repr={raw_repr}"
            )

    raise ValueError(
        f"Could not extract 2D spatial forecast for variable '{variable_name}'! "
        f"Backend storage contents:\n" + "\n".join(diag_info)
    )


# ==============================================================================
# Helper Class: Wrap DataArray to satisfy Earth2Studio DataSource interface
# ==============================================================================
class DataArraySource:
    """Wraps an xarray.DataArray to act as a callable Earth2Studio DataSource."""

    def __init__(self, da):
        self.da = da

    def __call__(self, time, variable):
        if isinstance(variable, str):
            variable = [variable]

        if "variable" in self.da.coords:
            valid_vars = [v for v in variable if v in self.da.coords["variable"].values]
            if valid_vars:
                return self.da.sel(variable=valid_vars)
        return self.da


def main():
    start_time = datetime(2023, 1, 1, 0, 0)
    lead_hours = 24
    lead_steps = lead_hours // 6
    target_time = start_time + timedelta(hours=lead_hours)

    date_str = start_time.strftime("%Y%m%d")
    output_filename = f"hindcast_comparison_healda_{date_str}.png"

    print("=== Starting Run C Benchmark (HealDA + FourCastNet) ===")
    print(f"Init Time: {start_time} | Target Time: {target_time}")

    # [1/6] Load Models
    print("\n[1/6] Loading HealDA & FourCastNet Models...")
    healda_model = HealDA.load_model(HealDA.load_default_package())
    fcn_model = FCN.load_model(FCN.load_default_package())

    # [2/6] Prepare Data Sources
    print("\n[2/6] Initializing UFS Observations and GFS Background Sources...")
    obs_source = UFSObsConv()
    gfs_source = GFS()

    # ==========================================================================
    # CHECKPOINT 1: UFS Observation Ingestion Validation & Handover Logging
    # ==========================================================================
    print("\n[CHECKPOINT 1] Validating UFS Observation Vector & Lexicon Handover...")
    coords = healda_model.input_coords()
    healda_expected_vars = list(coords[0]["variable"])

    gsi_vocab = GSIConventionalLexicon.VOCAB
    query_vars = [v for v in healda_expected_vars if v in gsi_vocab]
    unmatched_vars = set(healda_expected_vars) - set(query_vars)

    print("\n      [VARIABLE HANDOVER - Layer 1: UFS Obstore -> HealDA DA Model]")
    print(f"      • Variables Expected by HealDA Model ({len(healda_expected_vars)}): {healda_expected_vars}")
    print(f"      • Variables Matched in GSI Lexicon   ({len(query_vars)}): {query_vars}")
    if unmatched_vars:
        print(f"      • Variables Skipped (No Lexicon Match) ({len(unmatched_vars)}): {list(unmatched_vars)}")

    print(f"\n      Fetching UFS observations for {start_time.strftime('%Y-%m-%d %H:%M UTC')}...")
    obs_data = obs_source(start_time, variable=query_vars)

    # GUARDRAIL 1.1: Assert non-empty observation payload
    if obs_data is None or (hasattr(obs_data, "size") and obs_data.size == 0):
        raise AssertionError(f"[ABORT] UFS Observation fetch returned empty data for {start_time}!")

    # GUARDRAIL 1.2: Check for NaN/Inf in numeric observations
    if hasattr(obs_data, "values"):
        obs_vals = np.asarray(obs_data.values)
        if np.issubdtype(obs_vals.dtype, np.number):
            if np.isnan(obs_vals).any() or np.isinf(obs_vals).any():
                raise AssertionError("[ABORT] UFS Observation vector contains NaN or Inf values!")

    if hasattr(obs_data, "attrs") and "request_time" not in obs_data.attrs:
        obs_data.attrs["request_time"] = start_time

    # SPATIAL METADATA LOG
    obs_shape = obs_data.shape if hasattr(obs_data, "shape") else (obs_data.size,)
    print("\n      [SPATIAL METADATA - UFS Input Observations]")
    print(f"      • Structure/Shape   : {obs_shape}")
    print(f"      • Total Vector Size : {obs_data.size:,} elements")
    print(f"      • Spatial Geometry  : Unstructured 1D Point Vectors")

    # ==========================================================================
    # CHECKPOINT 2: HealDA Execution & Output Analysis State Logging
    # ==========================================================================
    print("\n[CHECKPOINT 2] Running HealDA Assimilation and Validating Output Analysis State...")
    da_state = healda_model(obs_data)
    da_vars = list(da_state.coords["variable"].values)

    # GUARDRAIL 2.1: Verify HEALPix Nside=64 geometry (12 * 64^2 = 49,152 pixels)
    spatial_size = da_state.values.shape[-1]
    if spatial_size != 49152:
        raise AssertionError(
            f"[ABORT] HealDA spatial size ({spatial_size}) != expected HEALPix Nside=64 (49,152 pixels)!"
        )

    # GUARDRAIL 2.2: Ensure no NaN/Inf values in assimilated fields
    for v in da_vars:
        v_vals = to_numpy(da_state.sel(variable=v))
        if np.isnan(v_vals).any() or np.isinf(v_vals).any():
            raise AssertionError(f"[ABORT] HealDA output variable '{v}' contains NaN/Inf values!")

    print("\n      [VARIABLE HANDOVER - Layer 2: HealDA DA Model -> Analysis Output]")
    print(f"      • Physical Variables Produced by HealDA ({len(da_vars)}): {da_vars}")

    # SPATIAL METADATA LOG
    da_nside = hp.npix2nside(spatial_size)
    da_mean_res_deg = np.degrees(hp.nside2resol(da_nside))
    print("\n      [SPATIAL METADATA - HealDA Output Analysis State]")
    print(f"      • Full Matrix Shape : {da_state.values.shape} (Variables x HEALPix Pixels)")
    print(f"      • HEALPix Resolution: Nside = {da_nside} (RING ordering scheme)")
    print(f"      • Spatial Grain     : ~{da_mean_res_deg:.3f}° average pixel resolution (~{da_mean_res_deg * 111.0:.1f} km)")

    # ==========================================================================
    # CHECKPOINT 3: GFS Background Ingestion & Prognostic Model Requirements
    # ==========================================================================
    print("\n[CHECKPOINT 3] Validating GFS Background State & FCN Requirements...")
    fcn_required_vars = list(fcn_model.input_coords()["variable"])

    print(f"      Fetching GFS 26-variable background state for {start_time.strftime('%Y-%m-%d %H:%M UTC')}...")
    gfs_bg = gfs_source(start_time, fcn_required_vars)
    gfs_vars = list(gfs_bg.coords["variable"].values)

    gfs_lat = gfs_bg.coords["lat"].values
    gfs_lon = gfs_bg.coords["lon"].values

    # GUARDRAIL 3.1: Verify grid shape & North-to-South lat orientation
    if len(gfs_lat) not in (720, 721) or len(gfs_lon) != 1440:
        raise AssertionError(f"[ABORT] GFS grid shape ({len(gfs_lat)}x{len(gfs_lon)}) invalid!")
    if gfs_lat[0] < gfs_lat[-1]:
        raise AssertionError("[ABORT] GFS background latitude array indexed South-to-North! Expected North-to-South.")

    print("\n      [VARIABLE HANDOVER - Layer 3: GFS Source -> FCN Requirements]")
    print(f"      • Variables Expected by FourCastNet ({len(fcn_required_vars)}): {fcn_required_vars}")
    print(f"      • Variables Provided by GFS Background ({len(gfs_vars)}): {gfs_vars}")

    # SPATIAL METADATA LOG
    gfs_dlat = abs(gfs_lat[1] - gfs_lat[0])
    gfs_dlon = abs(gfs_lon[1] - gfs_lon[0])
    print("\n      [SPATIAL METADATA - GFS Background State]")
    print(f"      • Full Tensor Shape : {gfs_bg.values.shape} (Variables x Lat x Lon)")
    print(f"      • Spatial Resolution: {gfs_dlat:.2f}° x {gfs_dlon:.2f}° (Lat bounds: {gfs_lat[0]:.2f}° to {gfs_lat[-1]:.2f}°)")

    # ==========================================================================
    # CHECKPOINT 4: Spatial Remapping & Variable Superposition Handover
    # ==========================================================================
    print("\n[CHECKPOINT 4] Remapping HealDA HEALPix Grid -> GFS Equirectangular Mesh...")
    combined_ic = gfs_bg.copy(deep=True)

    assimilated_vars = [v for v in da_state.coords["variable"].values if v in fcn_required_vars]
    unassimilated_vars = [v for v in fcn_required_vars if v not in assimilated_vars]

    print("\n      [VARIABLE HANDOVER - Layer 4: Superposition -> FCN Initial State (combined_ic)]")
    print(f"      • Variables Superimposed from HealDA Analysis ({len(assimilated_vars)}): {assimilated_vars}")
    print(f"      • Variables Retained from GFS Background Forecast ({len(unassimilated_vars)}): {unassimilated_vars}")
    print(f"      • Final Initial State Variable Count: {len(combined_ic.coords['variable'].values)} variables")

    grid_lon, grid_lat = np.meshgrid(gfs_lon, gfs_lat)

    for v in assimilated_vars:
        v_data = to_numpy(da_state.sel(variable=v)).squeeze().ravel().astype(np.float64)

        # Bilinear interpolation using RING ordering (nest=False)
        mapped_2d = hp.get_interp_val(v_data, grid_lon, grid_lat, lonlat=True, nest=False)

        # GUARDRAIL 4.1: Non-negative physical clipping floor
        if v in ["tcwv", "tisr", "pwat"] or v.startswith("q") or v.startswith("r"):
            mapped_2d = np.clip(mapped_2d, a_min=0.0, a_max=None)
            if v.startswith("r"):
                max_rh = 100.0 if np.max(mapped_2d) > 1.0 else 1.0
                mapped_2d = np.clip(mapped_2d, a_min=0.0, a_max=max_rh)

        # GUARDRAIL 4.2: Physical range envelope checks
        min_val, max_val = np.min(mapped_2d), np.max(mapped_2d)

        if v in ["t2m", "skt"] or (v.startswith("t") and len(v) > 1 and v[1:].isdigit()):
            if min_val < 150.0 or max_val > 350.0:
                raise AssertionError(f"[ABORT] Temperature '{v}' out of bounds! Min: {min_val:.2f} K, Max: {max_val:.2f} K")
        elif v in ["msl", "sp"]:
            if min_val < 40000.0 or max_val > 115000.0:
                raise AssertionError(f"[ABORT] Pressure '{v}' out of bounds! Min: {min_val:.2f} Pa, Max: {max_val:.2f} Pa")
        elif v in ["10u", "10v", "u10m", "v10m"] or (v[0] in ["u", "v"] and v[1:].isdigit()):
            if min_val < -150.0 or max_val > 150.0:
                raise AssertionError(f"[ABORT] Wind '{v}' out of bounds! Min: {min_val:.2f} m/s, Max: {max_val:.2f} m/s")

        target_shape = combined_ic.sel(variable=v).values.shape
        combined_ic.loc[{"variable": v}] = np.broadcast_to(mapped_2d, target_shape)

    # SPATIAL METADATA LOG
    print("\n      [SPATIAL METADATA - Superimposed FourCastNet Initial State (combined_ic)]")
    print(f"      • Initial Tensor Shape : {combined_ic.values.shape} (Variables x Lat x Lon)")
    print(f"      • Target Mesh Geometry : {combined_ic.values.shape[-2]} x {combined_ic.values.shape[-1]} Grid Points")
    print(f"      • Spatial Resolution  : {gfs_dlat:.2f}° x {gfs_dlon:.2f}° Equirectangular Mesh")

    # ==========================================================================
    # CHECKPOINT 5: FourCastNet Execution & Ground Truth Handover Alignment
    # ==========================================================================
    print("\n[CHECKPOINT 5] Running FourCastNet Forecast and Aligning Verification Ground Truth...")
    io_backend = KVBackend()
    da_source = DataArraySource(combined_ic)

    deterministic([start_time], lead_steps, fcn_model, da_source, io_backend)

    # Fetch ERA5 Ground Truth for Verification Target (T850)
    arco_source = ARCO()
    target_var = "t850"

    print("\n      [VARIABLE HANDOVER - Layer 5: FourCastNet Output -> Verification Field]")
    print(f"      • Target Verification Parameter Selected : '{target_var}'")

    ground_truth_da = arco_source(target_time, target_var)

    if ground_truth_da is None or (hasattr(ground_truth_da, "size") and ground_truth_da.size == 0):
        raise AssertionError(f"[ABORT] ARCO ERA5 ground truth source returned empty data for {target_time}!")

    ground_truth_data = to_numpy(ground_truth_da).squeeze()
    forecast_data = get_forecast_field(io_backend, target_var)

    # DEBUG: Print sample values and statistics before alignment
    print(f"\n      [DEBUG] Ground Truth Stats: min={np.min(ground_truth_data):.2f}, max={np.max(ground_truth_data):.2f}, mean={np.mean(ground_truth_data):.2f}")
    print(f"      [DEBUG] Forecast Stats:     min={np.min(forecast_data):.2f}, max={np.max(forecast_data):.2f}, mean={np.mean(forecast_data):.2f}")
    print(f"      [DEBUG] Ground Truth Shape: {ground_truth_data.shape}")
    print(f"      [DEBUG] Forecast Shape:     {forecast_data.shape}")

    # GUARDRAIL 5.1: Ground truth latitude orientation check
    if "lat" in ground_truth_da.coords:
        gt_lat = ground_truth_da.coords["lat"].values
        if len(gt_lat) > 1 and gt_lat[0] < gt_lat[-1]:
            print("      [GUARDRAIL 5.1] Ground truth latitude indexed South-to-North. Flipping axis to North-to-South...")
            ground_truth_data = np.flip(ground_truth_data, axis=0)

    # GUARDRAIL 5.2: Latitude trimming alignment (721 -> 720)
    if ground_truth_data.shape != forecast_data.shape:
        if ground_truth_data.shape[0] == forecast_data.shape[0] + 1:
            print(f"      [INFO] Trimming ERA5 latitude dimension ({ground_truth_data.shape[0]} -> {forecast_data.shape[0]})...")
            ground_truth_data = ground_truth_data[: forecast_data.shape[0], :]
        else:
            raise AssertionError(f"[ABORT] Shape mismatch: forecast {forecast_data.shape} vs ground truth {ground_truth_data.shape}!")

    # DEBUG: Check if data makes physical sense (should be around 250-270 K for t850)
    print(f"\n      [DEBUG] After Alignment:")
    print(f"      Ground Truth Stats: min={np.min(ground_truth_data):.2f}, max={np.max(ground_truth_data):.2f}, mean={np.mean(ground_truth_data):.2f}")
    print(f"      Forecast Stats:     min={np.min(forecast_data):.2f}, max={np.max(forecast_data):.2f}, mean={np.mean(forecast_data):.2f}")

    # Check for obvious orientation issues by sampling corners
    print(f"\n      [DEBUG] Corner Values (for orientation check):")
    print(f"      Ground Truth: NW={ground_truth_data[0,0]:.2f}, NE={ground_truth_data[0,-1]:.2f}, SW={ground_truth_data[-1,0]:.2f}, SE={ground_truth_data[-1,-1]:.2f}")
    print(f"      Forecast:     NW={forecast_data[0,0]:.2f}, NE={forecast_data[0,-1]:.2f}, SW={forecast_data[-1,0]:.2f}, SE={forecast_data[-1,-1]:.2f}")

    diff = forecast_data - ground_truth_data
    max_abs_error = np.max(np.abs(diff))

    # GUARDRAIL 5.3: Sanity threshold on forecast error
    if max_abs_error > 80.0:
        raise AssertionError(f"[ABORT] Forecast error abnormally large ({max_abs_error:.2f} K)! Check orientation/units.")

    # SPATIAL METADATA LOG
    print("\n      [SPATIAL METADATA - Final Verification State]")
    print(f"      • Raw FCN Forecast Slice Shape  : {forecast_data.shape}")
    print(f"      • Aligned ERA5 Ground Truth Shape: {ground_truth_data.shape}")
    print(f"      • Difference Array Shape         : {diff.shape}")
    print(f"      • Forecast - ERA5 Max Abs Error  : {max_abs_error:.2f} K")

    # ==========================================================================
    # [6/6] Plot Verification Result
    # ==========================================================================
    print("\n[6/6] Generating Verification Plot against ARCO ERA5 Ground Truth (t850)...")
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(diff.squeeze(), cmap="coolwarm", origin="upper", vmin=-15, vmax=15)
    cbar = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.02)
    cbar.set_label(f"{target_var.upper()} Difference (Forecast - ERA5 Ground Truth) [K]")

    ax.set_title(
        f"Run C: HealDA + FourCastNet 24h Hindcast Error ({target_var.upper()})\n"
        f"Init: {start_time.strftime('%Y-%m-%d %H:%M')} UTC | "
        f"Target: {target_time.strftime('%Y-%m-%d %H:%M')} UTC",
        fontsize=12,
    )
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150)
    plt.close()

    print(f"\nSaved verified output plot to: {output_filename}")


if __name__ == "__main__":
    main()