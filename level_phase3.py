from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from level_core import make_progress

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None

try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    gaussian_filter = None

# -------------------------- User-editable settings --------------------------
# INPUT_SHP = Path("output/phase2_leveled_partial_overlap.shp")
INPUT_SHP = Path("shp/AG_Fusionn_imp.shp")
OUTPUT_TIF = Path("output/phase3_ag_imp_idw_pre.tif")

VALUE_COLUMN = "Ag_imp"
PHASE2_STATUS_COLUMN = "P2_STAT"
FILTER_EXCLUDED_PHASE2_POINTS = True
PHASE2_EXCLUDED_STATUS_PREFIXES = ("excluded_",)

# Grid settings (in target CRS units, meters if projected)
TARGET_CRS = None  # e.g. "EPSG:32198"; None => keep projected CRS or auto-UTM
PIXEL_SIZE = 250.0

# IDW settings
# Set AUTO_TUNE_IDW_PARAMS=True to derive parameters from point-spacing statistics.
AUTO_TUNE_IDW_PARAMS = False
IDW_POWER = 1.0
IDW_K_NEIGHBORS = 12
IDW_MAX_DISTANCE = 20_000.0  # meters; set <=0 for unlimited
QUERY_CHUNK_SIZE = 100_000

# Optional raster smoothing to reduce point-centric speckles.
# Set to 0 to disable.
POST_SMOOTH_SIGMA_PIXELS = 1.0

# Output settings
NODATA_VALUE = -9999.0
COMPRESS = "lzw"
# ---------------------------------------------------------------------------


def choose_target_crs(gdf: gpd.GeoDataFrame):
    if TARGET_CRS:
        return TARGET_CRS
    if gdf.crs is None:
        raise ValueError("Input shapefile has no CRS. Set TARGET_CRS explicitly.")
    if getattr(gdf.crs, "is_geographic", False):
        utm = gdf.estimate_utm_crs()
        if utm is None:
            raise ValueError(
                "Could not infer projected CRS from geographic input. Set TARGET_CRS."
            )
        return utm
    return gdf.crs


def prepare_points(
    gdf: gpd.GeoDataFrame,
) -> tuple[np.ndarray, np.ndarray, gpd.GeoDataFrame]:
    if VALUE_COLUMN not in gdf.columns:
        raise KeyError(f"Missing required column: {VALUE_COLUMN}")

    vals = pd.to_numeric(gdf[VALUE_COLUMN], errors="coerce")
    geom = gdf.geometry

    # Expected input is point samples. For non-point geometries, use representative points.
    if (geom.geom_type == "Point").all():
        px = geom.x.to_numpy(dtype=float)
        py = geom.y.to_numpy(dtype=float)
    else:
        reps = geom.representative_point()
        px = reps.x.to_numpy(dtype=float)
        py = reps.y.to_numpy(dtype=float)

    mask = np.isfinite(vals.to_numpy(dtype=float)) & np.isfinite(px) & np.isfinite(py)
    if not np.any(mask):
        raise ValueError("No finite interpolation points found after filtering.")

    x = px[mask]
    y = py[mask]
    z = vals.to_numpy(dtype=float)[mask]
    # `mask` is positional; use iloc to avoid label-based KeyError on non-default indexes.
    clean = gdf.iloc[np.flatnonzero(mask)].copy()

    return np.column_stack([x, y]), z, clean


def filter_phase2_excluded_points(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if not FILTER_EXCLUDED_PHASE2_POINTS:
        return gdf

    if PHASE2_STATUS_COLUMN not in gdf.columns:
        print(
            f"{PHASE2_STATUS_COLUMN} not found; skipping exclusion filter in phase 3."
        )
        return gdf

    status = gdf[PHASE2_STATUS_COLUMN].fillna("").astype(str)
    excluded = np.zeros(len(gdf), dtype=bool)
    for prefix in PHASE2_EXCLUDED_STATUS_PREFIXES:
        excluded |= status.str.startswith(prefix)

    n_excluded = int(excluded.sum())
    if n_excluded > 0:
        print(
            f"Phase-2 exclusion filter: removed {n_excluded} points "
            f"(prefixes={PHASE2_EXCLUDED_STATUS_PREFIXES})"
        )

    kept = gdf.iloc[np.flatnonzero(~excluded)].copy()
    if kept.empty:
        raise ValueError(
            "No points left after phase-2 exclusion filtering; cannot interpolate."
        )
    return kept


def compute_grid(bounds: tuple[float, float, float, float]) -> tuple[int, int, object]:
    minx, miny, maxx, maxy = bounds
    if not np.isfinite([minx, miny, maxx, maxy]).all():
        raise ValueError("Invalid bounds for interpolation grid.")
    if maxx <= minx or maxy <= miny:
        raise ValueError("Degenerate bounds; cannot build raster grid.")

    ncols = int(math.ceil((maxx - minx) / PIXEL_SIZE))
    nrows = int(math.ceil((maxy - miny) / PIXEL_SIZE))
    transform = from_origin(minx, maxy, PIXEL_SIZE, PIXEL_SIZE)
    return nrows, ncols, transform


def idw_predict_chunk(
    tree: cKDTree,
    values: np.ndarray,
    query_points: np.ndarray,
    idw_power: float,
    idw_k_neighbors: int,
    idw_max_distance: float,
) -> np.ndarray:
    k = max(1, int(idw_k_neighbors))
    dist_upper = float(idw_max_distance) if idw_max_distance > 0 else np.inf

    dist, ind = tree.query(query_points, k=k, distance_upper_bound=dist_upper)

    if k == 1:
        dist = dist[:, None]
        ind = ind[:, None]

    valid = np.isfinite(dist)
    out = np.full(query_points.shape[0], np.nan, dtype=float)

    any_valid = valid.any(axis=1)
    if not np.any(any_valid):
        return out

    # Exact hits take the exact sampled value.
    zero_mask = valid & (dist == 0.0)
    exact_rows = zero_mask.any(axis=1)
    if np.any(exact_rows):
        zr = np.where(exact_rows)[0]
        zc = np.argmax(zero_mask[zr], axis=1)
        out[zr] = values[ind[zr, zc]]

    # For remaining rows, use weighted average of finite neighbors.
    rem = (~exact_rows) & any_valid
    if np.any(rem):
        d = dist[rem]
        ii = ind[rem]
        v = valid[rem]

        w = np.zeros_like(d, dtype=float)
        w[v] = 1.0 / np.power(d[v], idw_power)

        # tree.query uses index == len(values) for invalid neighbors.
        padded_values = np.concatenate([values, np.array([0.0], dtype=float)])
        z = padded_values[ii]

        num = np.sum(w * z, axis=1)
        den = np.sum(w, axis=1)

        good = den > 0
        pred = np.full(rem.sum(), np.nan, dtype=float)
        pred[good] = num[good] / den[good]
        out[rem] = pred

    return out


def auto_tune_idw_params(
    tree: cKDTree,
    points_xy: np.ndarray,
) -> tuple[float, int, float]:
    n = points_xy.shape[0]
    if n < 10:
        return IDW_POWER, max(3, min(IDW_K_NEIGHBORS, n)), IDW_MAX_DISTANCE

    # Nearest-neighbor spacing (excluding self).
    d2, _ = tree.query(points_xy, k=2)
    nn = d2[:, 1]
    nn = nn[np.isfinite(nn) & (nn > 0)]
    if nn.size == 0:
        return IDW_POWER, IDW_K_NEIGHBORS, IDW_MAX_DISTANCE

    p10, p50, p90 = np.quantile(nn, [0.10, 0.50, 0.90])
    hetero_ratio = float(p90 / max(p10, 1e-9))

    # Higher heterogeneity => use more neighbors and lower power for smoother fields.
    if hetero_ratio < 2.0:
        k_auto = 16
        power_auto = 1.6
    elif hetero_ratio < 4.0:
        k_auto = 22
        power_auto = 1.4
    elif hetero_ratio < 8.0:
        k_auto = 28
        power_auto = 1.25
    else:
        k_auto = 36
        power_auto = 1.1

    # Scale k slightly with dataset size but cap for runtime.
    if n > 500_000:
        k_auto += 8
    elif n > 200_000:
        k_auto += 4
    k_auto = int(np.clip(k_auto, 12, 48))
    k_auto = int(min(k_auto, n - 1))

    # Max distance based on k-neighbor reach in sparse zones, with a generous floor.
    dk, _ = tree.query(points_xy, k=k_auto + 1)
    d_k = dk[:, -1]
    d_k = d_k[np.isfinite(d_k) & (d_k > 0)]
    if d_k.size == 0:
        max_dist_auto = float(max(IDW_MAX_DISTANCE, 6.0 * p50))
    else:
        max_dist_auto = float(np.quantile(d_k, 0.99))
        max_dist_auto = max(max_dist_auto, 6.0 * float(p50))

    print(
        "Auto-tuned IDW from point spacing: "
        f"n={n}, nn_p10={p10:.3g}, nn_p50={p50:.3g}, nn_p90={p90:.3g}, "
        f"heterogeneity_ratio={hetero_ratio:.3g}, "
        f"power={power_auto:.3g}, k={k_auto}, max_distance={max_dist_auto:.3g}"
    )
    return power_auto, k_auto, max_dist_auto


def smooth_raster_ignore_nodata(
    raster: np.ndarray,
    nodata_value: float,
    sigma_pixels: float,
) -> np.ndarray:
    if sigma_pixels <= 0:
        return raster
    if gaussian_filter is None:
        print("scipy.ndimage not available; skipping raster smoothing.")
        return raster

    valid = raster != nodata_value
    if not np.any(valid):
        return raster

    data = np.where(valid, raster, 0.0).astype(np.float64)
    weight = valid.astype(np.float64)

    data_s = gaussian_filter(data, sigma=sigma_pixels, mode="nearest")
    weight_s = gaussian_filter(weight, sigma=sigma_pixels, mode="nearest")

    out = np.full_like(raster, nodata_value, dtype=np.float32)
    good = valid & (weight_s > 1e-9)
    out[good] = (data_s[good] / weight_s[good]).astype(np.float32)
    return out


def run_interpolation() -> None:
    if cKDTree is None:
        raise ImportError(
            "scipy is required for phase 3 interpolation (cKDTree). "
            "Install scipy and rerun."
        )

    print(f"Reading leveled shapefile: {INPUT_SHP}")
    gdf = gpd.read_file(INPUT_SHP)
    print(f"Rows: {len(gdf)}")
    gdf = filter_phase2_excluded_points(gdf)
    print(f"Rows after phase-2 exclusion filter: {len(gdf)}")

    target_crs = choose_target_crs(gdf)
    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)
        print(f"Reprojected to {target_crs}")
    else:
        print(f"Using CRS: {target_crs}")

    pts_xy, vals, clean = prepare_points(gdf)
    print(f"Interpolation points: {len(vals)}")

    nrows, ncols, transform = compute_grid(tuple(clean.total_bounds))
    total_cells = nrows * ncols
    print(f"Grid: {nrows} x {ncols} ({total_cells} cells), pixel={PIXEL_SIZE}")

    tree = cKDTree(pts_xy)
    if AUTO_TUNE_IDW_PARAMS:
        idw_power, idw_k_neighbors, idw_max_distance = auto_tune_idw_params(
            tree,
            pts_xy,
        )
    else:
        idw_power = float(IDW_POWER)
        idw_k_neighbors = int(IDW_K_NEIGHBORS)
        idw_max_distance = float(IDW_MAX_DISTANCE)
        print(
            "Using manual IDW parameters: "
            f"power={idw_power:.3g}, k={idw_k_neighbors}, max_distance={idw_max_distance:.3g}"
        )

    raster_flat = np.full(total_cells, NODATA_VALUE, dtype=np.float32)

    progress = make_progress(total_cells, "Phase 3 interpolation", unit="cell")

    minx, miny, maxx, maxy = clean.total_bounds
    for start in range(0, total_cells, QUERY_CHUNK_SIZE):
        end = min(total_cells, start + QUERY_CHUNK_SIZE)
        idx = np.arange(start, end, dtype=np.int64)

        row = idx // ncols
        col = idx % ncols
        xq = minx + (col + 0.5) * PIXEL_SIZE
        yq = maxy - (row + 0.5) * PIXEL_SIZE
        qpts = np.column_stack([xq, yq])

        pred = idw_predict_chunk(
            tree,
            vals,
            qpts,
            idw_power=idw_power,
            idw_k_neighbors=idw_k_neighbors,
            idw_max_distance=idw_max_distance,
        )
        chunk = np.where(np.isfinite(pred), pred, NODATA_VALUE).astype(np.float32)
        raster_flat[start:end] = chunk

        progress.update(end - start)

    progress.close()

    raster = raster_flat.reshape((nrows, ncols))
    if POST_SMOOTH_SIGMA_PIXELS > 0:
        print(f"Applying post-smoothing: sigma={POST_SMOOTH_SIGMA_PIXELS} px")
        raster = smooth_raster_ignore_nodata(
            raster,
            nodata_value=NODATA_VALUE,
            sigma_pixels=POST_SMOOTH_SIGMA_PIXELS,
        )

    OUTPUT_TIF.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": nrows,
        "width": ncols,
        "count": 1,
        "dtype": "float32",
        "crs": target_crs,
        "transform": transform,
        "nodata": float(NODATA_VALUE),
        "compress": COMPRESS,
    }

    with rasterio.open(OUTPUT_TIF, "w", **profile) as dst:
        dst.write(raster, 1)

    valid = raster[raster != NODATA_VALUE]
    if valid.size:
        print(
            f"Saved {OUTPUT_TIF} | valid cells={valid.size}, "
            f"min={float(valid.min()):.6g}, max={float(valid.max()):.6g}"
        )
    else:
        print(f"Saved {OUTPUT_TIF} | no valid cells produced")


def main() -> None:
    run_interpolation()


if __name__ == "__main__":
    main()
