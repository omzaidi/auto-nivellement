from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from level_config import (
    ELEMENT,
    IMPUTED_VALUE_COLUMN,
    PHASE3_INPUT_SHP,
    PHASE3_OUTPUT_TIF,
    PHASE3_VARIOGRAM_DIAGNOSTIC_DIR,
)
from level_core import make_progress

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None

try:
    from scipy.optimize import least_squares
except ImportError:
    least_squares = None


# -------------------------- User-editable settings --------------------------
# Main file/element settings live in level_config.py.
INPUT_SHP = PHASE3_INPUT_SHP
OUTPUT_TIF = PHASE3_OUTPUT_TIF

VALUE_COLUMN = IMPUTED_VALUE_COLUMN
PHASE2_STATUS_COLUMN = "P2_STAT"
FILTER_EXCLUDED_PHASE2_POINTS = True
PHASE2_EXCLUDED_STATUS_PREFIXES = ("excluded_",)

# Grid settings (in target CRS units, meters if projected)
TARGET_CRS = None  # e.g. "EPSG:32198"; None => keep projected CRS or auto-UTM
PIXEL_SIZE = 1_000.0

# Ordinary kriging settings
# A log transform is usually gentler for positive geochemistry values.
LOG_TRANSFORM_VALUES = True
KRIGING_K_NEIGHBORS = 24
KRIGING_MIN_NEIGHBORS = 0
KRIGING_MAX_DISTANCE = 5_000.0  # meters; set <=0 for unlimited
QUERY_CHUNK_SIZE = 10_000
MAX_GRID_CELLS = 5_000_000

# Automatic spherical variogram estimation
VARIOGRAM_SAMPLE_POINTS = 5_000
VARIOGRAM_PAIR_COUNT = 80_000
VARIOGRAM_RANDOM_SEED = 42
VARIOGRAM_NUGGET_FRACTION = 0.05
VARIOGRAM_FIT_MAX_LAG = 10_000.0
VARIOGRAM_FIT_STATISTIC = "median"  # "median" or "mean"
VARIOGRAM_MIN_PAIRS_PER_BIN = 30
SAVE_VARIOGRAM_DIAGNOSTICS = True
VARIOGRAM_DIAGNOSTIC_DIR = PHASE3_VARIOGRAM_DIAGNOSTIC_DIR
VARIOGRAM_BIN_COUNT = 40
VARIOGRAM_PLOT_SAMPLE_PAIRS = 10_000

# Numerical stabilizer for local kriging systems
KRIGING_MATRIX_JITTER = 1e-10

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


def prepare_points(
    gdf: gpd.GeoDataFrame,
) -> tuple[np.ndarray, np.ndarray, gpd.GeoDataFrame]:
    if VALUE_COLUMN not in gdf.columns:
        raise KeyError(f"Missing required column: {VALUE_COLUMN}")

    vals = pd.to_numeric(gdf[VALUE_COLUMN], errors="coerce").to_numpy(dtype=float)
    geom = gdf.geometry

    # Expected input is point samples. For non-point geometries, use representative points.
    if (geom.geom_type == "Point").all():
        px = geom.x.to_numpy(dtype=float)
        py = geom.y.to_numpy(dtype=float)
    else:
        reps = geom.representative_point()
        px = reps.x.to_numpy(dtype=float)
        py = reps.y.to_numpy(dtype=float)

    mask = np.isfinite(vals) & np.isfinite(px) & np.isfinite(py)
    if LOG_TRANSFORM_VALUES:
        mask &= vals > 0
    if not np.any(mask):
        raise ValueError("No finite interpolation points found after filtering.")

    clean = gdf.iloc[np.flatnonzero(mask)].copy()
    points = np.column_stack([px[mask], py[mask]])
    values = vals[mask]

    if LOG_TRANSFORM_VALUES:
        values = np.log(values)
        print("Using natural-log transformed values for kriging.")

    points, values = aggregate_duplicate_points(points, values)
    return points, values, clean


def aggregate_duplicate_points(
    points_xy: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    df = pd.DataFrame({"x": points_xy[:, 0], "y": points_xy[:, 1], "z": values})
    grouped = df.groupby(["x", "y"], sort=False, as_index=False)["z"].mean()
    removed = len(df) - len(grouped)
    if removed:
        print(f"Aggregated {removed} duplicate-coordinate samples by mean value.")
    return grouped[["x", "y"]].to_numpy(dtype=float), grouped["z"].to_numpy(dtype=float)


def compute_grid(bounds: tuple[float, float, float, float]) -> tuple[int, int, object]:
    minx, miny, maxx, maxy = bounds
    if not np.isfinite([minx, miny, maxx, maxy]).all():
        raise ValueError("Invalid bounds for interpolation grid.")
    if maxx <= minx or maxy <= miny:
        raise ValueError("Degenerate bounds; cannot build raster grid.")

    ncols = int(math.ceil((maxx - minx) / PIXEL_SIZE))
    nrows = int(math.ceil((maxy - miny) / PIXEL_SIZE))
    total_cells = nrows * ncols
    if total_cells > MAX_GRID_CELLS:
        raise ValueError(
            f"Grid would contain {total_cells:,} cells at PIXEL_SIZE={PIXEL_SIZE:g}. "
            f"Increase PIXEL_SIZE or MAX_GRID_CELLS before running kriging."
        )
    transform = from_origin(minx, maxy, PIXEL_SIZE, PIXEL_SIZE)
    return nrows, ncols, transform


def spherical_semivariogram(
    distance: np.ndarray,
    *,
    nugget: float,
    sill: float,
    range_: float,
) -> np.ndarray:
    h = np.asarray(distance, dtype=float)
    hr = np.clip(h / max(range_, 1e-12), 0.0, None)
    partial_sill = max(sill - nugget, 0.0)
    gamma = np.where(
        h <= 0,
        0.0,
        np.where(
            hr < 1.0,
            nugget + partial_sill * (1.5 * hr - 0.5 * hr**3),
            sill,
        ),
    )
    return gamma


def build_variogram_bins(
    distances: np.ndarray,
    semivariance: np.ndarray,
    *,
    bin_count: int,
    max_lag: float | None = None,
) -> pd.DataFrame:
    if distances.size == 0:
        return pd.DataFrame(
            columns=[
                "bin",
                "distance_min",
                "distance_max",
                "distance_mid",
                "pair_count",
                "semivariance_mean",
                "semivariance_median",
            ]
        )

    max_distance = (
        float(max_lag) if max_lag is not None else float(np.nanmax(distances))
    )
    edges = np.linspace(0.0, max_distance, max(2, int(bin_count) + 1))
    rows: list[dict] = []
    for bin_idx in range(len(edges) - 1):
        lo = edges[bin_idx]
        hi = edges[bin_idx + 1]
        if bin_idx == len(edges) - 2:
            keep = (distances >= lo) & (distances <= hi)
        else:
            keep = (distances >= lo) & (distances < hi)

        vals = semivariance[keep]
        rows.append(
            {
                "bin": bin_idx + 1,
                "distance_min": lo,
                "distance_max": hi,
                "distance_mid": 0.5 * (lo + hi),
                "pair_count": int(vals.size),
                "semivariance_mean": float(np.nanmean(vals)) if vals.size else np.nan,
                "semivariance_median": (
                    float(np.nanmedian(vals)) if vals.size else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def sample_variogram_pairs(
    points_xy: np.ndarray,
    values: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int]:
    fit_max_lag = float(VARIOGRAM_FIT_MAX_LAG)
    if fit_max_lag <= 0:
        raise ValueError("VARIOGRAM_FIT_MAX_LAG must be positive.")

    tree = cKDTree(points_xy)
    pair_idx = tree.query_pairs(fit_max_lag, output_type="ndarray")
    available_pairs = int(len(pair_idx))
    if available_pairs == 0:
        raise ValueError(
            f"No point pairs found within VARIOGRAM_FIT_MAX_LAG={fit_max_lag:g}."
        )

    pair_count = min(int(VARIOGRAM_PAIR_COUNT), available_pairs)
    if available_pairs > pair_count:
        keep_idx = rng.choice(available_pairs, size=pair_count, replace=False)
        pair_idx = pair_idx[keep_idx]

    i = pair_idx[:, 0]
    j = pair_idx[:, 1]
    distances = np.linalg.norm(points_xy[i] - points_xy[j], axis=1)
    semivariance = 0.5 * (values[i] - values[j]) ** 2
    keep = np.isfinite(distances) & np.isfinite(semivariance) & (distances > 0)
    return distances[keep], semivariance[keep], available_pairs


def _initial_variogram_parameters(
    bins: pd.DataFrame,
    values: np.ndarray,
    y_column: str,
) -> tuple[float, float, float]:
    fit_bins = bins[
        (bins["pair_count"] >= VARIOGRAM_MIN_PAIRS_PER_BIN)
        & np.isfinite(bins[y_column])
    ]
    if fit_bins.empty:
        sill = float(np.nanvar(values))
        nugget = float(np.clip(0.05 * sill, 0.0, VARIOGRAM_NUGGET_FRACTION * sill))
        return nugget, sill, float(0.5 * VARIOGRAM_FIT_MAX_LAG)

    y = fit_bins[y_column].to_numpy(dtype=float)
    h = fit_bins["distance_mid"].to_numpy(dtype=float)
    sill = float(np.nanquantile(y, 0.80))
    if not np.isfinite(sill) or sill <= 0:
        sill = float(np.nanvar(values))
    nugget = float(y[0])
    nugget = float(np.clip(nugget, 0.0, VARIOGRAM_NUGGET_FRACTION * sill))

    target = nugget + 0.95 * max(sill - nugget, 0.0)
    reached = h[y >= target]
    if reached.size:
        range_ = float(reached[0])
    else:
        range_ = float(np.nanmedian(h))
    if not np.isfinite(range_) or range_ <= 0:
        range_ = float(0.5 * VARIOGRAM_FIT_MAX_LAG)
    return nugget, sill, range_


def fit_spherical_variogram_to_bins(
    bins: pd.DataFrame,
    values: np.ndarray,
) -> tuple[float, float, float, pd.DataFrame]:
    y_column = (
        "semivariance_median"
        if VARIOGRAM_FIT_STATISTIC == "median"
        else "semivariance_mean"
    )
    fit_bins = bins[
        (bins["pair_count"] >= VARIOGRAM_MIN_PAIRS_PER_BIN)
        & np.isfinite(bins[y_column])
    ].copy()
    if len(fit_bins) < 3 or least_squares is None:
        nugget, sill, range_ = _initial_variogram_parameters(bins, values, y_column)
        fit_bins["used_for_fit"] = True
        return nugget, sill, range_, fit_bins

    h = fit_bins["distance_mid"].to_numpy(dtype=float)
    y = fit_bins[y_column].to_numpy(dtype=float)
    weights = np.sqrt(
        fit_bins["pair_count"].to_numpy(dtype=float)
        / max(float(np.nanmedian(fit_bins["pair_count"])), 1.0)
    )

    nugget0, sill0, range0 = _initial_variogram_parameters(fit_bins, values, y_column)
    partial0 = max(sill0 - nugget0, 1e-12)
    max_y = max(float(np.nanmax(y)), float(np.nanvar(values)), 1e-12)
    min_range = max(float(np.nanmin(h[h > 0])) * 0.25, 1.0)
    max_range = max(float(VARIOGRAM_FIT_MAX_LAG) * 2.0, min_range * 2.0)

    def residual(params: np.ndarray) -> np.ndarray:
        nugget, partial_sill, range_ = params
        model = spherical_semivariogram(
            h,
            nugget=nugget,
            sill=nugget + partial_sill,
            range_=range_,
        )
        return (model - y) * weights

    result = least_squares(
        residual,
        x0=np.array([nugget0, partial0, range0], dtype=float),
        bounds=(
            np.array([0.0, 1e-12, min_range], dtype=float),
            np.array([max_y, max_y * 5.0, max_range], dtype=float),
        ),
    )
    nugget, partial_sill, range_ = result.x
    sill = nugget + partial_sill
    fit_bins["used_for_fit"] = True
    return float(nugget), float(sill), float(range_), fit_bins


def save_variogram_diagnostics(
    distances: np.ndarray,
    semivariance: np.ndarray,
    *,
    nugget: float,
    sill: float,
    range_: float,
    bins: pd.DataFrame,
) -> None:
    if not SAVE_VARIOGRAM_DIAGNOSTICS:
        return

    VARIOGRAM_DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)

    bins_path = VARIOGRAM_DIAGNOSTIC_DIR / "empirical_variogram_bins.csv"
    bins.to_csv(bins_path, index=False)

    params_path = VARIOGRAM_DIAGNOSTIC_DIR / "fitted_variogram_parameters.csv"
    pd.DataFrame(
        [
            {
                "model": "spherical",
                "nugget": nugget,
                "sill": sill,
                "range": range_,
                "sampled_pair_count": int(distances.size),
                "log_transform_values": LOG_TRANSFORM_VALUES,
                "fit_max_lag": VARIOGRAM_FIT_MAX_LAG,
                "fit_statistic": VARIOGRAM_FIT_STATISTIC,
                "min_pairs_per_bin": VARIOGRAM_MIN_PAIRS_PER_BIN,
            }
        ]
    ).to_csv(params_path, index=False)

    if plt is None:
        print(
            "Saved variogram CSV diagnostics, but matplotlib is not available "
            "so no variogram plot was created."
        )
        return

    rng = np.random.default_rng(VARIOGRAM_RANDOM_SEED)
    plot_count = min(int(VARIOGRAM_PLOT_SAMPLE_PAIRS), distances.size)
    if plot_count < distances.size:
        plot_idx = rng.choice(distances.size, size=plot_count, replace=False)
    else:
        plot_idx = np.arange(distances.size)

    x_model = np.linspace(0.0, float(np.nanmax(distances)), 400)
    y_model = spherical_semivariogram(
        x_model,
        nugget=nugget,
        sill=sill,
        range_=range_,
    )

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(
        distances[plot_idx],
        semivariance[plot_idx],
        s=5,
        alpha=0.12,
        color="#4c78a8",
        label="sampled pairs",
    )
    good_bins = bins["pair_count"] > 0
    y_column = (
        "semivariance_median"
        if VARIOGRAM_FIT_STATISTIC == "median"
        else "semivariance_mean"
    )
    ax.scatter(
        bins.loc[good_bins, "distance_mid"],
        bins.loc[good_bins, y_column],
        s=42,
        color="#f58518",
        label=f"binned {VARIOGRAM_FIT_STATISTIC}",
        zorder=3,
    )
    ax.plot(x_model, y_model, color="#d62728", lw=2.0, label="spherical fit")
    ax.axhline(sill, color="black", lw=1.0, ls="--", alpha=0.5, label="sill")
    ax.axvline(range_, color="black", lw=1.0, ls=":", alpha=0.5, label="range")
    ax.set_title(
        f"{ELEMENT} phase 3 variogram | nugget={nugget:.3g}, "
        f"sill={sill:.3g}, range={range_:.3g}"
    )
    ax.set_xlabel("Distance")
    ax.set_ylabel("Semivariance")
    ax.set_yscale("log")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()

    plot_path = VARIOGRAM_DIAGNOSTIC_DIR / "fitted_variogram.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved variogram diagnostics: {bins_path}, {params_path}, {plot_path}")


def estimate_spherical_variogram(
    points_xy: np.ndarray,
    values: np.ndarray,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(VARIOGRAM_RANDOM_SEED)
    n = len(values)
    if n < 3:
        raise ValueError("At least 3 points are required for ordinary kriging.")

    sample_n = min(n, int(VARIOGRAM_SAMPLE_POINTS))
    sample_idx = rng.choice(n, size=sample_n, replace=False)
    pts = points_xy[sample_idx]
    vals = values[sample_idx]

    distances, semivariance, available_pairs = sample_variogram_pairs(pts, vals, rng)
    if distances.size == 0:
        raise ValueError("Could not estimate a variogram from the input points.")

    if not np.isfinite(np.nanvar(values)) or float(np.nanvar(values)) <= 0:
        raise ValueError("Input values have no variance; kriging cannot be fit.")

    bins = build_variogram_bins(
        distances,
        semivariance,
        bin_count=VARIOGRAM_BIN_COUNT,
        max_lag=VARIOGRAM_FIT_MAX_LAG,
    )
    nugget, sill, range_, fit_bins = fit_spherical_variogram_to_bins(bins, values)
    bins["used_for_fit"] = (
        bins["pair_count"] >= VARIOGRAM_MIN_PAIRS_PER_BIN
    ) & np.isfinite(
        bins[
            (
                "semivariance_median"
                if VARIOGRAM_FIT_STATISTIC == "median"
                else "semivariance_mean"
            )
        ]
    )

    print(
        "Estimated spherical variogram: "
        f"sample_points={sample_n}, local_pairs={distances.size}/{available_pairs}, "
        f"fit_max_lag={VARIOGRAM_FIT_MAX_LAG:.6g}, "
        f"fit_bins={len(fit_bins)}, statistic={VARIOGRAM_FIT_STATISTIC}, "
        f"nugget={nugget:.6g}, sill={sill:.6g}, range={range_:.6g}"
    )
    save_variogram_diagnostics(
        distances,
        semivariance,
        nugget=nugget,
        sill=sill,
        range_=range_,
        bins=bins,
    )
    return nugget, sill, range_


def ordinary_kriging_predict_one(
    query_xy: np.ndarray,
    neighbor_xy: np.ndarray,
    neighbor_values: np.ndarray,
    *,
    nugget: float,
    sill: float,
    range_: float,
) -> float:
    n = len(neighbor_values)
    if n == 0:
        return np.nan
    if n == 1:
        return float(neighbor_values[0])

    neighbor_dist = np.linalg.norm(
        neighbor_xy[:, None, :] - neighbor_xy[None, :, :],
        axis=2,
    )
    query_dist = np.linalg.norm(neighbor_xy - query_xy[None, :], axis=1)

    matrix = np.empty((n + 1, n + 1), dtype=float)
    matrix[:n, :n] = spherical_semivariogram(
        neighbor_dist,
        nugget=nugget,
        sill=sill,
        range_=range_,
    )
    matrix[:n, n] = 1.0
    matrix[n, :n] = 1.0
    matrix[n, n] = 0.0
    matrix[:n, :n] += np.eye(n) * KRIGING_MATRIX_JITTER

    rhs = np.empty(n + 1, dtype=float)
    rhs[:n] = spherical_semivariogram(
        query_dist,
        nugget=nugget,
        sill=sill,
        range_=range_,
    )
    rhs[n] = 1.0

    try:
        weights = np.linalg.solve(matrix, rhs)[:n]
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(matrix, rhs, rcond=None)[0][:n]

    return float(np.dot(weights, neighbor_values))


def ordinary_kriging_predict_chunk(
    tree: cKDTree,
    points_xy: np.ndarray,
    values: np.ndarray,
    query_points: np.ndarray,
    *,
    nugget: float,
    sill: float,
    range_: float,
) -> np.ndarray:
    k = max(1, int(KRIGING_K_NEIGHBORS))
    k = min(k, len(values))
    dist_upper = float(KRIGING_MAX_DISTANCE) if KRIGING_MAX_DISTANCE > 0 else np.inf

    distances, indices = tree.query(query_points, k=k, distance_upper_bound=dist_upper)
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    out = np.full(query_points.shape[0], np.nan, dtype=float)
    for row_idx in range(query_points.shape[0]):
        valid = np.isfinite(distances[row_idx]) & (indices[row_idx] < len(values))
        if int(np.sum(valid)) < KRIGING_MIN_NEIGHBORS:
            continue

        # Exact hits take the exact sampled value.
        exact = valid & (distances[row_idx] == 0.0)
        if np.any(exact):
            out[row_idx] = values[indices[row_idx][np.flatnonzero(exact)[0]]]
            continue

        ii = indices[row_idx][valid]
        out[row_idx] = ordinary_kriging_predict_one(
            query_points[row_idx],
            points_xy[ii],
            values[ii],
            nugget=nugget,
            sill=sill,
            range_=range_,
        )
    return out


def back_transform(values: np.ndarray) -> np.ndarray:
    if not LOG_TRANSFORM_VALUES:
        return values
    return np.exp(values)


def run_interpolation() -> None:
    if cKDTree is None:
        raise ImportError(
            "scipy is required for phase 3 ordinary kriging (cKDTree). "
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
    print(f"Kriging points: {len(vals)}")

    nrows, ncols, transform = compute_grid(tuple(clean.total_bounds))
    total_cells = nrows * ncols
    print(f"Grid: {nrows} x {ncols} ({total_cells} cells), pixel={PIXEL_SIZE}")

    nugget, sill, range_ = estimate_spherical_variogram(pts_xy, vals)
    tree = cKDTree(pts_xy)
    raster_flat = np.full(total_cells, NODATA_VALUE, dtype=np.float32)

    progress = make_progress(total_cells, "Phase 3 ordinary kriging", unit="cell")

    minx, miny, maxx, maxy = clean.total_bounds
    for start in range(0, total_cells, QUERY_CHUNK_SIZE):
        end = min(total_cells, start + QUERY_CHUNK_SIZE)
        idx = np.arange(start, end, dtype=np.int64)

        row = idx // ncols
        col = idx % ncols
        xq = minx + (col + 0.5) * PIXEL_SIZE
        yq = maxy - (row + 0.5) * PIXEL_SIZE
        qpts = np.column_stack([xq, yq])

        pred = ordinary_kriging_predict_chunk(
            tree,
            pts_xy,
            vals,
            qpts,
            nugget=nugget,
            sill=sill,
            range_=range_,
        )
        pred = back_transform(pred)
        chunk = np.where(np.isfinite(pred), pred, NODATA_VALUE).astype(np.float32)
        raster_flat[start:end] = chunk

        progress.update(end - start)

    progress.close()

    raster = raster_flat.reshape((nrows, ncols))

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
