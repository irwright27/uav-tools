from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import csv
import json
import math
import shutil
import subprocess

import geopandas as gpd
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


# ============================================================
# Metadata helpers
# ============================================================

def fallback_capture_id(path: Path) -> str:
    """
    Infer a MicaSense capture ID from the filename if CaptureId metadata
    is unavailable.

    Example
    -------
    IMG_0123_1.tif -> IMG_0123
    """
    stem = path.stem
    parts = stem.rsplit("_", 1)

    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]

    return stem


def read_image_metadata(
    source_dirs: Sequence[str | Path],
) -> list[dict]:
    """
    Read MicaSense TIFF metadata from one or more directories using ExifTool.

    Parameters
    ----------
    source_dirs
        Directories containing raw TIFF/TIFF images.

    Returns
    -------
    list of dict
        ExifTool metadata records.
    """
    if shutil.which("exiftool") is None:
        raise RuntimeError(
            "ExifTool is not installed.\n"
            "Install it on macOS with:\n"
            "    brew install exiftool"
        )

    source_dirs = [Path(p).expanduser() for p in source_dirs]

    for directory in source_dirs:
        if not directory.exists():
            raise FileNotFoundError(
                f"Source directory does not exist:\n{directory}"
            )

    cmd = [
        "exiftool",
        "-j",
        "-n",
        "-r",
        "-ext", "tif",
        "-ext", "tiff",
        "-GPSLatitude",
        "-GPSLongitude",
        "-GPSXYAccuracy",
        "-CaptureId",
        "-BandName",
        "-RigCameraIndex",
    ]

    cmd += [str(p) for p in source_dirs]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )

    return json.loads(result.stdout)


def group_captures(
    metadata: Iterable[dict],
) -> dict[str, list[dict]]:
    """
    Group all band images belonging to the same MicaSense capture.
    """
    captures: dict[str, list[dict]] = {}

    for record in metadata:
        path = Path(record["SourceFile"])

        capture_id = record.get("CaptureId")

        if not capture_id:
            capture_id = fallback_capture_id(path)

        item = {
            "path": path,
            "capture_id": capture_id,
            "lat": record.get("GPSLatitude"),
            "lon": record.get("GPSLongitude"),
            "gps_accuracy": record.get("GPSXYAccuracy"),
            "band": str(record.get("BandName", "")),
            "rig_index": record.get("RigCameraIndex"),
        }

        captures.setdefault(capture_id, []).append(item)

    return captures


def find_band(
    items: Sequence[dict],
    band_name: str,
) -> dict | None:
    """
    Find a particular band within one MicaSense capture.

    Matching is case-insensitive and partial, so ``"panchro"`` will match
    metadata such as ``"Panchro"`` or ``"Panchromatic"``.
    """
    target = band_name.lower()

    for item in items:
        if target in item["band"].lower():
            return item

    return None


# ============================================================
# AOI helpers
# ============================================================

def load_aoi(
    aoi: str | Path | gpd.GeoDataFrame | BaseGeometry | Sequence[float],
    *,
    aoi_crs: str | int | None = None,
    layer: str | None = None,
) -> gpd.GeoDataFrame:
    """
    Load/construct an AOI.

    Parameters
    ----------
    aoi
        AOI can be any of:

        - path to a vector file (.gpkg, .shp, .geojson, etc.)
        - GeoDataFrame
        - Shapely geometry
        - bounding box: (xmin, ymin, xmax, ymax)

    aoi_crs
        Required when `aoi` is a Shapely geometry or bounding-box sequence.
        Examples:
            "EPSG:32610"
            32610
            "EPSG:4326"

    layer
        Optional layer name when reading from a multi-layer file.

    Returns
    -------
    geopandas.GeoDataFrame
    """
    if isinstance(aoi, gpd.GeoDataFrame):
        gdf = aoi.copy()

    elif isinstance(aoi, BaseGeometry):
        if aoi_crs is None:
            raise ValueError(
                "aoi_crs must be provided when AOI is a Shapely geometry."
            )

        gdf = gpd.GeoDataFrame(
            geometry=[aoi],
            crs=aoi_crs,
        )

    elif (
        isinstance(aoi, Sequence)
        and not isinstance(aoi, (str, bytes, Path))
        and len(aoi) == 4
    ):
        if aoi_crs is None:
            raise ValueError(
                "aoi_crs must be provided when AOI is a bounding box."
            )

        xmin, ymin, xmax, ymax = map(float, aoi)

        gdf = gpd.GeoDataFrame(
            geometry=[box(xmin, ymin, xmax, ymax)],
            crs=aoi_crs,
        )

    else:
        path = Path(aoi).expanduser()

        if not path.exists():
            raise FileNotFoundError(
                f"AOI file does not exist:\n{path}"
            )

        if layer is None:
            gdf = gpd.read_file(path)
        else:
            gdf = gpd.read_file(path, layer=layer)

    if len(gdf) == 0:
        raise ValueError("AOI contains no features.")

    if gdf.crs is None:
        raise ValueError(
            "AOI has no CRS. Supply/assign the correct CRS first."
        )

    return gdf


# ============================================================
# File helpers
# ============================================================

def copy_safely(
    src: Path,
    out_dir: Path,
) -> Path:
    """
    Copy a file without silently overwriting another file with the same name.
    """
    dst = out_dir / src.name

    if dst.exists():
        dst = out_dir / f"{src.parent.name}_{src.name}"

    shutil.copy2(src, dst)

    return dst


# ============================================================
# Primary public function
# ============================================================

def select_gsplat_images(
    source_dirs: Sequence[str | Path],
    aoi: str | Path | gpd.GeoDataFrame | BaseGeometry | Sequence[float],
    output_dir: str | Path,
    *,
    aoi_crs: str | int | None = None,
    aoi_layer: str | None = None,
    camera_height_m: float = 20.0,
    hfov_deg: float = 46.0,
    vfov_deg: float = 35.0,
    extra_buffer_m: float = 2.0,
    aoi_buffer_m: float = 0.0,
    representative_band: str = "panchro",
    copy_mode: str = "panchro",
    copy_files: bool = True,
    recursive: bool = True,
    write_csv: bool = True,
    write_review_gpkg: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Select UAV images whose approximate camera footprint intersects an AOI.

    The selection is intentionally conservative. Each image footprint is
    represented by a circular envelope whose radius is the half-diagonal
    of the theoretical image footprint plus an optional safety margin and
    reported GPS uncertainty.

    Parameters
    ----------
    source_dirs
        One or more directories containing raw MicaSense imagery.

    aoi
        AOI specification. May be:

        - path to vector file
        - GeoDataFrame
        - Shapely geometry
        - bounding box (xmin, ymin, xmax, ymax)

    output_dir
        Directory where selected images will be copied.

    aoi_crs
        CRS for Shapely geometry or bounding-box AOIs.

    aoi_layer
        Layer name for a multi-layer vector file.

    camera_height_m
        Approximate camera height above the target canopy/ground surface.

    hfov_deg
        Horizontal field of view of the representative camera.

    vfov_deg
        Vertical field of view of the representative camera.

    extra_buffer_m
        Additional safety radius added to the theoretical image footprint.

    aoi_buffer_m
        Expand the AOI by this distance before image selection. Useful for
        retaining neighboring orchard structure for SfM / 3DGS.

    representative_band
        Band used to represent the GPS position of each capture.

    copy_mode
        What to copy for each selected capture:

        ``"panchro"``
            Copy only the Panchro/Panchromatic band.

        ``"representative"``
            Copy only `representative_band`.

        ``"all"``
            Copy every band belonging to each selected capture.

    copy_files
        If False, calculate the selection without copying files.

    recursive
        Currently retained for API clarity. ExifTool metadata search is
        recursive.

    write_csv
        Write ``image_selection.csv`` in output_dir.parent.

    write_review_gpkg
        Write a GeoPackage containing selected camera points, theoretical
        selection envelopes, and the AOI.

    verbose
        Print progress information.

    Returns
    -------
    pandas.DataFrame
        One row per capture, including selection status and distances.
    """
    source_dirs = [
        Path(p).expanduser()
        for p in source_dirs
    ]

    output_dir = Path(output_dir).expanduser()

    if copy_files:
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    if camera_height_m <= 0:
        raise ValueError("camera_height_m must be > 0.")

    if not 0 < hfov_deg < 180:
        raise ValueError("hfov_deg must be between 0 and 180.")

    if not 0 < vfov_deg < 180:
        raise ValueError("vfov_deg must be between 0 and 180.")

    if extra_buffer_m < 0:
        raise ValueError("extra_buffer_m cannot be negative.")

    if aoi_buffer_m < 0:
        raise ValueError("aoi_buffer_m cannot be negative.")

    valid_copy_modes = {
        "panchro",
        "representative",
        "all",
    }

    if copy_mode not in valid_copy_modes:
        raise ValueError(
            f"copy_mode must be one of {sorted(valid_copy_modes)}"
        )

    # --------------------------------------------------------
    # AOI
    # --------------------------------------------------------

    aoi_gdf = load_aoi(
        aoi,
        aoi_crs=aoi_crs,
        layer=aoi_layer,
    )

    # Camera GPS coordinates are WGS84.
    aoi_wgs84 = aoi_gdf.to_crs("EPSG:4326")

    # Pick the local UTM CRS automatically.
    metric_crs = aoi_wgs84.estimate_utm_crs()

    if metric_crs is None:
        raise RuntimeError(
            "Could not determine a suitable local metric CRS."
        )

    aoi_metric = aoi_wgs84.to_crs(metric_crs)

    aoi_geom = unary_union(
        aoi_metric.geometry
    )

    if aoi_buffer_m > 0:
        aoi_geom = aoi_geom.buffer(
            aoi_buffer_m
        )

    transformer = Transformer.from_crs(
        "EPSG:4326",
        metric_crs,
        always_xy=True,
    )

    # --------------------------------------------------------
    # Theoretical footprint
    # --------------------------------------------------------

    half_width_m = (
        camera_height_m
        * math.tan(
            math.radians(hfov_deg / 2.0)
        )
    )

    half_height_m = (
        camera_height_m
        * math.tan(
            math.radians(vfov_deg / 2.0)
        )
    )

    half_diagonal_m = math.hypot(
        half_width_m,
        half_height_m,
    )

    if verbose:
        print("Approximate image footprint")
        print("-" * 40)
        print(
            f"Camera height:      "
            f"{camera_height_m:.2f} m"
        )
        print(
            f"Footprint width:    "
            f"{2 * half_width_m:.2f} m"
        )
        print(
            f"Footprint height:   "
            f"{2 * half_height_m:.2f} m"
        )
        print(
            f"Half diagonal:      "
            f"{half_diagonal_m:.2f} m"
        )
        print(
            f"AOI buffer:         "
            f"{aoi_buffer_m:.2f} m"
        )
        print()

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    if verbose:
        print("Reading image metadata...")

    metadata = read_image_metadata(
        source_dirs
    )

    captures = group_captures(
        metadata
    )

    if verbose:
        print(
            f"Found {len(metadata):,} image files "
            f"in {len(captures):,} captures."
        )
        print()

    # --------------------------------------------------------
    # Selection
    # --------------------------------------------------------

    results = []

    selected_point_records = []
    selected_circle_records = []

    copied_count = 0
    selected_count = 0

    for capture_id, items in captures.items():

        representative = find_band(
            items,
            representative_band,
        )

        # Fall back to any image in the capture if needed.
        if representative is None:
            representative = items[0]

        lat = representative["lat"]
        lon = representative["lon"]

        if lat is None or lon is None:
            results.append(
                {
                    "capture_id": capture_id,
                    "latitude": None,
                    "longitude": None,
                    "distance_to_aoi_m": None,
                    "selection_radius_m": None,
                    "gps_accuracy_m": None,
                    "representative_file":
                        representative["path"].name,
                    "selected": False,
                    "files_copied": 0,
                    "reason": "missing GPS",
                }
            )
            continue

        lat = float(lat)
        lon = float(lon)

        x, y = transformer.transform(
            lon,
            lat,
        )

        camera_point = Point(
            x,
            y,
        )

        distance_to_aoi_m = (
            camera_point.distance(aoi_geom)
        )

        try:
            gps_accuracy_m = float(
                representative["gps_accuracy"]
            )
        except (TypeError, ValueError):
            gps_accuracy_m = 0.0

        selection_radius_m = (
            half_diagonal_m
            + extra_buffer_m
            + gps_accuracy_m
        )

        selected = (
            distance_to_aoi_m
            <= selection_radius_m
        )

        files_copied = 0

        if selected:
            selected_count += 1

            if copy_mode == "all":
                files_to_copy = list(items)

            elif copy_mode == "representative":
                files_to_copy = [
                    representative
                ]

            elif copy_mode == "panchro":
                panchro = find_band(
                    items,
                    "panch",
                )

                if panchro is None:
                    files_to_copy = []
                else:
                    files_to_copy = [
                        panchro
                    ]

            if copy_files:
                for item in files_to_copy:
                    copy_safely(
                        item["path"],
                        output_dir,
                    )

                    files_copied += 1
                    copied_count += 1

            selected_point_records.append(
                {
                    "capture_id": capture_id,
                    "distance_m":
                        distance_to_aoi_m,
                    "geometry":
                        camera_point,
                }
            )

            selected_circle_records.append(
                {
                    "capture_id": capture_id,
                    "radius_m":
                        selection_radius_m,
                    "geometry":
                        camera_point.buffer(
                            selection_radius_m
                        ),
                }
            )

        results.append(
            {
                "capture_id":
                    capture_id,

                "latitude":
                    lat,

                "longitude":
                    lon,

                "distance_to_aoi_m":
                    distance_to_aoi_m,

                "selection_radius_m":
                    selection_radius_m,

                "gps_accuracy_m":
                    gps_accuracy_m,

                "representative_file":
                    representative["path"].name,

                "selected":
                    selected,

                "files_copied":
                    files_copied,

                "reason":
                    "intersects AOI envelope"
                    if selected
                    else "outside AOI envelope",
            }
        )

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Outputs
    # --------------------------------------------------------

    parent_dir = output_dir.parent

    parent_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if write_csv:
        csv_path = (
            parent_dir
            / "image_selection.csv"
        )

        results_df.to_csv(
            csv_path,
            index=False,
        )

    if (
        write_review_gpkg
        and selected_point_records
    ):
        review_path = (
            parent_dir
            / "selection_review.gpkg"
        )

        # Remove an existing review file so stale layers do not remain.
        if review_path.exists():
            review_path.unlink()

        points_gdf = gpd.GeoDataFrame(
            selected_point_records,
            crs=metric_crs,
        )

        circles_gdf = gpd.GeoDataFrame(
            selected_circle_records,
            crs=metric_crs,
        )

        buffered_aoi_gdf = (
            gpd.GeoDataFrame(
                {
                    "aoi_buffer_m": [
                        aoi_buffer_m
                    ]
                },
                geometry=[aoi_geom],
                crs=metric_crs,
            )
        )

        points_gdf.to_file(
            review_path,
            layer="selected_camera_points",
            driver="GPKG",
        )

        circles_gdf.to_file(
            review_path,
            layer="selection_envelopes",
            driver="GPKG",
        )

        buffered_aoi_gdf.to_file(
            review_path,
            layer="selection_aoi",
            driver="GPKG",
        )

    if verbose:
        print("=" * 50)
        print("Image selection complete")
        print("=" * 50)

        print(
            f"Selected captures: "
            f"{selected_count:,} / "
            f"{len(captures):,}"
        )

        if copy_files:
            print(
                f"Files copied:      "
                f"{copied_count:,}"
            )

            print(
                f"Output directory:  "
                f"{output_dir}"
            )

        if write_csv:
            print(
                f"Selection CSV:     "
                f"{parent_dir / 'image_selection.csv'}"
            )

        if (
            write_review_gpkg
            and selected_point_records
        ):
            print(
                f"QGIS review:       "
                f"{parent_dir / 'selection_review.gpkg'}"
            )

    return results_df