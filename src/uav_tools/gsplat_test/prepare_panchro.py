from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import tifffile
from PIL import Image


def find_tiffs(
    input_dir: str | Path,
) -> list[Path]:
    """
    Find valid TIFF image files in a directory.

    Ignores hidden macOS AppleDouble files such as:
        ._IMG_0157_6.tif

    Parameters
    ----------
    input_dir
        Directory containing selected Panchro TIFF images.

    Returns
    -------
    list of Path
        Sorted TIFF paths.
    """
    input_dir = Path(input_dir).expanduser()

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory does not exist:\n{input_dir}"
        )

    files = []

    for path in input_dir.iterdir():

        if not path.is_file():
            continue

        # Ignore macOS hidden/resource-fork files.
        if path.name.startswith("."):
            continue

        if path.suffix.lower() in {".tif", ".tiff"}:
            files.append(path)

    files = sorted(files)

    if not files:
        raise FileNotFoundError(
            f"No TIFF images found in:\n{input_dir}"
        )

    return files


def read_panchro(
    path: str | Path,
) -> np.ndarray:
    """
    Read a single Panchro TIFF as a 2-D NumPy array.

    Raises an error if the TIFF does not appear to be single-band.
    """
    path = Path(path)

    image = tifffile.imread(path)

    # Remove singleton dimensions, e.g. (1, H, W).
    image = np.squeeze(image)

    if image.ndim != 2:
        raise ValueError(
            f"Expected a single-band Panchro image, but {path.name} "
            f"has shape {image.shape}."
        )

    return image


def estimate_global_stretch(
    image_paths: Sequence[str | Path],
    *,
    low_percentile: float = 1.0,
    high_percentile: float = 99.5,
    sample_stride: int = 16,
    ignore_zeros: bool = True,
) -> tuple[float, float]:
    """
    Estimate one global intensity stretch for an entire image collection.

    A subsample of pixels from every image is combined and global
    percentiles are calculated. Using one stretch for the whole dataset
    avoids independently auto-stretching each UAV image.

    Parameters
    ----------
    image_paths
        Panchro TIFF files.

    low_percentile
        Lower percentile used for the 8-bit stretch.

    high_percentile
        Upper percentile used for the 8-bit stretch.

    sample_stride
        Use every Nth row and column when estimating the percentiles.
        This greatly reduces memory use.

    ignore_zeros
        Exclude zero-valued pixels from stretch estimation.

    Returns
    -------
    (low_value, high_value)
        Global intensity limits.
    """
    if not 0 <= low_percentile < high_percentile <= 100:
        raise ValueError(
            "Percentiles must satisfy "
            "0 <= low_percentile < high_percentile <= 100."
        )

    if sample_stride < 1:
        raise ValueError("sample_stride must be >= 1.")

    samples = []

    for path in image_paths:
        image = read_panchro(path)

        sample = image[
            ::sample_stride,
            ::sample_stride,
        ].ravel()

        if ignore_zeros:
            sample = sample[sample > 0]

        if sample.size:
            samples.append(
                sample.astype(np.float32)
            )

    if not samples:
        raise RuntimeError(
            "No valid image pixels were available for "
            "global stretch estimation."
        )

    pixels = np.concatenate(samples)

    low_value = float(
        np.percentile(
            pixels,
            low_percentile,
        )
    )

    high_value = float(
        np.percentile(
            pixels,
            high_percentile,
        )
    )

    if high_value <= low_value:
        raise RuntimeError(
            "Invalid global stretch: upper intensity "
            "limit is not greater than lower limit."
        )

    return low_value, high_value


def stretch_to_uint8(
    image: np.ndarray,
    low_value: float,
    high_value: float,
) -> np.ndarray:
    """
    Convert a Panchro image to uint8 using fixed global limits.
    """
    image = image.astype(
        np.float32,
        copy=False,
    )

    scaled = (
        (image - low_value)
        / (high_value - low_value)
    )

    scaled = np.clip(
        scaled,
        0.0,
        1.0,
    )

    return np.round(
        scaled * 255.0
    ).astype(np.uint8)


def panchro_to_rgb(
    image_8bit: np.ndarray,
) -> np.ndarray:
    """
    Duplicate one Panchro channel into R, G, and B.

    Returns
    -------
    ndarray
        H x W x 3 uint8 RGB array.
    """
    if image_8bit.ndim != 2:
        raise ValueError(
            "Expected a 2-D grayscale image."
        )

    return np.repeat(
        image_8bit[..., np.newaxis],
        3,
        axis=2,
    )


def resize_rgb(
    image_rgb: np.ndarray,
    downsample_factor: int,
) -> np.ndarray:
    """
    Downsample an RGB image using high-quality Lanczos interpolation.

    Examples
    --------
    factor=1
        Keep original size.

    factor=2
        Half width and half height.

    factor=4
        Quarter width and quarter height.
    """
    if downsample_factor < 1:
        raise ValueError(
            "downsample_factor must be >= 1."
        )

    if downsample_factor == 1:
        return image_rgb

    image = Image.fromarray(
        image_rgb,
        mode="RGB",
    )

    width, height = image.size

    new_size = (
        max(1, width // downsample_factor),
        max(1, height // downsample_factor),
    )

    image = image.resize(
        new_size,
        resample=Image.Resampling.LANCZOS,
    )

    return np.asarray(image)


def convert_panchro_image(
    input_path: str | Path,
    output_path: str | Path,
    *,
    low_value: float,
    high_value: float,
    downsample_factor: int = 1,
    overwrite: bool = False,
) -> dict:
    """
    Convert one Panchro TIFF to a 3-channel 8-bit PNG.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists:\n{output_path}"
        )

    image = read_panchro(
        input_path
    )

    original_height, original_width = (
        image.shape
    )

    image_8bit = stretch_to_uint8(
        image,
        low_value,
        high_value,
    )

    image_rgb = panchro_to_rgb(
        image_8bit
    )

    image_rgb = resize_rgb(
        image_rgb,
        downsample_factor,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(
        image_rgb,
        mode="RGB",
    ).save(
        output_path,
        format="PNG",
        optimize=False,
    )

    output_height, output_width = (
        image_rgb.shape[:2]
    )

    return {
        "input_file": input_path.name,
        "output_file": output_path.name,
        "original_width": original_width,
        "original_height": original_height,
        "output_width": output_width,
        "output_height": output_height,
        "low_value": low_value,
        "high_value": high_value,
    }


def prepare_panchro_for_gsplat(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    low_percentile: float = 1.0,
    high_percentile: float = 99.5,
    sample_stride: int = 16,
    downsample_factor: int = 1,
    overwrite: bool = False,
    write_manifest: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Prepare selected MicaSense Panchro TIFF images for COLMAP / 3DGS.

    All images are converted to three-channel 8-bit PNGs using ONE
    global intensity stretch calculated across the entire dataset.

    Parameters
    ----------
    input_dir
        Directory containing selected Panchro TIFFs.

    output_dir
        Directory for generated PNG files.

    low_percentile, high_percentile
        Global percentile stretch.

    sample_stride
        Pixel sampling interval used when calculating the global stretch.

    downsample_factor
        Spatial downsampling:

        1 = original resolution
        2 = half width/height
        4 = quarter width/height

    overwrite
        Replace existing PNG files.

    write_manifest
        Write preparation_manifest.csv in output_dir.

    verbose
        Print processing information.

    Returns
    -------
    pandas.DataFrame
        Manifest containing one row per converted image.
    """
    input_dir = Path(
        input_dir
    ).expanduser()

    output_dir = Path(
        output_dir
    ).expanduser()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_paths = find_tiffs(
        input_dir
    )

    if verbose:
        print(
            f"Found {len(image_paths):,} Panchro TIFFs."
        )
        print()
        print(
            "Estimating global intensity stretch..."
        )

    low_value, high_value = (
        estimate_global_stretch(
            image_paths,
            low_percentile=low_percentile,
            high_percentile=high_percentile,
            sample_stride=sample_stride,
        )
    )

    if verbose:
        print(
            f"Global {low_percentile:g}–"
            f"{high_percentile:g}% stretch:"
        )
        print(
            f"    low  = {low_value:.2f}"
        )
        print(
            f"    high = {high_value:.2f}"
        )
        print()
        print(
            f"Downsample factor: "
            f"{downsample_factor}×"
        )
        print()
        print("Converting images...")

    records = []

    for i, input_path in enumerate(
        image_paths,
        start=1,
    ):
        output_path = (
            output_dir
            / f"{input_path.stem}.png"
        )

        if (
            output_path.exists()
            and not overwrite
        ):
            if verbose:
                print(
                    f"[{i:03d}/{len(image_paths):03d}] "
                    f"SKIP {output_path.name}"
                )

            continue

        record = convert_panchro_image(
            input_path,
            output_path,
            low_value=low_value,
            high_value=high_value,
            downsample_factor=downsample_factor,
            overwrite=overwrite,
        )

        records.append(
            record
        )

        if verbose:
            print(
                f"[{i:03d}/{len(image_paths):03d}] "
                f"{input_path.name} "
                f"-> {output_path.name}"
            )

    manifest = pd.DataFrame(
        records
    )

    if write_manifest:
        manifest_path = (
            output_dir
            / "preparation_manifest.csv"
        )

        manifest.to_csv(
            manifest_path,
            index=False,
        )

        if verbose:
            print()
            print(
                f"Manifest: {manifest_path}"
            )

    if verbose:
        print()
        print("=" * 50)
        print("Preparation complete")
        print("=" * 50)
        print(
            f"Input images:  {len(image_paths):,}"
        )
        print(
            f"Converted:     {len(records):,}"
        )
        print(
            f"Output:        {output_dir}"
        )

    return manifest