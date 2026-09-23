from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
import json
import os
import tempfile

import numpy as np

from .srf import BandSRF, SensorSRF

if TYPE_CHECKING:
    from spec_tools import Spectrum


def wavelengths_match(
    *arrays: Iterable[float] | np.ndarray,
    rtol: float = 0.0,
    atol: float = 1e-12,
) -> bool:
    """
    Check whether multiple wavelength arrays match.

    Parameters
    ----------
    *arrays : iterable of array-like
        Two or more wavelength arrays to compare.
    rtol : float, optional
        Relative tolerance for np.allclose.
    atol : float, optional
        Absolute tolerance for np.allclose.

    Returns
    -------
    bool
        True if all arrays have the same shape and values within tolerance.

    Notes
    -----
    This is a checker only. It does not modify or interpolate anything.
    """
    if len(arrays) < 2:
        raise ValueError("Provide at least two wavelength arrays to compare.")

    ref = np.asarray(arrays[0], dtype=float)

    if ref.ndim != 1:
        raise ValueError("All wavelength arrays must be 1D.")

    for arr in arrays[1:]:
        arr = np.asarray(arr, dtype=float)

        if arr.ndim != 1:
            raise ValueError("All wavelength arrays must be 1D.")

        if ref.shape != arr.shape:
            return False

        if not np.allclose(ref, arr, rtol=rtol, atol=atol):
            return False

    return True


def align_band_srfs_to_wavelengths(
    wavelengths: Iterable[float] | np.ndarray,
    *bands: BandSRF,
) -> list[BandSRF]:
    """
    Interpolate one or more BandSRF objects onto a shared wavelength grid.

    Parameters
    ----------
    wavelengths : array-like
        Target wavelength grid to interpolate onto.
    *bands : BandSRF
        One or more BandSRF objects.

    Returns
    -------
    list[BandSRF]
        New BandSRF objects interpolated to the target wavelength grid.

    Notes
    -----
    Values outside the original SRF wavelength range are set to 0, using the
    existing BandSRF.interpolate_to() behavior.
    """
    target_wavelengths = np.asarray(wavelengths, dtype=float)

    if target_wavelengths.ndim != 1:
        raise ValueError("wavelengths must be a 1D array-like object.")

    if target_wavelengths.size == 0 or not np.all(np.isfinite(target_wavelengths)):
        raise ValueError("wavelengths must be nonempty and finite.")

    diffs = np.diff(target_wavelengths)
    if np.any(diffs <= 0):
        raise ValueError("wavelengths must be strictly increasing.")

    if len(bands) == 0:
        raise ValueError("Provide at least one BandSRF to align.")

    aligned_bands = []
    for band in bands:
        if not isinstance(band, BandSRF):
            raise TypeError(
                f"All inputs after wavelengths must be BandSRF objects. "
                f"Got {type(band).__name__}."
            )
        aligned_bands.append(band.interpolate_to(target_wavelengths))

    return aligned_bands

def _curve(wavelengths, values, label):
    x, y = np.asarray(wavelengths, dtype=float), np.asarray(values, dtype=float)
    if (x.ndim != 1 or len(x) < 2 or x.shape != y.shape
            or not np.all(np.isfinite(x)) or np.any(x <= 0)
            or np.any(np.diff(x) <= 0)):
        raise ValueError(f"{label}: expected matching arrays and finite, positive, increasing wavelengths.")
    return x, y


def calculate_sbaf(spectrum: Spectrum, source_band: BandSRF,
                   target_band: BandSRF) -> float:
    """Return target/source SRF-weighted reflectance (multiply source pixels).

    Curves are piecewise linear. Products are integrated exactly on their
    combined knots. No reflectance extrapolation or bridging of missing samples
    is performed. Inputs are never modified. Wavelength units must match.
    """
    if spectrum.quantity not in {"reflectance", "absolute_reflectance"}:
        raise ValueError("SBAF requires a reflectance spectrum.")
    if spectrum.value_unit != "1":
        raise ValueError('Spectrum must declare fractional reflectance with value_unit="1".')
    x, y = _curve(spectrum.wavelengths, spectrum.values, "Spectrum")
    if np.any(np.isinf(y)):
        raise ValueError("Spectrum contains infinite values.")

    def weighted(band):
        if band.wavelength_unit != spectrum.wavelength_unit:
            raise ValueError("Spectrum and SRF wavelength units must match.")
        bx, by = _curve(band.wavelengths, band.responses, band.name)
        if not np.all(np.isfinite(by)) or np.any(by < 0) or not np.any(by > 0):
            raise ValueError(f"{band.name}: responses must be finite, nonnegative and nonzero.")
        # Include zero-valued endpoints next to the active response.
        active = np.flatnonzero((by[:-1] > 0) | (by[1:] > 0))
        lo, hi = bx[active[0]], bx[active[-1] + 1]
        if x[0] > lo or x[-1] < hi:
            raise ValueError(f"Spectrum does not cover the active response of {band.name}.")
        grid = np.unique(np.concatenate(([lo, hi], bx[(bx > lo) & (bx < hi)],
                                         x[(x > lo) & (x < hi)])))
        response = np.interp(grid, bx, by)
        reflectance = np.interp(grid, x, y)
        used = (response[:-1] > 0) | (response[1:] > 0)
        if not np.all(np.isfinite(reflectance[:-1][used])) or not np.all(np.isfinite(reflectance[1:][used])):
            raise ValueError(f"Missing spectrum measurements within {band.name} response.")
        dx = np.diff(grid)[used]
        a, b = reflectance[:-1][used], reflectance[1:][used]
        c, d = response[:-1][used], response[1:][used]
        area = np.sum(dx * (c + d) / 2)
        value = np.sum(dx * (2*a*c + a*d + b*c + 2*b*d) / 6) / area
        if not np.isfinite(value):
            raise ValueError(f"Nonfinite weighted reflectance for {band.name}.")
        return value

    source, target = weighted(source_band), weighted(target_band)
    if source <= 1e-12 or target < 0:
        raise ValueError("Source weighted reflectance must exceed 1e-12 and target must be nonnegative.")
    factor = target / source
    if not np.isfinite(factor):
        raise ValueError("Nonfinite SBAF.")
    return float(factor)


def calculate_sbaf_factors(class_spectra: Mapping[int, Spectrum],
                           source_sensor: SensorSRF, target_sensor: SensorSRF,
                           band_pairs: Mapping[int, tuple[str, str]]) -> dict[int, dict[int, float]]:
    """Build {class ID: {one-based raster band: target/source factor}}."""
    if not class_spectra or not band_pairs:
        raise ValueError("Provide class spectra and band pairs.")
    for band in band_pairs:
        if isinstance(band, bool) or not isinstance(band, (int, np.integer)) or band < 1:
            raise ValueError("Raster band numbers must be positive integers (one-based).")
    result = {}
    for class_id, spectrum in class_spectra.items():
        if isinstance(class_id, bool) or not isinstance(class_id, (int, np.integer)):
            raise ValueError("Class IDs must be integers.")
        result[int(class_id)] = {
            int(index): calculate_sbaf(spectrum, source_sensor.get_band(pair[0]),
                                      target_sensor.get_band(pair[1]))
            for index, pair in band_pairs.items()
        }
    return result


def apply_sbaf_array(data, classification, factors, *, valid_mask=None,
                     classification_valid_mask=None, unmapped="error"):
    """Return a float64 masked (bands, rows, columns) array without mutation.

    Masked inputs, nonfinite pixels and optional False validity entries are
    invalid. Classification NoData masks adjusted bands only. Unmapped valid
    classes raise by default; ``unmapped='preserve'`` leaves them unchanged.
    Arrays must already share a spatial grid; the TIFF wrapper verifies this.
    """
    if unmapped not in {"error", "preserve"}:
        raise ValueError("unmapped must be 'error' or 'preserve'.")
    result = np.ma.array(data, dtype=np.float64, copy=True)
    classes = np.ma.asarray(classification)
    if result.ndim != 3 or classes.shape != result.shape[1:]:
        raise ValueError("Expected band-first data and a matching 2D classification.")
    result.mask = np.ma.getmaskarray(result) | ~np.isfinite(result.data)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape not in {result.shape, result.shape[1:]}:
            raise ValueError("valid_mask must match data or its spatial shape.")
        result.mask |= ~mask
    class_valid = ~np.ma.getmaskarray(classes) & np.isfinite(classes.data)
    if classification_valid_mask is not None:
        mask = np.asarray(classification_valid_mask, dtype=bool)
        if mask.shape != classes.shape:
            raise ValueError("classification_valid_mask must match classification.")
        class_valid &= mask
    if np.any(classes.data[class_valid] != np.floor(classes.data[class_valid])):
        raise ValueError("Valid classification values must be integers.")
    if not factors:
        raise ValueError("Factor table cannot be empty.")
    bands = set(next(iter(factors.values())))
    if not bands or any(set(row) != bands for row in factors.values()):
        raise ValueError("Each class must have the same nonempty set of raster bands.")
    for band in bands:
        if not isinstance(band, (int, np.integer)) or isinstance(band, bool) or not 1 <= band <= len(result):
            raise ValueError("Factor table references an invalid raster band.")
    for class_id, row in factors.items():
        if not isinstance(class_id, (int, np.integer)) or isinstance(class_id, bool):
            raise ValueError("Factor table class IDs must be integers.")
        if any(not np.isfinite(v) or v < 0 for v in row.values()):
            raise ValueError("Factors must be finite and nonnegative.")
    relevant = class_valid & np.any(~result.mask[[b-1 for b in bands]], axis=0)
    unknown = set(np.unique(classes.data[relevant])) - set(factors)
    if unknown and unmapped == "error":
        raise ValueError(f"No reference spectrum for classification values: {sorted(unknown)}")
    for band in bands:
        result.mask[band-1] |= ~class_valid
    for class_id, row in factors.items():
        pixels = class_valid & (classes.data == class_id)
        for band, factor in row.items():
            selected = pixels & ~result.mask[band-1]
            with np.errstate(over="ignore", invalid="ignore"):
                result.data[band-1, selected] *= factor
            if not np.all(np.isfinite(result.data[band-1, selected])):
                raise ValueError("SBAF multiplication overflowed.")
    return result


@dataclass(frozen=True)
class SBAFResult:
    """Paths to the corrected TIFF and JSON report, plus calculated factors."""
    output_path: Path
    report_path: Path
    factors: dict[int, dict[int, float]]


def apply_sbaf_tif(uav_path, classification_path, output_path, *, class_spectra,
                   source_sensor, target_sensor, band_pairs, unmapped="error") -> SBAFResult:
    """Apply class-specific SBAFs by raster windows and save float32 reflectance.

    Inputs must have identical CRS, transform and dimensions. Raster scale and
    offset are decoded for all bands; output scales/offsets are 1/0. Unmapped
    bands retain their physical values and masks. No clipping or spatial
    resampling is performed. Existing output/report files are never overwritten.
    A companion ``<output>.sbaf.json`` records factors and provenance.
    """
    import rasterio

    factors = calculate_sbaf_factors(class_spectra, source_sensor, target_sensor, band_pairs)
    output = Path(output_path).expanduser().resolve()
    report = output.with_suffix(output.suffix + ".sbaf.json")
    inputs = {Path(p).expanduser().resolve() for p in (uav_path, classification_path)}
    if output in inputs or report in inputs or output.exists() or report.exists():
        raise FileExistsError("Output and report must be new files, distinct from inputs.")
    if unmapped not in {"error", "preserve"}:
        raise ValueError("unmapped must be 'error' or 'preserve'.")
    record = {
        "direction": "target/source", "source_sensor": source_sensor.name,
        "target_sensor": target_sensor.name, "uav_path": str(Path(uav_path).resolve()),
        "classification_path": str(Path(classification_path).resolve()),
        "band_pairs": {str(k): list(v) for k, v in band_pairs.items()},
        "factors": factors, "unmapped": unmapped,
        "spectra": {str(k): {"name": v.name, "provenance": v.provenance,
                             "quantity": v.quantity, "value_unit": v.value_unit,
                             "wavelength_unit": v.wavelength_unit}
                    for k, v in class_spectra.items()},
    }
    with rasterio.open(uav_path) as src, rasterio.open(classification_path) as cls:
        if (cls.count != 1 or src.crs is None or cls.crs != src.crs
                or cls.transform != src.transform or cls.width != src.width or cls.height != src.height):
            raise ValueError("Classification must be single-band and match UAV CRS, transform and dimensions.")
        if any(index > src.count for index in band_pairs):
            raise ValueError("Band mapping exceeds raster band count.")
        scales, offsets = np.asarray(src.scales), np.asarray(src.offsets)
        if not np.all(np.isfinite(scales)) or not np.all(np.isfinite(offsets)):
            raise ValueError("Raster scales and offsets must be finite.")
        record["input_scales"], record["input_offsets"] = scales.tolist(), offsets.tolist()
        profile = dict(driver="GTiff", width=src.width, height=src.height,
                       count=src.count, crs=src.crs, transform=src.transform,
                       dtype="float32", nodata=float("nan"), tiled=True,
                       compress="deflate", BIGTIFF="IF_SAFER")
        # Stage in the output directory; failed validation never leaves a final TIFF.
        with tempfile.TemporaryDirectory(prefix=".sbaf-", dir=output.parent) as staging:
            staged = Path(staging) / "result.tif"
            staged_report = Path(staging) / "report.json"
            with rasterio.open(staged, "w", **profile) as dst:
                # Do not copy statistics or encoding metadata invalidated by adjustment.
                safe_tags = {k: v for k, v in src.tags().items()
                             if k in {"AREA_OR_POINT", "TIFFTAG_DATETIME", "TIFFTAG_IMAGEDESCRIPTION"}}
                dst.update_tags(**safe_tags)
                dst.update_tags(SBAF_DIRECTION="target/source", SBAF_REPORT=report.name)
                for i, description in enumerate(src.descriptions, 1):
                    if description:
                        dst.set_band_description(i, description)
                    if src.units[i-1]:
                        dst.set_band_unit(i, src.units[i-1])
                    if i in band_pairs:
                        dst.update_tags(i, SBAF_SOURCE_BAND=band_pairs[i][0], SBAF_TARGET_BAND=band_pairs[i][1])
                for _, window in dst.block_windows(1):
                    data = src.read(window=window, masked=True).astype(np.float64)
                    data = data * scales[:, None, None] + offsets[:, None, None]
                    adjusted = apply_sbaf_array(data, cls.read(1, window=window, masked=True),
                                                factors, unmapped=unmapped)
                    if np.any(np.abs(adjusted.compressed()) > np.finfo(np.float32).max):
                        raise ValueError("Adjusted data exceeds float32 range.")
                    dst.write(adjusted.filled(np.nan).astype(np.float32), window=window)
            staged_report.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
            # Exclusive creation also protects against a concurrent writer.
            os.link(staged_report, report)
            try:
                os.link(staged, output)
            except BaseException:
                report.unlink()
                raise
    return SBAFResult(output, report, factors)
