
## Class-based spectral band adjustment (SBAF)

SBAF adjusts UAV reflectance toward another sensor's spectral response while
retaining the UAV spatial grid. One reference `spec_tools.Spectrum` represents
each classification value; one target/source factor is calculated per class and
mapped raster band. This does not perform spatial resampling or calibration of
raw digital numbers.

Install the local projects in the same Python 3.12+ environment:

```sh
python -m pip install -e /path/to/spec-tools
python -m pip install -e '/path/to/uav-tools[sbaf]'
```

`spec-tools` is an optional dependency for spectral reference loading. Importing
uav-tools and using SRF utilities does not require it.

```python
from spec_tools import Spectrum
from uav_tools.spectral import apply_sbaf_tif

# source_sensor and target_sensor are SensorSRF objects built from BandSRFs.
# Names below must match those objects. Raster indexes start at 1.
result = apply_sbaf_tif(
    "uav_reflectance.tif",
    "classification.tif",
    "uav_adjusted.tif",
    class_spectra={
        1: Spectrum.from_asd("grass.ASD"),
        2: Spectrum.from_asd("soil.ASD"),
    },
    source_sensor=source_sensor,
    target_sensor=target_sensor,
    band_pairs={1: ("blue", "B2"), 2: ("green", "B3")},
)
print(result.factors)
print(result.output_path, result.report_path)
```

The factor is the target band's SRF-weighted mean reference reflectance divided
by the source band's mean. Integration uses piecewise-linear curves and an exact
product integral on their combined wavelength knots. SRF amplitudes need not be
normalized. The spectrum must declare reflectance or absolute reflectance with
`value_unit="1"`; wavelength units must match. Missing measurements within an
active response, inadequate spectral coverage, invalid SRFs, and source means
at or below 1e-12 raise errors. No reflectance extrapolation is performed.
Gaussian SRFs have nonzero tails: the reference must cover their supplied grid.
Choose that grid deliberately; no implicit response-tail cutoff is applied.

The classification must be single-band with exactly matching CRS, transform,
width and height. Valid class codes must be integers; zero is a valid class
unless explicitly marked NoData. Unknown classes raise by default; use
`unmapped="preserve"` to retain their values. Classification NoData masks adjusted
bands. UAV masks and nonfinite values remain invalid. Unmapped raster bands
retain their physical values and validity.

Processing uses bounded raster windows. Input scale/offset metadata is decoded
for every band, and the output is float32 with identity scale/offset and NaN
NoData. Thus unadjusted bands retain physical values, not necessarily their
original integer encoding. Values are not clipped. The output retains the
spatial grid, band descriptions, units and selected descriptive TIFF tags;
stale statistics are not copied. A `.tif.sbaf.json` companion records factors,
band mapping, spectrum names/provenance, and original scale/offset metadata.
Existing outputs are never overwritten. Validation/processing failures do not
publish a partial output TIFF.

For arrays already loaded in memory:

```python
from uav_tools.spectral import calculate_sbaf_factors, apply_sbaf_array

factors = calculate_sbaf_factors(class_spectra, source_sensor, target_sensor, band_pairs)
corrected = apply_sbaf_array(data, classification, factors)
```

`data` is physical reflectance with shape `(bands, rows, columns)`;
`classification` has shape `(rows, columns)`. Both can be masked arrays.
Optional `valid_mask` and `classification_valid_mask` use True for valid pixels.
The result is a new float64 masked array. The caller is responsible for spatial
alignment and decoding scale/offset when using this array API.

Run the synthetic calculation and GeoTIFF tests after installing both projects:

```sh
python -m unittest discover -s tests -v
```
