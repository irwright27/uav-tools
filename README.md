
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

## UAV calibration for STL partitioning

### Calibrating mixed-pixel contributions

For comparison with Jiang UAV contributions, select the mode explicitly:

```python
cal = define_background(parent_folder, vi="NDVI", scope="local",
                        quantity="jiang_contribution")
```

This reads `D_soil` and `D_olive`, not `mean_NDVI_soil` and
`mean_NDVI_olive`. D bands already include class fraction and relative
brightness. They are averaged equally across accepted raster cells on common
finite support, including zero contributions where a class is absent; no second
class-area weighting is applied. Coverage and footprint selection still apply.
The soil parameter is the equal-date mean soil contribution. Lambda is
`(max(D_olive) - min(D_olive)) / min(D_olive)` across flight summaries, with a
positive minimum required. Soil is not subtracted again. `define_lambda` rejects
`soil_background` in this mode and needs only the woody bands.

This mode is NDVI-only. Reports record the selected quantity in each flight and
parameter settings. `class_means` contains spatial means of the selected
quantity, so consult `quantity` before interpreting this legacy field name.
`class_cells` counts cells where the class is present, while contribution means
also include valid zeros in cells where it is absent.

Contribution lambda reflects fraction and relative-brightness changes as well
as canopy greenness. It is an observed-range proxy, not an independent estimate
of physiological seasonality. A constant soil contribution and synchronized
seasonal phases remain assumptions. Match the satellite footprint and spatial
statistic: an equal-cell mean of contributions is not an orchard median or NDVI
computed from orchard-pooled reflectances. Inspect seasonal soil variability and
negative modeled cover contributions after using these estimates.

The default `quantity="class_mean"` preserves the prior behavior documented
below, including support for other VIs. Use it explicitly when the intended
quantity is class-only VI rather than a mixed-pixel contribution.


`uav_tools.calibrate_stl` estimates a constant soil background and a woody
seasonality parameter from class means in existing aggregated UAV GeoTIFFs.
It reads the products; it does not rerun aggregation or modify imagery.

```python
from uav_tools.calibrate_stl import define_background, define_soil, define_lambda

parent_folder = "/path/to/COR_CS3/auto_processed"
cal = define_background(parent_folder, vi="NDVI")  # discover matching flights
print(cal.soil_background, cal.lambda_woody)

# Select flights explicitly; YYYYMMDD, YYYY-MM-DD and date objects are accepted.
dates = ["20250812", "20260127", "20260317", "20260507"]
soil = define_soil(parent_folder, dates, vi="NDVI")
lam = define_lambda(parent_folder, dates, vi="NDVI", soil_background=soil.value)
print(soil.value, lam.value)

# Retain per-flight values, areas, fractions, locations, settings and flags.
print(cal.observations)
print(cal.soil.location_diagnostics)
print(cal.flags)
cal.save("olive_ndvi_calibration.json")  # refuses to overwrite
```

The parent folder contains `<YYYYMMDD>/agg/*_<VI>_AGG_jiang.tif`. Filenames are
parsed from the right, so underscores in block and sensor names are supported.
`vi` is normalized to uppercase; file tokens and band descriptions must match
that spelling. Only flights containing that VI are discovered with `dates=None`;
explicit dates must each have exactly one matching product. Duplicate products,
inconsistent date tags and missing bands raise errors instead of guessing.

Required bands are `mean_<VI>_soil`, `mean_<VI>_olive`, `f_full_soil`,
`f_full_olive` and `valid_coverage_class`. `define_soil` only needs the soil
bands; `define_lambda` only needs the woody bands when a background is supplied.
`soil_class` and `woody_class` allow other class labels. For example,
`vi="GNDVI"` reads `mean_GNDVI_soil` and `mean_GNDVI_olive` from GNDVI products.
The input means must actually represent the named VI; renaming NDVI bands does
not create another index. No NDVI bounds or Jiang NDVI reflectance identities
are imposed on other indices. `D_*` and `L_*` contribution bands are not used.

A flight's class mean is weighted by its valid class area (`f_full_*` times pixel
area). A projected CRS is required; reported areas use squared CRS units.
NoData, masks, nonfinite values, and raster scale/offset metadata are honored.
Cells require `valid_coverage_class >= 0.9` by default (`min_coverage` can change
this). Footprint selection uses, in order: an explicit aligned raster `mask`
(positive values inside), an `orchard_footprint` band, the same-flight
`<prefix>_<date>_aggregation_class_metrics.csv` footprint, or all raster cells.
The last case is flagged `no_footprint_mask`. Explicit masks must match the
flight grid; no silent reprojection is performed. Footprints/class mappings
should represent the same orchard across dates. `summarize_flights(...)` exposes
these summaries without estimating either parameter.

### A shared calibration for multiple olive orchards

```python
locations = {
    "COR_CS3": {
        "parent_folder": "/path/to/COR_CS3/auto_processed",
        "dates": ["20250812", "20260127", "20260317", "20260507"],
    },
    "another_orchard": {
        "parent_folder": "/path/to/another_orchard/auto_processed",
        "dates": None,
        # "mask": "/path/to/aligned_orchard_mask.tif",  # optional
    },
}
cal = define_background(locations, vi="NDVI", scope="general")
print(cal.soil_background, cal.lambda_woody)
print(cal.woody.per_location)
print(cal.woody.between_location_sd, cal.woody.bootstrap_ci)
print(cal.woody.leave_one_out)
```

A mapping from names directly to folders is also supported. Top-level `dates`
and `mask` apply to each location unless overridden in its dictionary. Duplicate
resolved folders are rejected to avoid counting an orchard twice. `scope="local"`
requires one location; `scope="general"` allows starting with one but flags that
transfer cannot be evaluated.

The estimator operates in three stages:

1. Compute class-area-weighted means within each flight.
2. For each orchard, average soil means equally across dates to obtain `d_i`.
   Estimate `lambda_i = (max(O_i) - min(O_i)) / (min(O_i) - d_i)`, where `O_i`
   is that orchard's woody-class VI series. At least two dates and a positive
   baseline above `baseline_epsilon=1e-6` are required for lambda.
3. Average the orchard estimates equally to obtain the shared soil background
   and lambda. Each orchard's lambda uses its own soil background, **before**
   pooling. More flights or a larger area do not give an orchard more influence.
   `location_weights={"COR_CS3": 1, "another_orchard": 2}` provides explicit
   positive relative weights for a different target orchard population.

Never take the extrema across a concatenated series of different orchards:
spatial differences in average greenness would become false seasonality.
When calling `define_lambda` separately for multiple orchards, leave
`soil_background=None` to estimate local backgrounds, or supply a mapping by
location. Supplying a scalar intentionally uses that same background everywhere;
passing a previously pooled soil scalar is not equivalent to `define_background`.

Results include per-location estimates, normalized weights, date counts, temporal
VI ranges/standard deviations, class-fraction ranges, and per-flight provenance.
`between_location_sd` is the weighted population standard deviation of the local
parameter estimates. The 95% `bootstrap_ci` uses 2,000 deterministic resamples
of whole locations (seed 0), retaining their relative weights. It describes
sampling of orchards, not sensor error, classification uncertainty, or missed
seasonal extrema. With one orchard, these between-orchard statistics are `None`;
fewer than five orchards receive a small-sample flag.

`leave_one_out` pools the other orchards and reports `training_estimate`,
`held_out_estimate`, and their difference (`error`) for each withheld orchard.
These are parameter-transfer diagnostics against UAV-derived local estimates,
not an independent validation of the satellite partition. The held-out orchard
never contributes to its training estimate.

### Scientific scope and using s2-tools

Lambda is an **observed-range proxy**, motivated by the background-corrected
woody baseline in Lu et al. (2003), equations 5 and 11:
<https://doi.org/10.1016/S0034-4257(03)00054-3>.
It assumes class-only VI variation represents relative woody variation, stable
canopy support, and compatible woody/herbaceous seasonal timing. Timing is not
fitted or tested here. Sparse observations, records shorter than one year, and
multi-year records (whose range may contain trends) are flagged. A constant
observed woody series gives zero, but sparse sampling cannot establish absence
of annual variation. Review the per-flight values and canopy fractions.

A global soil intercept is also an assumption: soils, moisture and management
can vary across orchards. Inspect local soil ranges and between-orchard spread
before transferring a single value. Representative seasonal sampling, consistent
classifications and sensor calibration are necessary. General calibration means
pooling representative orchards; it does not establish universality for all
olive orchards. Other VIs require their own scientific assessment and calibration.

The existing NDVI partition function accepts these results directly:

```python
from s2_tools.partition import isolate_olive_ndvi

partition = isolate_olive_ndvi(
    stl_result,
    soil_ndvi=cal.soil_background,
    lambda_woody=cal.lambda_woody,
)
```

This addition makes UAV calibration VI-generic. It does not change s2-tools'
currently NDVI-specific partition names, checks or outputs. Generalizing that
partition is a separate change; do not pass other indices into an NDVI-only
interface without adapting and validating it.
