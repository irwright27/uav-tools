"""UAV class-mean or Jiang-contribution calibrations for STL vegetation partitioning.

Lambda is an observed-range proxy, not an independently validated biophysical
parameter. Locations are summarized before pooling; all selected dates within a
location have equal weight. No NDVI-specific bounds or reflectance identities
are applied to other vegetation indices.
"""
from __future__ import annotations

import csv
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import date, datetime
from numbers import Real
from pathlib import Path

import numpy as np
import rasterio

__all__ = ["FlightSummary", "ParameterEstimate", "CalibrationResult",
           "summarize_flights", "define_soil", "define_lambda", "define_background"]


@dataclass(frozen=True)
class FlightSummary:
    location: str
    date: str
    path: str
    vi: str
    class_means: dict[str, float]
    class_areas: dict[str, float]
    class_cells: dict[str, int]
    class_fractions: dict[str, float]
    footprint_source: str
    flags: tuple[str, ...]
    quantity: str = "class_mean"


@dataclass(frozen=True)
class ParameterEstimate:
    """A scalar plus its provenance and between-location diagnostics.

    bootstrap_ci resamples entire locations, not pixels or dates. It is absent
    for a single location and does not measure sensor or temporal uncertainty.
    leave_one_out compares held-out local parameter estimates to pooled training
    estimates; it does not validate the satellite partition or annual phenology.
    """
    value: float
    vi: str
    method: str
    per_location: dict[str, float]
    location_weights: dict[str, float]
    location_diagnostics: dict[str, dict]
    between_location_sd: float | None
    bootstrap_ci: tuple[float, float] | None
    leave_one_out: dict[str, dict[str, float]]
    observations: tuple[FlightSummary, ...]
    flags: tuple[str, ...]
    settings: dict

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        """Save a JSON report; refuse to overwrite an existing file."""
        return _save(self.to_dict(), path)


@dataclass(frozen=True)
class CalibrationResult:
    vi: str
    scope: str
    soil: ParameterEstimate
    woody: ParameterEstimate

    @property
    def soil_background(self):
        return self.soil.value

    @property
    def lambda_woody(self):
        return self.woody.value

    @property
    def observations(self):
        return self.soil.observations

    @property
    def flags(self):
        return tuple(dict.fromkeys(self.soil.flags + self.woody.flags))

    def to_dict(self):
        return {"schema": "uav_tools.calibrate_stl.v1", "vi": self.vi,
                "scope": self.scope, "soil_background": self.soil_background,
                "lambda_woody": self.lambda_woody,
                "soil": self.soil.to_dict(), "woody": self.woody.to_dict()}

    def save(self, path):
        """Save values, per-flight observations, settings and diagnostics."""
        return _save(self.to_dict(), path)


def _save(data, path):
    text = json.dumps(data, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    with path.open("x") as stream:
        stream.write(text)
    return path


def _finite(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)


def _date(value):
    if isinstance(value, datetime):
        value = value.date()
    if isinstance(value, date):
        return value.strftime("%Y%m%d")
    if isinstance(value, str):
        for fmt in ("%Y%m%d", "%Y-%m-%d"):
            try:
                parsed = datetime.strptime(value, fmt)
                if parsed.strftime(fmt) == value:
                    return parsed.strftime("%Y%m%d")
            except ValueError:
                pass
    raise ValueError(f"Invalid flight date {value!r}; use YYYYMMDD or YYYY-MM-DD")


def _locations(parent_folder, dates, mask):
    if isinstance(parent_folder, Mapping):
        if not parent_folder:
            raise ValueError("Supply at least one location")
        entries = parent_folder
    else:
        entries = {Path(parent_folder).resolve().parent.name: parent_folder}
    result, roots = [], set()
    for name, spec in entries.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Location names must be nonempty strings")
        if isinstance(spec, Mapping):
            unknown = set(spec) - {"parent_folder", "dates", "mask"}
            if unknown or "parent_folder" not in spec:
                raise ValueError(f"{name}: expected parent_folder, optional dates and mask; unknown {unknown}")
            root = Path(spec["parent_folder"]).expanduser().resolve()
            selected, region = spec.get("dates", dates), spec.get("mask", mask)
        else:
            root = Path(spec).expanduser().resolve()
            selected, region = dates, mask
        if not root.is_dir():
            raise FileNotFoundError(root)
        if root in roots:
            raise ValueError(f"Duplicate location folder: {root}")
        roots.add(root)
        if selected is not None:
            if isinstance(selected, (str, date)):
                selected = [selected]
            selected = [_date(d) for d in selected]
            if not selected or len(set(selected)) != len(selected):
                raise ValueError(f"{name}: dates must be nonempty and unique")
            selected = sorted(selected)
        result.append((name, root, selected, region))
    return result


def _files(root, dates, vi):
    # Parse from the right: underscores in block/sensor prefixes are harmless.
    def matches(folder):
        return sorted(p for p in folder.glob("*.tif")
                      if p.stem.rsplit("_", 3)[-3:] == [vi, "AGG", "jiang"])
    if dates is None:
        dates = sorted(p.name for p in root.iterdir()
                       if p.is_dir() and re.fullmatch(r"\d{8}", p.name)
                       and matches(p / "agg"))
    if not dates:
        raise FileNotFoundError(f"No {vi}_AGG_jiang.tif flights under {root}")
    for day in dates:
        _date(day)
        found = matches(root / day / "agg")
        if len(found) != 1:
            raise ValueError(f"{root / day}: expected one {vi}_AGG_jiang.tif, found {len(found)}")
        path = found[0]
        if day not in path.stem.split("_"):
            raise ValueError(f"Filename date does not match folder: {path}")
        yield day, path


def _band(ds, name):
    if ds.descriptions.count(name) != 1:
        raise ValueError(f"{ds.name}: expected exactly one band named {name!r}")
    index = ds.descriptions.index(name) + 1
    return (ds.read(index, masked=True).astype(float).filled(np.nan)
            * ds.scales[index - 1] + ds.offsets[index - 1])


def _footprint(ds, path, mask):
    if mask is not None:
        mask = Path(mask).expanduser().resolve()
        with rasterio.open(mask) as region:
            if (region.count != 1 or region.shape != ds.shape or region.crs != ds.crs
                    or not region.transform.almost_equals(ds.transform)):
                raise ValueError(f"Footprint mask must be a single band on the flight grid: {mask}")
            values = region.read(1, masked=True).astype(float).filled(np.nan)
            return np.isfinite(values) & (values > 0), str(Path(mask).resolve()), ()
    if "orchard_footprint" in ds.descriptions:
        values = _band(ds, "orchard_footprint")
        return np.isfinite(values) & (values > 0), "orchard_footprint band", ()
    # Existing COR products store the orchard mask in the companion CSV.
    suffix = f"_{ds.tags().get('uav_date', path.parent.parent.name)}_"
    prefix = path.name.rsplit(suffix, 1)[0] + suffix.rstrip("_")
    sidecar = path.with_name(prefix + "_aggregation_class_metrics.csv")
    if sidecar.exists():
        with sidecar.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"row", "col", "orchard_footprint"}.issubset(reader.fieldnames or []):
                raise ValueError(f"Missing footprint columns in {sidecar}")
            out, seen = np.zeros(ds.shape, dtype=bool), set()
            for row in reader:
                i, j = int(row["row"]), int(row["col"])
                if not (0 <= i < ds.height and 0 <= j < ds.width) or (i, j) in seen:
                    raise ValueError(f"Invalid or duplicate footprint cell in {sidecar}")
                seen.add((i, j))
                flag = row["orchard_footprint"].strip().lower()
                if flag not in {"true", "false", "1", "0"}:
                    raise ValueError(f"Invalid footprint value in {sidecar}: {flag}")
                out[i, j] = flag in {"true", "1"}
            if len(seen) != ds.width * ds.height:
                raise ValueError(f"Incomplete footprint grid in {sidecar}")
        return out, str(sidecar), ()
    return np.ones(ds.shape, dtype=bool), "all raster cells", ("no_footprint_mask",)


def summarize_flights(parent_folder, dates=None, *, vi="NDVI", classes=("soil", "olive"),
                      min_coverage=0.9, mask=None, quantity="class_mean"):
    """Read class means or already-weighted Jiang contributions.

    quantity='class_mean' preserves the legacy class-area-weighted VI means.
    quantity='jiang_contribution' reads D_<class> (NDVI only), averaged equally
    over accepted raster cells, including zero contributions for absent classes.
    D already includes class fraction and brightness: do not weight it again.
    All requested D bands use common finite support. class_means then contains
    spatial means of contributions, as recorded by FlightSummary.quantity.

    parent_folder is an auto_processed folder or a mapping of location names to
    folders / {parent_folder, dates, mask} dictionaries. Explicit dates must all
    resolve; dates=None discovers only flights containing the requested VI.
    Bands are mean_<VI>_<class>, f_full_<class>, valid_coverage_class.
    A projected CRS is required for area weighting. Raster scales, offsets,
    NoData, masks and nonfinite values are honored. No VI range is assumed.
    mask optionally selects an aligned positive-valued footprint raster; otherwise
    use orchard_footprint band, companion CSV, or all cells (with a flag).
    """
    if not isinstance(vi, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9-]*", vi):
        raise ValueError("vi must be a single filename token, e.g. NDVI, GNDVI or EVI")
    vi = vi.upper()
    if quantity not in {"class_mean", "jiang_contribution"}:
        raise ValueError("quantity must be class_mean or jiang_contribution")
    contribution = quantity == "jiang_contribution"
    if contribution and vi != "NDVI":
        raise ValueError("Jiang D bands are supported only for NDVI")
    min_coverage = _finite(min_coverage, "min_coverage")
    if not 0 <= min_coverage <= 1:
        raise ValueError("min_coverage must be in [0, 1]")
    if isinstance(classes, str):
        classes = (classes,)
    classes = tuple(classes)
    if not classes or len(set(classes)) != len(classes) or any(not isinstance(c, str) or not c for c in classes):
        raise ValueError("classes must contain unique nonempty names")
    summaries = []
    for location, root, selected, region in _locations(parent_folder, dates, mask):
        for day, path in _files(root, selected, vi):
            with rasterio.open(path) as ds:
                if ds.crs is None or not ds.crs.is_projected:
                    raise ValueError(f"Projected CRS required for area weights: {path}")
                if ds.tags().get("uav_date", day) != day:
                    raise ValueError(f"uav_date tag conflicts with folder: {path}")
                pixel_area = abs(ds.transform.a * ds.transform.e - ds.transform.b * ds.transform.d)
                if not np.isfinite(pixel_area) or pixel_area <= 0:
                    raise ValueError(f"Invalid pixel area: {path}")
                footprint, source, flags = _footprint(ds, path, region)
                coverage = _band(ds, "valid_coverage_class")
                support = footprint & np.isfinite(coverage) & (coverage >= min_coverage) & (coverage > 0)
                if np.any(footprint & np.isfinite(coverage) & ((coverage < 0) | (coverage > 1 + 1e-5))):
                    raise ValueError(f"Invalid coverage fractions: {path}")
                if contribution:
                    contribution_bands = {cls: _band(ds, f"D_{cls}") for cls in classes}
                    fraction_bands = {cls: _band(ds, f"f_full_{cls}") for cls in classes}
                    for cls in classes:
                        f = fraction_bands[cls]
                        if np.any(support & np.isfinite(f) & ((f < 0) | (f > coverage + 1e-5))):
                            raise ValueError(f"Invalid class area fractions: {path}")
                    common = support.copy()
                    for cls in classes:
                        common &= np.isfinite(contribution_bands[cls]) & np.isfinite(fraction_bands[cls])
                    support = common
                means, areas, cells, fractions = {}, {}, {}, {}
                for cls in classes:
                    values = contribution_bands[cls] if contribution else _band(ds, f"mean_{vi}_{cls}")
                    weights = fraction_bands[cls] if contribution else _band(ds, f"f_full_{cls}")
                    if np.any(support & np.isfinite(weights) & ((weights < 0) | (weights > coverage + 1e-5))):
                        raise ValueError(f"Invalid class area fractions for {cls}: {path}")
                    valid = support & np.isfinite(values) & np.isfinite(weights) & ((weights >= 0) if contribution else (weights > 0))
                    if not valid.any():
                        raise ValueError(f"No valid {cls} area for {location} on {day}")
                    means[cls] = float(np.mean(values[valid]) if contribution else
                                       np.average(values[valid], weights=weights[valid]))
                    areas[cls] = float(weights[valid].sum() * pixel_area)
                    cells[cls] = int((valid & (weights > 0)).sum())
                    fractions[cls] = float(weights[valid].sum() / coverage[support].sum())
                summaries.append(FlightSummary(location, day, str(path), vi, means, areas,
                                               cells, fractions, source, flags, quantity))
    return tuple(summaries)


def _pool(values, weights):
    return float(np.average(list(values.values()), weights=[weights[k] for k in values]))


def _estimate(values, observations, method, weights, flags, settings):
    names = list(values)
    if weights is None:
        weights = dict.fromkeys(names, 1.0)
    elif not isinstance(weights, Mapping) or set(weights) != set(names):
        raise ValueError("location_weights must contain exactly the selected location names")
    weights = {k: _finite(weights[k], f"weight for {k}", positive=True) for k in names}
    largest = max(weights.values())
    weights = {k: v / largest for k, v in weights.items()}
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    value = _pool(values, weights)
    spread, interval, loo = None, None, {}
    if len(names) > 1:
        x, w = np.array(list(values.values())), np.array(list(weights.values()))
        spread = float(np.sqrt(np.sum(w * (x - value)**2)))
        rng = np.random.default_rng(0)
        picks = rng.integers(0, len(names), size=(2000, len(names)))
        draws = np.sum(x[picks] * w[picks], axis=1) / np.sum(w[picks], axis=1)
        interval = tuple(float(v) for v in np.quantile(draws, [0.025, 0.975]))
        for held in names:
            training = {k: v for k, v in values.items() if k != held}
            predicted = _pool(training, weights)
            loo[held] = {"training_estimate": predicted, "held_out_estimate": values[held],
                         "error": predicted - values[held]}
        if len(names) < 5:
            flags.append("few_locations_for_general_calibration")
    else:
        flags.append("single_location_no_transfer_validation")
    flags.extend(f"{o.location}/{o.date}: {f}" for o in observations for f in o.flags)
    return ParameterEstimate(value, observations[0].vi, method, values, weights,
                             _diagnostics(observations),
                             spread, interval, loo, observations,
                             tuple(dict.fromkeys(flags)), settings)


def _group(observations):
    grouped = {}
    for obs in observations:
        grouped.setdefault(obs.location, []).append(obs)
    return grouped


def _diagnostics(observations):
    result = {}
    for name, flights in _group(observations).items():
        classes = {}
        for cls in flights[0].class_means:
            values = np.array([o.class_means[cls] for o in flights])
            fractions = [o.class_fractions[cls] for o in flights]
            classes[cls] = {
                "mean": float(values.mean()), "sd": float(values.std()),
                "min": float(values.min()), "max": float(values.max()),
                "range": float(np.ptp(values)),
                "min_date": flights[int(values.argmin())].date,
                "max_date": flights[int(values.argmax())].date,
                "fraction_min": min(fractions), "fraction_max": max(fractions),
            }
        result[name] = {"n_dates": len(flights), "first_date": flights[0].date,
                        "last_date": flights[-1].date, "classes": classes}
    return result


def _soil(observations, soil_class, weights, settings):
    values = {name: float(np.mean([o.class_means[soil_class] for o in flights]))
              for name, flights in _group(observations).items()}
    return _estimate(values, observations,
                     "equal_date_contribution_mean" if settings.get("quantity") == "jiang_contribution" else "equal_date_class_mean", weights,
                     ["constant_soil_background_assumed"], settings)


def _lambda(observations, soil_class, woody_class, backgrounds, weights, epsilon, settings):
    grouped = _group(observations)
    contribution = settings.get("quantity") == "jiang_contribution"
    if contribution:
        if backgrounds is not None:
            raise ValueError("Do not supply soil_background for D_woody: soil is already excluded")
        backgrounds = dict.fromkeys(grouped, 0.0)
        background_method = "none_contribution_already_excludes_soil"
    elif backgrounds is None:
        backgrounds = {k: float(np.mean([o.class_means[soil_class] for o in v]))
                       for k, v in grouped.items()}
        background_method = "local_equal_date_soil_mean"
    elif isinstance(backgrounds, Mapping):
        if set(backgrounds) != set(grouped):
            raise ValueError("soil_background mapping must match the location names")
        backgrounds = dict(backgrounds)
        background_method = "supplied_per_location"
    else:
        backgrounds = dict.fromkeys(grouped, _finite(backgrounds, "soil_background"))
        background_method = "supplied_shared_scalar"
    values, flags = {}, ["observed_range_proxy", "shared_seasonal_phase_not_tested",
                          "stable_canopy_support_assumed"]
    if contribution:
        flags.append("contribution_variation_includes_fraction_and_brightness")
    for name, flights in grouped.items():
        if len(flights) < 2:
            raise ValueError(f"{name}: lambda requires at least two distinct flight dates")
        d = _finite(backgrounds[name], f"soil_background for {name}")
        backgrounds[name] = d
        series = [o.class_means[woody_class] for o in flights]
        baseline = min(series) - d
        if baseline <= epsilon:
            raise ValueError(f"{name}: woody baseline minus soil background must exceed {epsilon}")
        values[name] = (max(series) - min(series)) / baseline
        span = (datetime.strptime(flights[-1].date, "%Y%m%d") -
                datetime.strptime(flights[0].date, "%Y%m%d")).days
        if span < 365:
            flags.append(f"{name}: less_than_one_year_observed")
        if span > 366:
            flags.append(f"{name}: multi_year_range_may_include_trend")
        if len(flights) < 6:
            flags.append(f"{name}: sparse_seasonal_sampling")
    settings = {**settings, "soil_background_method": background_method,
                "soil_backgrounds": backgrounds, "baseline_epsilon": epsilon}
    return _estimate(values, observations, "contribution_observed_range" if contribution else "background_corrected_observed_range",
                     weights, flags, settings)


def define_soil(parent_folder, dates=None, *, vi="NDVI", soil_class="soil",
                min_coverage=0.9, mask=None, location_weights=None, quantity="class_mean"):
    """Estimate soil background: equal-date means, then equal-location pooling."""
    observations = summarize_flights(parent_folder, dates, vi=vi, classes=(soil_class,),
                                     min_coverage=min_coverage, mask=mask, quantity=quantity)
    return _soil(observations, soil_class, location_weights,
                 {"soil_class": soil_class, "min_coverage": float(min_coverage), "quantity": quantity})


def define_lambda(parent_folder, dates=None, *, vi="NDVI", soil_background=None,
                  soil_class="soil", woody_class="olive", min_coverage=0.9,
                  mask=None, location_weights=None, baseline_epsilon=1e-6,
                  quantity="class_mean"):
    """Estimate (max woody VI - min woody VI) / (min woody VI - soil).

    With quantity='jiang_contribution', use range(D_woody)/min(D_woody).
    No soil is subtracted and supplying soil_background is an error.
    The default class_mean mode retains the legacy estimator below.
    Without soil_background, estimate soil separately within each location.
    A scalar applies everywhere; a mapping supplies each location's background.
    Supplying a background removes the need for soil bands. Compute each local
    lambda before pooling: differences between orchard means are not seasonality.
    """
    epsilon = _finite(baseline_epsilon, "baseline_epsilon", positive=True)
    if soil_class == woody_class:
        raise ValueError("soil_class and woody_class must differ")
    classes = (soil_class, woody_class) if soil_background is None and quantity == "class_mean" else (woody_class,)
    observations = summarize_flights(parent_folder, dates, vi=vi, classes=classes,
                                     min_coverage=min_coverage, mask=mask, quantity=quantity)
    return _lambda(observations, soil_class, woody_class, soil_background, location_weights,
                   epsilon, {"soil_class": soil_class, "woody_class": woody_class,
                             "min_coverage": float(min_coverage), "quantity": quantity})


def define_background(parent_folder, dates=None, *, vi="NDVI", scope="general",
                      soil_class="soil", woody_class="olive", min_coverage=0.9,
                      mask=None, location_weights=None, baseline_epsilon=1e-6,
                  quantity="class_mean"):
    """Build both parameters in one read of the selected flights.

    General calibration pools within-location estimates, equally by default.
    Custom positive location_weights change the intended orchard population.
    scope='local' requires one location; scope='general' permits a single-location
    starting estimate but flags that transfer cannot yet be evaluated.
    In class_mean mode lambda uses each orchard's own soil estimate.
    With quantity='jiang_contribution', soil is the equal-date mean D_soil,
    and lambda is range(D_woody)/min(D_woody), with no second soil subtraction.
    Spatial summaries are equal-cell means, not medians or pooled-reflectance
    NDVI. Match the satellite footprint and spatial statistic before comparison.
    """
    if scope not in {"general", "local"}:
        raise ValueError("scope must be 'general' or 'local'")
    if soil_class == woody_class:
        raise ValueError("soil_class and woody_class must differ")
    epsilon = _finite(baseline_epsilon, "baseline_epsilon", positive=True)
    observations = summarize_flights(parent_folder, dates, vi=vi, classes=(soil_class, woody_class),
                                     min_coverage=min_coverage, mask=mask, quantity=quantity)
    if scope == "local" and len(_group(observations)) != 1:
        raise ValueError("scope='local' requires exactly one location")
    settings = {"soil_class": soil_class, "woody_class": woody_class,
                "min_coverage": float(min_coverage), "quantity": quantity}
    soil = _soil(observations, soil_class, location_weights, settings)
    woody = _lambda(observations, soil_class, woody_class, None, location_weights,
                    epsilon, settings)
    return CalibrationResult(observations[0].vi, scope, soil, woody)
