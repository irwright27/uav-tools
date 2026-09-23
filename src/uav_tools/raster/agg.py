"""Review draft: area-weighted UAV attribution on an explicit S2 parent grid.

Only numpy and rasterio are required. No resampling or coordinate inference.
Areas are planar square CRS units (normally m² in S2 UTM), not geodesic areas.
Pixels are treated as uniform rectangles. No S2 data/masks are read implicitly.

D = NDVI attribution after reflectance aggregation; L = attribution of the
mean fine-pixel NDVI. Neither is standalone class NDVI or causal attribution.
All fractions/contributions describe observed support, not unobserved area.
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from numbers import Integral
import json
import math
import os
import tempfile

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine, array_bounds
from rasterio.windows import Window


@dataclass(frozen=True)
class Grid:
    crs: CRS
    transform: Affine
    width: int
    height: int

    def __post_init__(self):
        object.__setattr__(self, 'crs', CRS.from_user_input(self.crs))
        t = self.transform
        if not isinstance(t, Affine) or not np.isfinite(tuple(t)).all():
            raise ValueError('A finite affine transform is required')
        if t.b != 0 or t.d != 0 or t.a <= 0 or t.e >= 0:
            raise ValueError('Only north-up, unrotated grids are supported')
        if not self.crs.is_projected:
            raise ValueError('Projected CRS required for planar overlap areas')
        if any(not isinstance(n, Integral) or n <= 0 for n in (self.width, self.height)):
            raise ValueError('Positive integer grid dimensions required')

    @property
    def bounds(self):
        return array_bounds(self.height, self.width, self.transform)

    @property
    def pixel_area(self):
        return -self.transform.a * self.transform.e


def reference_grid(reference) -> Grid:
    """Accept Grid, raster path/open raster, ODC GeoBox or ODC xarray object.

    s2_tools.load_s2 returns an xarray Dataset with .odc.geobox. This adapter
    reads geometry only, leaving time/bands/dask data untouched. That loaded
    grid is authoritative; it need not equal the original ESA tile grid.
    Bare resolution, coordinate arrays and ungeoreferenced objects are rejected.
    """
    if isinstance(reference, Grid):
        return reference
    if isinstance(reference, (str, os.PathLike)):
        with rasterio.open(reference) as ds:
            return reference_grid(ds)
    odc = getattr(reference, 'odc', None)
    if odc is not None:
        reference = odc.geobox
        if reference is None:
            raise ValueError('S2 object has no ODC geobox')
    if reference is None or any(not hasattr(reference, k) for k in
                                ('crs', 'transform', 'width', 'height')):
        raise TypeError('Supply a raster, ODC spatial object or explicit Grid')
    crs = reference.crs
    if crs is None:
        raise ValueError('Reference CRS is missing')
    return Grid(CRS.from_user_input(str(crs)), reference.transform,
                int(reference.width), int(reference.height))


@dataclass(frozen=True)
class Band:
    """Raster path plus 1-based band index. Raster scale/offset are decoded."""
    path: str | Path
    index: int = 1


@dataclass
class Cell:
    row: int                       # cropped output row
    col: int                       # cropped output column
    values: dict[str, float]


@dataclass
class Aggregation:
    grid: Grid
    parent_window: Window
    metadata: dict
    _factory: object

    def __iter__(self):
        """Replayable stream; each iteration reopens sources read-only."""
        return self._factory()


def _window(grid, bounds):
    left, bottom, right, top = bounds
    t = grid.transform
    # Deliberately no broad rounding tolerance: actual positive slivers count.
    c0 = max(0, math.floor((left - t.c) / t.a))
    c1 = min(grid.width, math.ceil((right - t.c) / t.a))
    r0 = max(0, math.floor((top - t.f) / t.e))
    r1 = min(grid.height, math.ceil((bottom - t.f) / t.e))
    if c1 <= c0 or r1 <= r0:
        raise ValueError('Grids have no positive-area intersection')
    return Window(c0, r0, c1-c0, r1-r0)


def _groups(groups):
    if not groups:
        raise ValueError('Supply at least one named class-code group')
    out, seen = {}, set()
    for name, codes in groups.items():
        if not isinstance(name, str) or not name or '/' in name or name == 'other':
            raise ValueError('Names must be nonempty, slash-free; other is reserved')
        codes = tuple(codes)
        if not codes or any(not isinstance(c, Integral) or isinstance(c, bool) for c in codes):
            raise ValueError('Class groups must contain integer codes')
        if len(set(codes)) != len(codes) or seen.intersection(codes):
            raise ValueError('Class-code groups overlap or contain duplicates')
        seen.update(codes)
        out[name] = codes
    return out


def _read(ds, index, window, physical=True):
    a = ds.read(index, window=window, masked=True)
    if not physical and np.issubdtype(a.dtype, np.integer):
        raw_valid = ~np.ma.getmaskarray(a)
        if np.any(raw_valid & ((a.data > 2**53) | (a.data < -2**53))):
            raise ValueError('Classification codes exceed exact float64 integer range')
    x = np.asarray(a.data, dtype=np.float64)
    valid = ~np.ma.getmaskarray(a) & np.isfinite(x)
    if physical:
        scale, offset = ds.scales[index-1], ds.offsets[index-1]
        if not np.isfinite([scale, offset]).all():
            raise ValueError('Nonfinite scale/offset metadata')
        with np.errstate(over='ignore', invalid='ignore'):
            x = x * scale + offset
        valid &= np.isfinite(x)
    return x, valid


def aggregate_reflectance(reference, classification, *, red, nir,
                          class_groups, **kwargs):
    """Method 1 on valid red/NIR/labels; fine zero denominators remain valid.

    Pass red/nir as Band(path, index). Returns a replayable bounded-memory
    Aggregation. See aggregate for coverage, nodata and grid constraints.
    """
    return aggregate(reference, classification, red=red, nir=nir,
                     class_groups=class_groups, method='reflectance', **kwargs)


def aggregate_ndvi(reference, classification, *, class_groups, ndvi=None,
                   red=None, nir=None, **kwargs):
    """Method 2 from an existing NDVI Band, or calculated from red/nir Bands.

    When red/nir accompany existing NDVI they also constrain valid support.
    Use aggregate(method='both') for a guaranteed paired comparison.
    """
    return aggregate(reference, classification, red=red, nir=nir, ndvi=ndvi,
                     class_groups=class_groups, method='ndvi', **kwargs)


def aggregate(reference, classification, *, class_groups, red=None, nir=None,
              ndvi=None, method='both', min_coverage=0.0, block_size=512,
              denominator_epsilon=1e-12, classification_nodata=(),
              reflectance_range=(0.0, None), ndvi_range=(-1.0, 1.0),
              quality_valid=None):
    """Shared exact rectangular-overlap engine.

    classification: single-band raster path; raw integer codes, never scaled.
    class_groups: mapping name -> iterable of codes; unmatched valid codes go
        to 'other'. Band masks/NoData are honored, including class zero only
        when marked NoData. classification_nodata adds explicitly invalid codes.
    method: reflectance, ndvi, or both. Both intersects label, red, NIR and NDVI
        validity (including abs(NIR+red)>epsilon if NDVI is derived). Thus its
        D support can differ from a standalone reflectance call. With external
        NDVI, validity of that raster defines NDVI support; it is not recomputed.
    ranges: decoded physical-value limits, inclusive; None disables a bound.
        Default rejects negative reflectance, allows reflectance >1; NDVI [-1,1].
    min_coverage: threshold relative to FULL S2 footprint. Below threshold,
        science metrics become NaN, but area/fraction diagnostics remain.
    quality_valid: optional callable(parent_row, parent_col)->bool. False masks
        science results without changing geometry or UAV coverage diagnostics.
        Select an S2 time explicitly outside this function. No implicit SCL use.

    All UAV sources must have EXACTLY equal CRS/transform/shape. Reference must
    share the projected CRS, but may have a different origin and resolution.
    Rotation/shear and geographic CRSs are rejected. No inputs are reprojected.
    Working arrays are at most block_size² plus O(number of classes). GDAL has
    its own bounded block cache. Iteration is per S2 cell; very small S2 pixels
    may cause repeated source block reads. This is a correctness-first draft.

    Empty cells: valid_area/coverage/class area/full_fraction=0;
    class common_fraction and all science metrics=NaN. Absent class within an
    accepted usable cell: contribution=0, class means=NaN. Invalid aggregate
    denominator: D and reflectance_ndvi=NaN for ALL classes, with flag=0.
    """
    parent = reference_grid(reference)
    groups = _groups(class_groups)
    if method not in {'both', 'reflectance', 'ndvi'}:
        raise ValueError('Unknown method')
    if (red is None) != (nir is None):
        raise ValueError('Supply both red and nir')
    if method in {'both', 'reflectance'} and red is None:
        raise ValueError('Reflectance attribution requires red and nir')
    if method == 'reflectance' and ndvi is not None:
        raise ValueError('Use method=both to constrain reflectance with NDVI')
    if method == 'ndvi' and red is None and ndvi is None:
        raise ValueError('Supply NDVI or red/nir')
    if not np.isfinite(min_coverage) or not 0 <= min_coverage <= 1:
        raise ValueError('min_coverage must lie in [0,1]')
    if not isinstance(block_size, Integral) or block_size < 1:
        raise ValueError('block_size must be a positive integer')
    if not np.isfinite(denominator_epsilon) or denominator_epsilon < 0:
        raise ValueError('denominator_epsilon must be finite and nonnegative')
    for limits in (reflectance_range, ndvi_range):
        if len(limits) != 2 or any(v is not None and not np.isfinite(v) for v in limits):
            raise ValueError('Ranges require two finite bounds or None')
        if all(v is not None for v in limits) and limits[0] > limits[1]:
            raise ValueError('Range bounds are reversed')
    if quality_valid is not None and not callable(quality_valid):
        raise TypeError('quality_valid must be a callable')
    extra_nodata = tuple(classification_nodata)
    if any(not isinstance(c, Integral) for c in extra_nodata):
        raise ValueError('classification_nodata must contain integer codes')
    sources = {k:v for k,v in [('red',red), ('nir',nir), ('ndvi',ndvi)] if v is not None}
    if any(not isinstance(v, Band) for v in sources.values()):
        raise TypeError('Use Band(path, 1-based index) for spectral inputs')

    def open_sources(stack):
        cls = stack.enter_context(rasterio.open(classification))
        source_grid = reference_grid(cls)
        if cls.count != 1:
            raise ValueError('Classification must be single-band')
        if source_grid.crs != parent.crs:
            raise ValueError('UAV and S2 CRSs differ; explicit preprocessing required')
        bands = {}
        for name, band in sources.items():
            ds = stack.enter_context(rasterio.open(band.path))
            if reference_grid(ds) != source_grid:
                raise ValueError(f'{name} and classification grids differ')
            if not isinstance(band.index, Integral) or not 1 <= band.index <= ds.count:
                raise ValueError(f'{name}: invalid band index')
            bands[name] = (ds, band.index)
        return cls, source_grid, bands

    with ExitStack() as stack:
        cls, source_grid, bands = open_sources(stack)
        parent_window = _window(parent, source_grid.bounds)
        encoding = {k: {'path': str(sources[k].path), 'band': i,
                        'scale': ds.scales[i-1], 'offset': ds.offsets[i-1],
                        'nodata': str(ds.nodatavals[i-1])}
                    for k,(ds,i) in bands.items()}
    out_grid = Grid(parent.crs, parent.transform * Affine.translation(
        parent_window.col_off, parent_window.row_off),
        int(parent_window.width), int(parent_window.height))
    names = [*groups, 'other']
    want_d, want_l = method != 'ndvi', method != 'reflectance'

    def in_range(x, limits):
        lo, hi = limits
        return (np.ones(x.shape, bool) if lo is None else x >= lo) & (
            np.ones(x.shape, bool) if hi is None else x <= hi)

    def cells():
        with ExitStack() as stack:
            cls, src, bands = open_sources(stack)
            if src != source_grid:
                raise ValueError('Source geometry changed since planning')
            for row in range(out_grid.height):
                for col in range(out_grid.width):
                    t = out_grid.transform * Affine.translation(col, row)
                    left, right, top, bottom = t.c, t.c+t.a, t.f, t.f+t.e
                    w = _window(src, (left, bottom, right, top))
                    sums = np.zeros((len(names), 4), dtype=np.float64)  # area, red, nir, ndvi
                    for rr in range(int(w.row_off), int(w.row_off+w.height), block_size):
                        for cc in range(int(w.col_off), int(w.col_off+w.width), block_size):
                            h = min(block_size, int(w.row_off+w.height)-rr)
                            width = min(block_size, int(w.col_off+w.width)-cc)
                            win = Window(cc, rr, width, h)
                            labels, valid = _read(cls, 1, win, physical=False)
                            valid &= ~np.isin(labels, extra_nodata)
                            if np.any(valid & ((labels != np.floor(labels)) | (np.abs(labels) > 2**53))):
                                raise ValueError('Valid labels must be exactly representable integers <=2**53')
                            data = {}
                            for name, (ds, index) in bands.items():
                                x, ok = _read(ds, index, win)
                                ok &= in_range(x, ndvi_range if name == 'ndvi' else reflectance_range)
                                valid &= ok
                                data[name] = x
                            if want_l and 'ndvi' not in data:
                                den = data['red'] + data['nir']
                                ok = np.isfinite(den) & (np.abs(den) > denominator_epsilon)
                                x = np.full(den.shape, np.nan)
                                np.divide(data['nir']-data['red'], den, out=x, where=ok)
                                valid &= ok & np.isfinite(x) & in_range(x, ndvi_range)
                                data['ndvi'] = x
                            st = src.transform
                            xs = st.c + np.arange(cc, cc+width)*st.a
                            ys = st.f + np.arange(rr, rr+h)*st.e
                            dx = np.maximum(0, np.minimum(xs+st.a, right)-np.maximum(xs, left))
                            dy = np.maximum(0, np.minimum(ys, top)-np.maximum(ys+st.e, bottom))
                            area = dy[:,None]*dx[None,:]
                            group_id = np.full(labels.shape, len(groups), dtype=np.int32)
                            for j, codes in enumerate(groups.values()):
                                group_id[np.isin(labels, codes)] = j
                            for j in range(len(names)):
                                keep = valid & (group_id == j) & (area > 0)
                                a = area[keep]
                                sums[j,0] += a.sum()
                                for k, name in enumerate(('red','nir','ndvi'), 1):
                                    if name in data:
                                        sums[j,k] += np.dot(a, data[name][keep])
                    total = sums[:,0].sum()
                    coverage = total / out_grid.pixel_area
                    qr = row + int(parent_window.row_off)
                    qc = col + int(parent_window.col_off)
                    quality = quality_valid is None or bool(quality_valid(qr, qc))
                    accepted = total > 0 and coverage >= min_coverage and quality
                    vals = dict(valid_area=total, coverage=coverage,
                                quality_valid=float(quality), accepted=float(accepted))
                    mean = sums[:,1:].sum(axis=0)/total if total > 0 else np.full(3,np.nan)
                    denominator = mean[0]+mean[1]
                    den_ok = bool(np.isfinite(denominator) and abs(denominator)>denominator_epsilon)
                    if want_d:
                        vals.update(red_mean=mean[0] if accepted else np.nan,
                                    nir_mean=mean[1] if accepted else np.nan,
                                    denominator_valid=float(total > 0 and den_ok),
                                    reflectance_ndvi=(mean[1]-mean[0])/denominator
                                    if accepted and den_ok else np.nan)
                    if want_l:
                        vals['mean_fine_ndvi'] = mean[2] if accepted else np.nan
                    for j, name in enumerate(names):
                        a, r, n, v = sums[j]
                        metrics = dict(valid_area=a, common_fraction=a/total if total>0 else np.nan,
                                       full_fraction=a/out_grid.pixel_area)
                        if want_d:
                            metrics.update(red_mean=r/a if accepted and a>0 else np.nan,
                                nir_mean=n/a if accepted and a>0 else np.nan,
                                C_red=r/total if accepted else np.nan,
                                C_nir=n/total if accepted else np.nan,
                                D=((n/total-r/total)/denominator) if accepted and den_ok else np.nan)
                        if want_l:
                            metrics.update(ndvi_mean=v/a if accepted and a>0 else np.nan,
                                           L=v/total if accepted else np.nan)
                        vals.update({f'{name}/{k}':v for k,v in metrics.items()})
                    yield Cell(row, col, vals)
    metadata = dict(method=method, support='intersection of labels and all supplied/derived inputs',
        class_groups={k:[int(c) for c in v] for k,v in groups.items()}, other='all valid unrequested codes',
        classification=str(classification), classification_nodata=[int(c) for c in extra_nodata],
        source_transform=list(source_grid.transform),
        source_shape=[source_grid.height,source_grid.width],
        encoding=encoding, min_coverage=min_coverage, denominator_epsilon=denominator_epsilon,
        reflectance_range=reflectance_range, ndvi_range=ndvi_range,
        quality_filter_supplied=quality_valid is not None, area_units='square projected CRS units',
        parent_crs=str(parent.crs), parent_transform=list(parent.transform),
        parent_shape=[parent.height,parent.width],
        parent_window=[int(parent_window.col_off),int(parent_window.row_off),
                       int(parent_window.width),int(parent_window.height)])
    return Aggregation(out_grid, parent_window, metadata, cells)


def write_tif(result: Aggregation, output_path):
    """Stream float64 metrics to one GeoTIFF, with descriptions and JSON tags.

    Diagnostics survive threshold/quality rejection. NaN is NoData; zeros remain
    valid. Output scale/offset are identity. Never overwrite existing outputs.
    A same-directory temporary file is atomically linked into place on success.
    No output is produced merely by constructing/iterating an Aggregation.
    """
    path = Path(output_path)
    if path.exists():
        raise FileExistsError(path)
    iterator = iter(result)
    try:
        first = next(iterator)
        keys = list(first.values)
        fd, tmp = tempfile.mkstemp(prefix='.agg-', suffix='.tif', dir=path.parent)
        os.close(fd)
        try:
            g = result.grid
            with rasterio.open(tmp, 'w', driver='GTiff', width=g.width, height=g.height,
                               count=len(keys), dtype='float64', nodata=np.nan,
                               crs=g.crs, transform=g.transform, compress='deflate',
                               BIGTIFF='IF_SAFER') as dst:
                for i,k in enumerate(keys,1):
                    dst.set_band_description(i,k)
                dst.update_tags(aggregation=json.dumps(result.metadata))
                def put(cell):
                    dst.write(np.array([cell.values[k] for k in keys]).reshape(-1,1,1),
                              window=Window(cell.col,cell.row,1,1))
                put(first)
                for cell in iterator:
                    put(cell)
            os.link(tmp,path)  # atomic no-clobber; same filesystem
        finally:
            os.unlink(tmp)
    finally:
        iterator.close()
    return path