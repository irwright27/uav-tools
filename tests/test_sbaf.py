import unittest
import tempfile
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin
from spec_tools import Spectrum
from uav_tools.spectral.srf import BandSRF, SensorSRF
from uav_tools.spectral.sbaf import calculate_sbaf, apply_sbaf_array, apply_sbaf_tif

class SBAFTests(unittest.TestCase):
    def setUp(self):
        self.spectrum = Spectrum('slope', [400, 500, 600], [.1, .2, .3], 'reflectance', value_unit='1')
        self.source = BandSRF.from_curve('source', [400, 500], [1, 1])
        self.target = BandSRF.from_curve('target', [500, 600], [1, 1])
        self.kw = dict(class_spectra={1: self.spectrum},
                       source_sensor=SensorSRF('uav', {'source': self.source}),
                       target_sensor=SensorSRF('target', {'target': self.target}),
                       band_pairs={1: ('source', 'target')})

    def test_direction_and_identity(self):
        self.assertAlmostEqual(calculate_sbaf(self.spectrum, self.source, self.target), 5/3)
        self.assertAlmostEqual(calculate_sbaf(self.spectrum, self.source, self.source), 1)
        self.assertAlmostEqual(calculate_sbaf(self.spectrum, self.source.normalized_copy(), self.target), 5/3)

    def test_missing_coverage_units_and_zero(self):
        for change in ('missing', 'coverage', 'unit', 'quantity', 'zero'):
            with self.subTest(change=change):
                s = self.spectrum.copy()
                if change == 'missing': s.values[1] = np.nan
                if change == 'coverage': s.wavelengths[0] = 450
                if change == 'unit': s.wavelength_unit = 'um'
                if change == 'quantity': s.quantity = 'radiance'
                if change == 'zero': s.values[:] = 0
                with self.assertRaises(ValueError): calculate_sbaf(s, self.source, self.target)

    def test_classes_masks_unmapped_and_no_mutation(self):
        data = np.ones((2, 2, 2))
        cls = np.ma.array([[1, 2], [3, 1]], mask=[[0, 0], [0, 1]])
        table = {1: {1: 2}, 2: {1: 3}}
        with self.assertRaises(ValueError): apply_sbaf_array(data, cls, table)
        out = apply_sbaf_array(data, cls, table, unmapped='preserve')
        np.testing.assert_array_equal(out.data[0], [[2, 3], [1, 1]])
        self.assertTrue(out.mask[0, 1, 1])
        self.assertFalse(out.mask[1].any())
        np.testing.assert_array_equal(data, 1)

    def test_exact_product_integration(self):
        # Integral of (.1 + .001*t)*(t/100), t=0..100, divided by 50.
        ramp = BandSRF.from_curve('ramp', [400, 500], [0, 1])
        self.assertAlmostEqual(calculate_sbaf(self.spectrum, self.source, ramp), 10/9)

    def test_tif_roundtrip_and_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            src, cls, out = [root / s for s in ('uav.tif', 'classes.tif', 'out.tif')]
            profile = dict(driver='GTiff', width=3, height=2, count=2, dtype='int16',
                           crs='EPSG:32610', transform=from_origin(100, 200, 1, 1), nodata=-999)
            pixels = np.full((2, 2, 3), 100, dtype=np.int16)
            pixels[0, 0, 0] = -999
            with rasterio.open(src, 'w', **profile) as dst:
                dst.write(pixels)
                dst.scales = (.001, .002)
                dst.offsets = (.1, .2)
                dst.set_band_description(1, 'blue')
            profile.update(count=1)
            classes = np.ones((1, 2, 3), dtype=np.int16)
            classes[0, 1, 2] = -999
            with rasterio.open(cls, 'w', **profile) as dst: dst.write(classes)
            result = apply_sbaf_tif(src, cls, out, **self.kw)
            self.assertTrue(result.report_path.exists())
            with rasterio.open(out) as dst:
                data = dst.read(masked=True)
                self.assertAlmostEqual(float(data[0, 0, 1]), 1/3, places=6)
                self.assertAlmostEqual(float(data[1, 0, 1]), .4, places=6)
                self.assertTrue(data.mask[0, 0, 0])
                self.assertTrue(data.mask[0, 1, 2])
                self.assertFalse(data.mask[1, 1, 2])
                self.assertEqual(dst.transform, profile['transform'])
                self.assertEqual(dst.scales, (1, 1))
                self.assertEqual(dst.offsets, (0, 0))
                self.assertEqual(dst.descriptions[0], 'blue')
            with self.assertRaises(FileExistsError): apply_sbaf_tif(src, cls, out, **self.kw)
            classes[:] = 9
            with rasterio.open(cls, 'w', **profile) as dst: dst.write(classes)
            failed = root / 'failed.tif'
            with self.assertRaises(ValueError): apply_sbaf_tif(src, cls, failed, **self.kw)
            self.assertFalse(failed.exists())
            self.assertFalse(failed.with_suffix('.tif.sbaf.json').exists())
            profile.update(transform=from_origin(101, 200, 1, 1))
            with rasterio.open(cls, 'w', **profile) as dst: dst.write(classes)
            with self.assertRaisesRegex(ValueError, 'match UAV'): apply_sbaf_tif(src, cls, failed, **self.kw)

if __name__ == '__main__': unittest.main()
