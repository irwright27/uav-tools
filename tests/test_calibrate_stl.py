import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from uav_tools.calibrate_stl import (
    define_background, define_lambda, define_soil, summarize_flights,
)


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def flight(self, name, day, soil=.2, olive=.6, vi="NDVI", extra=None,
               scales=None, offsets=None):
        root = self.root / name / "auto_processed"
        folder = root / day / "agg"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"RANCH_BLOCK_{name}_{day}_APT_{vi}_AGG_jiang.tif"
        data = {f"mean_{vi}_soil": [soil, soil], f"mean_{vi}_olive": [olive, olive],
                "f_full_soil": [.5, .5], "f_full_olive": [.5, .5],
                "valid_coverage_class": [1, 1], "D_soil": [99, 99]}
        data.update(extra or {})
        with rasterio.open(path, "w", driver="GTiff", width=2, height=1,
                           count=len(data), dtype="float64", crs="EPSG:32610",
                           transform=from_origin(100, 200, 10, 10), nodata=-999) as dst:
            for i, (key, values) in enumerate(data.items(), 1):
                dst.write(np.array(values, dtype=float).reshape(1, 2), i)
                dst.set_band_description(i, key)
            dst.update_tags(uav_date=day)
            if scales:
                dst.scales = scales
            if offsets:
                dst.offsets = offsets
        return root, path

    def test_area_then_date_weighting_and_proxy(self):
        root, _ = self.flight("A", "20250101", extra={
            "mean_NDVI_soil": [.1, .3], "f_full_soil": [.25, .75]})
        self.flight("A", "20250701", soil=.35, olive=.9)
        cal = define_background(root)
        self.assertAlmostEqual(cal.soil_background, .3)
        self.assertAlmostEqual(cal.lambda_woody, 1.)
        self.assertAlmostEqual(cal.observations[0].class_areas["soil"], 100.)
        self.assertAlmostEqual(cal.soil.location_diagnostics["A"]["classes"]["soil"]["range"], .1)
        self.assertEqual(cal.woody.method, "background_corrected_observed_range")
        self.assertIsNone(cal.woody.bootstrap_ci)

    def test_orchard_baselines_are_not_seasonality(self):
        a, _ = self.flight("A", "20250101", olive=.4)
        self.flight("A", "20250701", olive=.4)
        b, _ = self.flight("B", "20250101", olive=.9)
        self.flight("B", "20250701", olive=.9)
        cal = define_background({"a": a, "b": b})
        self.assertEqual(cal.lambda_woody, 0)
        self.assertEqual(cal.woody.bootstrap_ci, (0., 0.))
        self.assertEqual(cal.woody.leave_one_out["a"]["training_estimate"], 0)

    def test_equal_orchard_weight_despite_more_dates_and_custom_weights(self):
        a, _ = self.flight("A", "20250101", soil=.1, olive=.5)
        self.flight("A", "20250401", soil=.1, olive=.6)
        self.flight("A", "20250701", soil=.1, olive=.7)
        b, _ = self.flight("B", "20250101", soil=.3, olive=.5)
        self.flight("B", "20250701", soil=.3, olive=.7)
        inputs = {"a": a, "b": b}
        cal = define_background(inputs)
        self.assertAlmostEqual(cal.soil_background, .2)
        self.assertAlmostEqual(cal.woody.per_location["a"], .5)
        self.assertAlmostEqual(cal.woody.per_location["b"], 1.)
        self.assertAlmostEqual(cal.lambda_woody, .75)
        self.assertAlmostEqual(cal.woody.leave_one_out["a"]["error"], .5)
        weighted = define_background(inputs, location_weights={"a": 3, "b": 1})
        self.assertAlmostEqual(weighted.soil_background, .15)
        self.assertAlmostEqual(weighted.lambda_woody, .625)

    def test_generic_vi_and_right_hand_filename_parsing(self):
        root, _ = self.flight("MANY_UNDERSCORES", "20250101", soil=1.2, olive=2., vi="EVI")
        self.flight("MANY_UNDERSCORES", "20250701", soil=1.2, olive=2.4, vi="EVI")
        self.flight("MANY_UNDERSCORES", "20250101", soil=.1, olive=.5, vi="NDVI")
        cal = define_background(root, vi="evi")
        self.assertEqual(cal.vi, "EVI")
        self.assertAlmostEqual(cal.soil_background, 1.2)
        self.assertAlmostEqual(cal.lambda_woody, .5)

    def test_mask_nodata_and_coverage(self):
        root, _ = self.flight("A", "20250101", extra={
            "mean_NDVI_soil": [.2, -999], "mean_NDVI_olive": [.6, .9]})
        self.assertAlmostEqual(define_soil(root).value, .2)
        self.flight("A", "20250701", extra={
            "valid_coverage_class": [1, .4], "f_full_soil": [.5, .2],
            "f_full_olive": [.5, .2], "mean_NDVI_soil": [.3, 8]})
        self.assertAlmostEqual(define_soil(root).value, .25)

    def test_footprint_csv_and_explicit_mask(self):
        root, path = self.flight("A", "20250101", extra={"mean_NDVI_soil": [.2, 9.]})
        csvpath = path.with_name("RANCH_BLOCK_A_20250101_aggregation_class_metrics.csv")
        with csvpath.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerows([["row", "col", "orchard_footprint"], [0, 0, True], [0, 1, False]])
        result = define_soil(root)
        self.assertAlmostEqual(result.value, .2)
        self.assertEqual(result.observations[0].footprint_source, str(csvpath.resolve()))
        mask = self.root / "mask.tif"
        with rasterio.open(path) as src:
            profile = src.profile
        profile.update(count=1)
        with rasterio.open(mask, "w", **profile) as dst:
            dst.write(np.array([[0, 1]], dtype=float), 1)
        self.assertEqual(define_soil(root, mask=mask).value, 9.)

    def test_footprint_band(self):
        root, _ = self.flight("A", "20250101", extra={
            "mean_NDVI_soil": [.2, 9.], "orchard_footprint": [1, 0]})
        self.assertAlmostEqual(define_soil(root).value, .2)

    def test_scale_offset_decoding(self):
        root, _ = self.flight("A", "20250101", soil=100, olive=300,
                              scales=(.001, .001, 1, 1, 1, 1),
                              offsets=(.1, .1, 0, 0, 0, 0))
        self.assertAlmostEqual(define_soil(root).value, .2)
        self.assertAlmostEqual(summarize_flights(root)[0].class_means["olive"], .4)

    def test_explicit_dates_per_location_and_discovery(self):
        root, _ = self.flight("A", "20250101")
        self.flight("A", "20250701", soil=.4)
        (root / "20250301").mkdir()
        self.assertEqual(len(summarize_flights(root)), 2)
        cal = define_soil({"a": {"parent_folder": root, "dates": ["2025-01-01"]}})
        self.assertEqual(cal.value, .2)
        with self.assertRaises(ValueError):
            define_soil(root, ["20250301"])
        with self.assertRaises(ValueError):
            define_soil(root, ["20250101", "2025-01-01"])

    def test_ambiguous_product_rejected(self):
        root, path = self.flight("A", "20250101")
        path.with_name("OTHER_" + path.name).write_bytes(path.read_bytes())
        with self.assertRaisesRegex(ValueError, "found 2"):
            define_soil(root)

    def test_wrong_vi_band_rejected(self):
        root, path = self.flight("A", "20250101")
        path.rename(path.with_name(path.name.replace("_NDVI_", "_GNDVI_")))
        with self.assertRaisesRegex(ValueError, "mean_GNDVI_soil"):
            define_soil(root, vi="GNDVI")

    def test_one_flight_supports_soil_but_not_lambda(self):
        root, _ = self.flight("A", "20250101")
        self.assertEqual(define_soil(root).value, .2)
        with self.assertRaisesRegex(ValueError, "two distinct"):
            define_lambda(root)

    def test_invalid_baseline_and_supplied_backgrounds(self):
        root, _ = self.flight("A", "20250101")
        self.flight("A", "20250701", olive=.8)
        with self.assertRaisesRegex(ValueError, "baseline"):
            define_lambda(root, soil_background=.6)
        self.assertAlmostEqual(define_lambda(root, soil_background=.1).value, .4)
        self.assertAlmostEqual(define_lambda({"a": root}, soil_background={"a": .2}).value, .5)
        with self.assertRaises(ValueError):
            define_lambda(root, soil_background=True)

    def test_supplied_background_does_not_require_soil_band(self):
        root, p1 = self.flight("A", "20250101")
        _, p2 = self.flight("A", "20250701", olive=.8)
        for path in (p1, p2):
            with rasterio.open(path, "r+") as dst:
                dst.set_band_description(1, "unrelated")
        self.assertAlmostEqual(define_lambda(root, soil_background=.2).value, .5)

    def test_report_roundtrip_and_no_overwrite(self):
        root, _ = self.flight("A", "20250101")
        self.flight("A", "20250701", olive=.8)
        result = define_background(root, scope="local")
        path = result.save(self.root / "calibration.json")
        saved = json.loads(path.read_text())
        self.assertAlmostEqual(saved["lambda_woody"], .5)
        self.assertEqual(saved["woody"]["settings"]["soil_backgrounds"], {"A": .2})
        self.assertEqual(len(saved["soil"]["observations"]), 2)
        with self.assertRaises(FileExistsError):
            result.save(path)

    def test_invalid_configurations(self):
        root, _ = self.flight("A", "20250101")
        self.flight("A", "20250701")
        for kw in [{"min_coverage": 2}, {"min_coverage": True}, {"vi": "../NDVI"},
                   {"location_weights": {"A": -1}}, {"location_weights": {}},
                   {"scope": "typo"}, {"baseline_epsilon": 0}]:
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                define_background(root, **kw)
        with self.assertRaises(ValueError):
            define_background({"a": root, "duplicate": root})

    def test_reject_geographic_grid_and_bad_area(self):
        root, path = self.flight("A", "20250101")
        with rasterio.open(path, "r+") as dst:
            dst.crs = "EPSG:4326"
        with self.assertRaisesRegex(ValueError, "Projected"):
            define_soil(root)
        self.flight("A", "20250101", extra={"f_full_soil": [-.1, .5]})
        with self.assertRaisesRegex(ValueError, "area fractions"):
            define_soil(root)

    def test_temporal_flags(self):
        root, _ = self.flight("A", "20240101")
        self.flight("A", "20260101", olive=.8)
        result = define_lambda(root)
        self.assertIn("A: multi_year_range_may_include_trend", result.flags)
        self.assertIn("shared_seasonal_phase_not_tested", result.flags)


    def test_jiang_contributions_not_class_weighted_or_soil_subtracted(self):
        root, _ = self.flight("A", "20250101", extra={
            "D_soil": [0, .2], "D_olive": [.2, .6],
            "f_full_soil": [0, .8], "f_full_olive": [1, .2]})
        self.flight("A", "20250701", extra={
            "D_soil": [.1, .3], "D_olive": [.4, .8]})
        cal = define_background(root, quantity="jiang_contribution")
        self.assertAlmostEqual(cal.soil_background, .15)
        self.assertAlmostEqual(cal.lambda_woody, .5)
        self.assertAlmostEqual(cal.observations[0].class_means["soil"], .1)
        self.assertEqual(cal.observations[0].quantity, "jiang_contribution")
        self.assertEqual(cal.woody.settings["quantity"], "jiang_contribution")
        self.assertEqual(cal.woody.method, "contribution_observed_range")
        self.assertAlmostEqual(define_soil(root, quantity="jiang_contribution").value, .15)
        self.assertAlmostEqual(define_lambda(root, quantity="jiang_contribution").value, .5)
        with self.assertRaisesRegex(ValueError, "already excluded"):
            define_lambda(root, quantity="jiang_contribution", soil_background=.15)

    def test_jiang_common_support_and_missing_bands(self):
        root, path = self.flight("A", "20250101", extra={
            "D_soil": [.1, -999], "D_olive": [.4, .9]})
        obs = summarize_flights(root, quantity="jiang_contribution")[0]
        self.assertAlmostEqual(obs.class_means["olive"], .4)
        with rasterio.open(path, "r+") as dst:
            dst.set_band_description(dst.descriptions.index("D_soil") + 1, "missing")
        with self.assertRaisesRegex(ValueError, "D_soil"):
            define_soil(root, quantity="jiang_contribution")
        with self.assertRaisesRegex(ValueError, "only for NDVI"):
            define_soil(root, quantity="jiang_contribution", vi="EVI")
        with self.assertRaisesRegex(ValueError, "quantity"):
            define_soil(root, quantity="typo")

    def test_jiang_nonpositive_baseline(self):
        root, _ = self.flight("A", "20250101", extra={"D_olive": [0, 0]})
        self.flight("A", "20250701", extra={"D_olive": [.4, .4]})
        with self.assertRaisesRegex(ValueError, "baseline"):
            define_lambda(root, quantity="jiang_contribution")


if __name__ == "__main__":
    unittest.main()
