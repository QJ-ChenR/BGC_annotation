"""Validate exported-vector alignment and unsupervised visualization outputs."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from visualization import umap_features


class UMAPFeaturesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]],
                                dtype=np.float32)
        self.ids = ["BGC_A", "BGC_B", "BGC_C", "BGC_D"]

    def features(self, vectors=None, rows=None, metadata=None, name="features"):
        directory = self.root / name
        directory.mkdir(exist_ok=True)
        array = self.vectors if vectors is None else np.asarray(vectors)
        np.save(directory / "embeddings.npy", array, allow_pickle=False)
        if rows is None:
            rows = pd.DataFrame({"row": range(len(array)),
                                 "bgc_id": self.ids[:len(array)],
                                 "token_count": [3, 5, 7, 11][:len(array)]})
        pd.DataFrame(rows).to_csv(directory / "bgc_ids.tsv", sep="\t", index=False)
        source = {"format": "bgc_vectors_v1", "status": "complete", "route": "esm",
                  "count": len(array), "dimension": array.shape[-1],
                  "extractor": {"route": "esm"}}
        source.update(metadata or {})
        (directory / "metadata.json").write_text(json.dumps(source), encoding="utf-8")
        return directory

    def labels(self, rows=None):
        if rows is None:
            rows = {"BGC_number": ["BGC_C", "BGC_A", "BGC_B", "UNUSED"],
                    "biosyn_class": [[], ["PKS", "NRPS"], ["ribosomal"], ["other"]]}
        path = self.root / "MAC_metadata.pkl"
        pd.DataFrame(rows).to_pickle(path)
        return path

    def args(self, features=None, labels=None, output="plot", *extra):
        return umap_features.build_parser().parse_args([
            "--features-dir", str(features or self.features()),
            "--labels", str(labels or self.labels()),
            "--output", str(self.root / output),
            "--n-neighbors", "2", *extra,
        ])

    def test_shuffled_manifest_uses_row_numbers_without_reordering_vectors(self):
        rows = pd.DataFrame({"row": [2, 0, 3, 1],
                             "bgc_id": ["BGC_C", "BGC_A", "BGC_D", "BGC_B"],
                             "token_count": [7, 3, 11, 5]})
        actual, aligned, metadata = umap_features.load_features(self.features(rows=rows))
        np.testing.assert_array_equal(actual, self.vectors)
        self.assertEqual(aligned["bgc_id"].tolist(), self.ids)
        self.assertEqual(aligned["token_count"].tolist(), [3, 5, 7, 11])
        self.assertEqual(metadata["route"], "esm")

    def test_output_dimension_is_read_from_matrix(self):
        for dimension in (5, 256, 768, 1024, 1280):
            with self.subTest(dimension=dimension):
                array = np.ones((4, dimension), dtype=np.float32)
                actual, _, _ = umap_features.load_features(
                    self.features(array, name=f"dim_{dimension}"))
                self.assertEqual(actual.shape, (4, dimension))

    def test_invalid_manifest_does_not_silently_drop_rows(self):
        valid = {"row": [0, 1, 2, 3], "bgc_id": self.ids,
                 "token_count": [3, 5, 7, 11]}
        cases = {
            "duplicate_id": {"bgc_id": ["BGC_A", "BGC_A", "BGC_C", "BGC_D"]},
            "blank_id": {"bgc_id": ["BGC_A", "", "BGC_C", "BGC_D"]},
            "duplicate_row": {"row": [0, 1, 1, 3]},
            "out_of_range_row": {"row": [0, 1, 2, 4]},
            "negative_row": {"row": [-1, 1, 2, 3]},
            "fractional_row": {"row": [0, 1.5, 2, 3]},
            "zero_token_count": {"token_count": [0, 5, 7, 11]},
            "negative_token_count": {"token_count": [3, -5, 7, 11]},
            "fractional_token_count": {"token_count": [3, 5, 7.5, 11]},
            "missing_token_count": {"token_count": [3, 5, np.nan, 11]},
        }
        for name, changes in cases.items():
            with self.subTest(name=name):
                directory = self.features(rows={**valid, **changes}, name=name)
                with self.assertRaises(ValueError):
                    umap_features.load_features(directory)
        with self.assertRaises(ValueError):
            umap_features.load_features(self.features(rows=pd.DataFrame(valid).iloc[:3],
                                                      name="missing_row"))

    def test_invalid_vectors_are_rejected(self):
        cases = {"nan": np.nan, "infinity": np.inf, "negative_infinity": -np.inf}
        for name, value in cases.items():
            with self.subTest(name=name):
                array = self.vectors.copy()
                array[1, 1] = value
                with self.assertRaises(ValueError):
                    umap_features.load_features(self.features(array, name=name))
        array = self.vectors.copy()
        array[1] = 0
        with self.assertRaises(ValueError):
            umap_features.load_features(self.features(array, name="zero_vector"))
        with self.assertRaises(ValueError):
            umap_features.load_features(self.features(np.ones((4, 2, 3)), name="rank3"))

    def test_metadata_count_dimension_and_failed_status_are_checked(self):
        for name, metadata in (
            ("count", {"count": 3}),
            ("dimension", {"dimension": 5}),
            ("failed", {"status": "failed"}),
            ("format", {"format": "unsupported_format"}),
        ):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    umap_features.load_features(self.features(metadata=metadata, name=name))

    def test_complete_with_exclusions_is_a_valid_source(self):
        _, _, metadata = umap_features.load_features(self.features(metadata={
            "status": "complete_with_exclusions", "excluded_count": 2,
        }))
        self.assertEqual(metadata["excluded_count"], 2)

    def test_multilabel_order_aliases_and_repeats_are_canonicalized(self):
        self.assertEqual(umap_features.normalize_labels(["PKS", "NRPS", "PKS"]),
                         ("NRPS", "PKS"))
        self.assertEqual(umap_features.normalize_labels(["ribosomal", "other", "terpene"]),
                         ("Other", "RiPP", "Terpene"))
        self.assertEqual(umap_features.normalize_labels(["saccharide"]), ("Saccharide",))

    def test_missing_labels_are_distinct_from_other(self):
        for value in ([], None, np.nan):
            with self.subTest(value=value):
                self.assertEqual(umap_features.normalize_labels(value), ())
        self.assertEqual(umap_features.normalize_labels(["other"]), ("Other",))

    def test_numpy_boolean_multihot_and_invalid_array_rank(self):
        self.assertEqual(umap_features.normalize_labels(
            np.array([True, False, True, False, False, False])), ("NRPS", "PKS"))
        for value in (np.array(1), np.ones((2, 3))):
            with self.subTest(shape=value.shape), self.assertRaises(ValueError):
                umap_features.normalize_labels(value)

    def test_multihot_uses_the_authors_exact_six_class_order(self):
        self.assertEqual(umap_features.normalize_labels([1, 0, 1, 0, 0, 0]),
                         ("NRPS", "PKS"))
        self.assertEqual(umap_features.normalize_labels(np.array([0, 1, 0, 1, 1, 1])),
                         ("Other", "RiPP", "Saccharide", "Terpene"))
        self.assertEqual(umap_features.normalize_labels([0, 0, 0, 0, 0, 0]), ())
        for value in ([1, 0], [1, 0, 2, 0, 0, 0], [1, 0, np.nan, 0, 0, 0]):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    umap_features.normalize_labels(value)

    def test_labels_join_by_id_and_keep_all_classes(self):
        mapping = umap_features.load_labels(self.labels())
        self.assertEqual(mapping["BGC_A"], ("NRPS", "PKS"))
        self.assertEqual(mapping["BGC_B"], ("RiPP",))
        self.assertEqual(mapping["BGC_C"], ())
        self.assertNotIn("BGC_D", mapping)

    def test_duplicate_label_ids_are_rejected_even_when_labels_agree(self):
        path = self.labels({"BGC_number": ["BGC_A", "BGC_A"],
                            "biosyn_class": [["PKS"], ["PKS"]]})
        with self.assertRaises(ValueError):
            umap_features.load_labels(path)

    def test_invalid_label_schema_and_unknown_class_are_rejected(self):
        with self.assertRaises(ValueError):
            umap_features.load_labels(self.labels({"BGC_number": ["BGC_A"]}))
        with self.assertRaises(ValueError):
            umap_features.normalize_labels(["not_a_biosynthetic_class"])

    def test_parser_defaults(self):
        args = umap_features.build_parser().parse_args([
            "--features-dir", "features", "--labels", "MAC_metadata.pkl", "--output", "plots",
        ])
        self.assertEqual(args.metric, "cosine")
        self.assertEqual(args.n_neighbors, 15)
        self.assertEqual(args.min_dist, 0.1)
        self.assertEqual(args.random_state, 42)

    def test_invalid_plot_or_umap_parameters_fail_before_fitting(self):
        features = self.features()
        labels = self.labels()
        for flag, value in (("--n-neighbors", "1"), ("--min-dist", "-0.1"),
                            ("--point-size", "0"), ("--alpha", "1.5"), ("--dpi", "0")):
            with self.subTest(flag=flag):
                args = self.args(features, labels, flag.removeprefix("--"), flag, value)
                with patch.object(umap_features, "fit_umap") as fit:
                    with self.assertRaises(ValueError):
                        umap_features.run(args)
                fit.assert_not_called()

    def test_run_fits_only_vectors_and_preserves_unknown_and_multilabel_rows(self):
        args = self.args()
        coords = np.array([[2, 7], [3, 9], [5, 11], [13, 17]], dtype=np.float32)
        with patch.object(umap_features, "fit_umap", return_value=coords) as fit, \
                patch.object(umap_features, "plot_outputs") as plots:
            metadata = umap_features.run(args)
        fit.assert_called_once()
        np.testing.assert_array_equal(fit.call_args.args[0], self.vectors)
        # The only UMAP data argument is the feature matrix: no target labels or counts.
        self.assertEqual(len(fit.call_args.args), 1)
        self.assertEqual(set(fit.call_args.kwargs),
                         {"metric", "n_neighbors", "min_dist", "random_state"})
        self.assertEqual(fit.call_args.kwargs["metric"], "cosine")
        plots.assert_called_once()
        table = pd.read_csv(args.output / "umap_coordinates.tsv", sep="\t")
        self.assertEqual(table["bgc_id"].tolist(), self.ids)
        self.assertEqual(table["labels"].tolist(), ["NRPS+PKS", "RiPP", "Unknown", "Unknown"])
        self.assertEqual(table["label_count"].tolist(), [2, 1, 0, 0])
        self.assertEqual(table["token_count"].tolist(), [3, 5, 7, 11])
        np.testing.assert_array_equal(table[["umap_1", "umap_2"]].to_numpy(), coords)
        plotted_table = plots.call_args.args[0]
        np.testing.assert_array_equal(plotted_table[["umap_1", "umap_2"]].to_numpy(), coords)
        on_disk = json.loads((args.output / "umap_metadata.json").read_text())
        self.assertEqual(on_disk, metadata)
        self.assertEqual(metadata["unknown_count"], 2)
        self.assertEqual(metadata["unknown_ids"], ["BGC_C", "BGC_D"])
        self.assertEqual(metadata["feature_dimension"], 3)
        self.assertEqual(metadata["umap_parameters"]["metric"], "cosine")
        self.assertEqual(metadata["umap_parameters"]["n_neighbors"], 2)
        legacy_label = "Token count (input domains / whole-protein fallbacks)"
        self.assertEqual(metadata["token_count_unit"], "domain_protein_tokens")
        self.assertEqual(metadata["plot_parameters"]["token_count_label"], legacy_label)
        self.assertEqual(plots.call_args.kwargs["token_count_label"], legacy_label)

    def test_dna_export_uses_bpe_units_without_changing_reducer_inputs(self):
        from matplotlib.figure import Figure

        vectors = np.ones((4, 768), dtype=np.float32)
        features = self.features(vectors, metadata={
            "route": "dna", "extractor": {"route": "dna"},
            "token_count_unit": "dna_bpe_tokens",
        })
        args = self.args(features=features)
        coords = np.array([[0, 0], [1, 3], [2, 1], [4, 5]], dtype=np.float32)
        colorbar_labels = []

        def capture_label(figure, path, *args, **kwargs):
            if Path(path).stem == "umap_token_count":
                colorbar_labels.append(figure.axes[-1].get_ylabel())

        with patch.object(umap_features, "fit_umap", return_value=coords) as fit, \
                patch.object(Figure, "savefig", new=capture_label):
            metadata = umap_features.run(args)
        np.testing.assert_array_equal(fit.call_args.args[0], vectors)
        self.assertEqual(len(fit.call_args.args), 1)
        self.assertEqual(set(fit.call_args.kwargs),
                         {"metric", "n_neighbors", "min_dist", "random_state"})
        expected_label = "DNA BPE token count (excluding special tokens)"
        self.assertEqual(colorbar_labels, [expected_label, expected_label])
        self.assertEqual(metadata["feature_dimension"], 768)
        self.assertEqual(metadata["token_count_unit"], "dna_bpe_tokens")
        self.assertEqual(metadata["feature_source"]["route"], "dna")
        self.assertEqual(metadata["plot_parameters"]["token_count_label"], expected_label)
        self.assertFalse(metadata["labels_used_for_fit"])
        self.assertFalse(metadata["token_count_used_for_fit"])
        on_disk = json.loads((args.output / "umap_metadata.json").read_text())
        self.assertEqual(on_disk, metadata)

    def test_invalid_reducer_coordinates_are_not_plotted_or_exported(self):
        features = self.features()
        labels = self.labels()
        cases = {"wrong_rows": np.ones((3, 2)), "wrong_dimensions": np.ones((4, 3)),
                 "nonfinite": np.array([[0., 1.], [np.nan, 1.], [2., 3.], [4., 5.]])}
        for name, coords in cases.items():
            with self.subTest(name=name):
                args = self.args(features, labels, name)
                with patch.object(umap_features, "fit_umap", return_value=coords), \
                        patch.object(umap_features, "plot_outputs") as plots:
                    with self.assertRaisesRegex(ValueError, "invalid coordinates"):
                        umap_features.run(args)
                plots.assert_not_called()
                self.assertFalse((args.output / "umap_coordinates.tsv").exists())
                self.assertFalse((args.output / "umap_metadata.json").exists())

    def test_existing_output_is_preserved_and_prevents_refitting(self):
        args = self.args()
        args.output.mkdir()
        sentinel = args.output / "umap_classes.png"
        sentinel.write_bytes(b"previous plot")
        with patch.object(umap_features, "fit_umap") as fit:
            with self.assertRaisesRegex(ValueError, "already exist"):
                umap_features.run(args)
        fit.assert_not_called()
        self.assertEqual(sentinel.read_bytes(), b"previous plot")
        self.assertFalse((args.output / "umap_coordinates.tsv").exists())

    def test_plot_files_are_nonempty_png_and_pdf(self):
        from matplotlib.figure import Figure

        args = self.args()
        coords = np.array([[0, 0], [1, 3], [2, 1], [4, 5]], dtype=np.float32)
        captured = {}
        original_save = Figure.savefig

        def capture_and_save(figure, path, *args, **kwargs):
            axes = figure.axes[0]
            offsets = np.concatenate([collection.get_offsets()
                                      for collection in axes.collections])
            captured[Path(path).name] = (offsets, axes.get_xlim(), axes.get_ylim())
            return original_save(figure, path, *args, **kwargs)

        with patch.object(umap_features, "fit_umap", return_value=coords), \
                patch.object(Figure, "savefig", new=capture_and_save):
            umap_features.run(args)
        for stem in ("umap_classes", "umap_token_count"):
            with self.subTest(stem=stem):
                png = (args.output / (stem + ".png")).read_bytes()
                pdf = (args.output / (stem + ".pdf")).read_bytes()
                self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
                self.assertTrue(pdf.startswith(b"%PDF-"))
                self.assertGreater(len(png), 1000)
                self.assertGreater(len(pdf), 1000)
        expected = sorted(map(tuple, coords))
        for offsets, _, _ in captured.values():
            self.assertEqual(sorted(map(tuple, offsets)), expected)
        self.assertEqual(captured["umap_classes.png"][1:],
                         captured["umap_token_count.png"][1:])


if __name__ == "__main__":
    unittest.main()
