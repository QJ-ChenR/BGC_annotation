import tempfile
import unittest
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import (
    AfterPosition, BeforePosition, SeqFeature, SimpleLocation, UnknownPosition,
    WithinPosition,
)
from Bio.SeqRecord import SeqRecord

from data_preparation.feature_inputs import _location, parse_bgc


def cds(start, sequence, name, strand=1):
    return SeqFeature(
        SimpleLocation(start, start + 3 * len(sequence), strand=strand),
        type="CDS",
        qualifiers={"gene": [name], "translation": [sequence]},
    )


def domain(start, sequence, parent, protein_start=None, strand=1):
    qualifiers = {
        "locus_tag": [parent], "translation": [sequence],
        "aSDomain": ["test_domain"], "label": [f"{parent}_{start}"],
    }
    if protein_start is not None:
        qualifiers.update({
            "protein_start": [str(protein_start)],
            "protein_end": [str(protein_start + len(sequence))],
        })
    return SeqFeature(
        SimpleLocation(start, start + 3 * len(sequence), strand=strand),
        type="aSDomain", qualifiers=qualifiers,
    )


class FeatureInputTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tempdir.name) / "test_bgc.gbk"

    def tearDown(self):
        self.tempdir.cleanup()

    def write(self, features, copies=1):
        record = SeqRecord(Seq("A" * 1000), id="test", name="test")
        record.annotations["molecule_type"] = "DNA"
        record.features = features
        SeqIO.write([record] * copies, self.path, "genbank")

    def test_domain_before_cds_and_gene_alias_resolve_true_parent(self):
        self.write([
            cds(0, "ACDE", "gene_a"),
            domain(100, "FG", "gene_b", protein_start=0),
            cds(100, "FGHIK", "gene_b"),
        ])
        parsed = parse_bgc(self.path)
        self.assertEqual([t["sequence"] for t in parsed["tokens"]], ["ACDE", "FG"])
        self.assertEqual(parsed["tokens"][1]["parent_identifiers"], {"gene": "gene_b"})
        self.assertEqual(parsed["parents"][1]["represented_residues"], 2)
        self.assertEqual(parsed["parents"][1]["unrepresented_residues"], 3)
        self.assertEqual(parsed["parents"][1]["residue_coverage"], 0.4)

    def test_negative_strand_orders_domains_by_protein_position(self):
        self.write([
            domain(100, "FG", "reverse_gene", protein_start=4, strand=-1),
            cds(100, "ACDEFG", "reverse_gene", strand=-1),
            domain(112, "AC", "reverse_gene", protein_start=0, strand=-1),
        ])
        parsed = parse_bgc(self.path)
        self.assertEqual([t["sequence"] for t in parsed["tokens"]], ["AC", "FG"])
        self.assertEqual([t["genomic_start"] for t in parsed["tokens"]], [112, 100])
        self.assertTrue(all(t["strand"] == -1 for t in parsed["tokens"]))
        self.assertEqual(len(parsed["tokens"]), 2)

    def test_genes_order_by_genomic_start_and_no_domain_protein_is_complete(self):
        self.write([cds(90, "FGHI", "late"), cds(0, "ACDE", "early")])
        parsed = parse_bgc(self.path)
        self.assertEqual([t["sequence"] for t in parsed["tokens"]], ["ACDE", "FGHI"])
        self.assertEqual([t["source"] for t in parsed["tokens"]], ["protein", "protein"])
        self.assertTrue(all(p["residue_coverage"] == 1 for p in parsed["parents"]))
        self.assertEqual(parsed["bgc_id"], "test_bgc")
        self.assertEqual(len(parsed["source_sha256"]), 64)

    def test_missing_coordinates_derived_from_unique_translation(self):
        self.write([cds(0, "ACDEFG", "gene"), domain(6, "DE", "gene")])
        token = parse_bgc(self.path)["tokens"][0]
        self.assertEqual((token["protein_start"], token["protein_end"]), (2, 4))
        self.assertEqual(token["protein_coordinate_source"], "unique_translation_match")

    def test_ambiguous_missing_coordinates_fail(self):
        self.write([cds(0, "ACAC", "gene"), domain(0, "AC", "gene")])
        with self.assertRaisesRegex(ValueError, "Ambiguous domain position"):
            parse_bgc(self.path)

    def test_inconsistent_protein_coordinates_fail(self):
        self.write([cds(0, "ACDE", "gene"), domain(6, "DE", "gene", protein_start=0)])
        with self.assertRaisesRegex(ValueError, "disagrees with parent"):
            parse_bgc(self.path)

    def test_parent_identifier_cannot_override_genomic_containment(self):
        self.write([
            cds(0, "ACDE", "a"), cds(100, "ACDE", "b"),
            domain(100, "AC", "a", protein_start=0),
        ])
        with self.assertRaisesRegex(ValueError, "found 0"):
            parse_bgc(self.path)

    def test_ambiguous_containment_without_identifiers_fails(self):
        domain_feature = domain(3, "CD", "unknown", protein_start=1)
        domain_feature.qualifiers.pop("locus_tag")
        self.write([cds(0, "ACDE", "a"), cds(0, "ACDE", "b"), domain_feature])
        with self.assertRaisesRegex(ValueError, "found 2"):
            parse_bgc(self.path)

    def test_missing_translation_fails_with_path(self):
        feature = cds(0, "ACDE", "gene")
        feature.qualifiers.pop("translation")
        self.write([feature])
        with self.assertRaisesRegex(ValueError, "test_bgc.gbk.*Missing 'translation'"):
            parse_bgc(self.path)

    def test_partial_cds_preserves_translation_and_uncertain_bounds(self):
        feature = cds(0, "ACDE", "partial")
        feature.location = SimpleLocation(BeforePosition(0), AfterPosition(12), strand=1)
        self.write([feature])
        record = parse_bgc(self.path)
        token = record["tokens"][0]
        self.assertEqual(token["sequence"], "ACDE")
        self.assertEqual((token["genomic_start"], token["genomic_end"]), (0, 12))
        self.assertFalse(token["genomic_coordinates_exact"])
        self.assertEqual(token["genomic_part_bound_types"], [
            {"start": "BeforePosition", "end": "AfterPosition"}
        ])
        self.assertIn("<0", token["genomic_location"])
        self.assertIn(">12", token["genomic_location"])
        self.assertEqual(record["parents"][0]["residue_coverage"], 1)

    def test_unknown_and_within_positions_remain_unsupported(self):
        for start in (UnknownPosition(), WithinPosition(0, left=0, right=3)):
            feature = cds(0, "ACDE", "partial")
            feature.location = SimpleLocation(start, 12, strand=1)
            with self.subTest(start=start), self.assertRaisesRegex(
                ValueError, "Unknown or ambiguous fuzzy location"
            ):
                _location(feature, 1000)

    def test_j_residue_rejected_for_esm_alphabet(self):
        self.write([cds(0, "ACJ", "gene")])
        with self.assertRaisesRegex(ValueError, "Unsupported translation characters.*J"):
            parse_bgc(self.path)

    def test_overlapping_domains_coverage_uses_union(self):
        self.write([
            cds(0, "ACDEFG", "gene"),
            domain(0, "ACDE", "gene", protein_start=0),
            domain(6, "DEFG", "gene", protein_start=2),
        ])
        parsed = parse_bgc(self.path)
        self.assertEqual(parsed["parents"][0]["represented_residues"], 6)
        self.assertEqual(parsed["parents"][0]["token_count"], 2)

    def test_multiple_records_and_empty_bgc_fail(self):
        self.write([cds(0, "ACDE", "gene")], copies=2)
        with self.assertRaisesRegex(ValueError, "Expected one GenBank record"):
            parse_bgc(self.path)
        self.write([])
        with self.assertRaisesRegex(ValueError, "No CDS"):
            parse_bgc(self.path)


if __name__ == "__main__":
    unittest.main()
