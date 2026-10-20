"""Deterministic GBK inputs for frozen feature extraction.

One file is one BGC and must contain one record. CDSs are ordered by genomic
start; domains within a CDS are ordered N to C, independently of strand.
Annotated proteins contribute only their aSDomains (not uncovered gaps).
Proteins without aSDomains contribute their complete annotated translation.
All coordinates in the returned metadata are zero-based, end-exclusive annotated
bounds. Common partial-feature bounds (< or >) retain their uncertainty; they
are not asserted to be exact biological endpoints. Translations are authoritative.
"""

from hashlib import sha256
from pathlib import Path


PARSER_VERSION = "gbk_ordered_domains_v1"
_IDENTIFIERS = ("locus_tag", "protein_id", "gene")
_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWYBXZUO")


def _one(feature, name, required=False):
    values = feature.qualifiers.get(name, [])
    if len(values) > 1:
        raise ValueError(f"Multiple {name!r} values on {feature.type} feature")
    if not values:
        if required:
            raise ValueError(f"Missing {name!r} on {feature.type} feature")
        return None
    return str(values[0])


def _translation(feature):
    sequence = "".join(_one(feature, "translation", required=True).split()).upper()
    if not sequence:
        raise ValueError(f"Empty translation on {feature.type} feature")
    invalid = set(sequence) - _AMINO_ACIDS
    if invalid:
        raise ValueError(
            f"Unsupported translation characters {sorted(invalid)} on {feature.type}"
        )
    return sequence


def _location(feature, record_length):
    from Bio.SeqFeature import AfterPosition, BeforePosition, ExactPosition

    if feature.location is None:
        raise ValueError(f"Missing location on {feature.type} feature")
    parts = []
    bounds = []
    for part in feature.location.parts:
        if part.ref is not None or part.ref_db is not None:
            raise ValueError(f"Remote location is unsupported on {feature.type}")
        if not isinstance(part.start, (ExactPosition, BeforePosition, AfterPosition)) or not isinstance(
            part.end, (ExactPosition, BeforePosition, AfterPosition)
        ):
            raise ValueError(f"Unknown or ambiguous fuzzy location is unsupported on {feature.type}")
        start, end = int(part.start), int(part.end)
        if not 0 <= start < end <= record_length:
            raise ValueError(f"Invalid genomic coordinates on {feature.type}: {start}:{end}")
        parts.append([start, end])
        bounds.append({
            "start": type(part.start).__name__,
            "end": type(part.end).__name__,
        })
    if feature.location.strand not in (-1, 1):
        raise ValueError(f"Unknown or mixed strand on {feature.type} feature")
    return {
        "genomic_start": int(feature.location.start),
        "genomic_end": int(feature.location.end),
        "genomic_parts": parts,
        "genomic_location": str(feature.location),
        "genomic_part_bound_types": bounds,
        "genomic_coordinates_exact": all(
            bound["start"] == "ExactPosition" and bound["end"] == "ExactPosition"
            for bound in bounds
        ),
        "strand": int(feature.location.strand),
    }


def _contained(domain, parent):
    return domain["strand"] == parent["strand"] and all(
        any(ps <= ds and de <= pe for ps, pe in parent["genomic_parts"])
        for ds, de in domain["genomic_parts"]
    )


def _protein_interval(feature, sequence, parent_sequence):
    start_text = _one(feature, "protein_start")
    end_text = _one(feature, "protein_end")
    if (start_text is None) != (end_text is None):
        raise ValueError("A domain must supply both protein_start and protein_end")
    if start_text is not None:
        try:
            start, end = int(start_text), int(end_text)
        except ValueError as exc:
            raise ValueError("Domain protein coordinates must be integers") from exc
        if not 0 <= start < end <= len(parent_sequence):
            raise ValueError(f"Domain protein coordinates out of bounds: {start}:{end}")
        if parent_sequence[start:end] != sequence:
            raise ValueError(
                f"Domain translation disagrees with parent at protein coordinates {start}:{end}"
            )
        return start, end, "annotation"
    start = parent_sequence.find(sequence)
    if start < 0:
        raise ValueError("Domain translation is absent from its parent protein")
    if parent_sequence.find(sequence, start + 1) >= 0:
        raise ValueError(
            "Ambiguous domain position: repeated translation and no protein coordinates"
        )
    return start, start + len(sequence), "unique_translation_match"


def _covered_length(intervals):
    total, previous_end = 0, 0
    for start, end in sorted(intervals):
        total += max(0, end - max(start, previous_end))
        previous_end = max(previous_end, end)
    return total


def parse_bgc(path):
    """Return one validated BGC record with sequence tokens and provenance.

    Invalid/ambiguous records raise ValueError with their file path. No missing
    translation, unsupported location, or unassigned domain is silently omitted.
    """
    from Bio import SeqIO

    path = Path(path)
    try:
        with path.open() as stream:
            records = list(SeqIO.parse(stream, "genbank"))
        if len(records) != 1:
            raise ValueError(f"Expected one GenBank record, found {len(records)}")
        record = records[0]
        parents = []
        for feature_index, feature in enumerate(record.features):
            if feature.type != "CDS":
                continue
            identifiers = {
                key: value
                for key in _IDENTIFIERS
                if (value := _one(feature, key)) is not None
            }
            sequence = _translation(feature)
            parents.append({
                **_location(feature, len(record.seq)),
                "identifiers": identifiers,
                "sequence": sequence,
                "protein_length": len(sequence),
                "sequence_sha256": sha256(sequence.encode("ascii")).hexdigest(),
                "feature_index": feature_index,
                "gene_kind": _one(feature, "gene_kind"),
                "domains": [],
            })
        if not parents:
            raise ValueError("No CDS features found")
        parents.sort(key=lambda p: (
            p["genomic_start"], p["genomic_end"], p["strand"], p["feature_index"]
        ))
        for index, parent in enumerate(parents):
            parent["parent_id"] = f"{path.stem}:cds:{index:05d}"
            parent["gene_order"] = index

        for feature_index, feature in enumerate(record.features):
            if feature.type != "aSDomain":
                continue
            sequence = _translation(feature)
            location = _location(feature, len(record.seq))
            aliases = {
                value for key in _IDENTIFIERS
                if (value := _one(feature, key)) is not None
            }
            alias_matches = [
                p for p in parents if aliases.intersection(p["identifiers"].values())
            ]
            candidates = [
                p for p in (alias_matches or parents) if _contained(location, p)
            ]
            if len(candidates) != 1:
                raise ValueError(
                    f"aSDomain feature {feature_index}: expected one parent matching "
                    f"identifiers/containment, found {len(candidates)}"
                )
            parent = candidates[0]
            start, end, coordinate_source = _protein_interval(
                feature, sequence, parent["sequence"]
            )
            parent["domains"].append({
                **location,
                "source": "domain",
                "sequence": sequence,
                "sequence_sha256": sha256(sequence.encode("ascii")).hexdigest(),
                "domain_name": _one(feature, "aSDomain"),
                "domain_id": (
                    _one(feature, "domain_id") or _one(feature, "asDomain_id")
                    or _one(feature, "label")
                ),
                "protein_start": start,
                "protein_end": end,
                "protein_coordinate_source": coordinate_source,
                "parent_resolution": "identifiers_and_containment" if alias_matches else "containment",
                "feature_index": feature_index,
            })

        tokens, parent_metadata = [], []
        for parent in parents:
            domains = sorted(parent["domains"], key=lambda d: (
                d["protein_start"], d["protein_end"], d["domain_id"] or "",
                d["feature_index"],
            ))
            units = domains or [{
                key: parent[key] for key in (
                    "sequence", "sequence_sha256", "genomic_start", "genomic_end",
                    "genomic_parts", "genomic_location", "genomic_part_bound_types",
                    "genomic_coordinates_exact", "strand", "feature_index"
                )
            }]
            if not domains:
                units[0].update(
                    source="protein", protein_start=0,
                    protein_end=parent["protein_length"],
                    protein_coordinate_source="full_translation",
                )
            for unit_index, unit in enumerate(units):
                tokens.append({
                    **unit,
                    "token_id": f"{parent['parent_id']}:{unit['source']}:{unit_index:04d}",
                    "parent_id": parent["parent_id"],
                    "parent_identifiers": parent["identifiers"],
                    "gene_order": parent["gene_order"],
                    "parent_protein_length": parent["protein_length"],
                    "length": len(unit["sequence"]),
                })
            covered = _covered_length([
                (unit["protein_start"], unit["protein_end"]) for unit in units
            ])
            parent_metadata.append({
                key: value for key, value in parent.items()
                if key not in ("sequence", "domains")
            } | {
                "token_count": len(units),
                "domain_count": len(domains),
                "represented_residues": covered,
                "unrepresented_residues": parent["protein_length"] - covered,
                "residue_coverage": covered / parent["protein_length"],
            })
        return {
            "bgc_id": path.stem,
            "record_id": record.id,
            "source_path": str(path.resolve()),
            "source_sha256": sha256(path.read_bytes()).hexdigest(),
            "parser_version": PARSER_VERSION,
            "coordinate_convention": "zero_based_end_exclusive_annotated_bounds",
            "partial_feature_policy": "BeforePosition_and_AfterPosition_preserved_translation_authoritative",
            "token_order": "CDS_annotated_genomic_start_then_domain_protein_start",
            "unannotated_gaps": "excluded_for_CDS_with_domains",
            "parents": parent_metadata,
            "tokens": tokens,
        }
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc
