"""Read a complete, single-record BGC GenBank sequence without CDS filtering."""

from collections import Counter
from hashlib import sha256
from pathlib import Path

from data_preparation.feature_cache import file_sha256


DNA_INPUT_VERSION = "whole_record_dna_v1"
DNA_ALPHABET = frozenset("ACGTRYSWKMBDHVN")


def discover_gbk(source):
    source = Path(source)
    if source.is_file():
        paths = [source]
    elif source.is_dir():
        paths = sorted(path for path in source.rglob("*")
                       if path.is_file() and path.suffix.lower() in {".gb", ".gbk", ".genbank"})
    else:
        raise ValueError(f"GenBank path does not exist: {source}")
    if not paths:
        raise ValueError("No GenBank files found.")
    ids = [path.stem for path in paths]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate GenBank basenames would collide as BGC IDs.")
    return sorted(paths, key=lambda path: path.stem)


def read_dna(path):
    """Use the file stem as BGC ID, matching the existing protein workflow.

    No region inference, reverse complementation, CDS concatenation or ambiguous
    base deletion is performed. A chromosome-sized record remains chromosome-sized.
    """
    from Bio import SeqIO

    path = Path(path)
    record = SeqIO.read(path, "genbank")
    sequence = str(record.seq).upper()
    if not sequence:
        raise ValueError("Empty DNA sequence")
    invalid = set(sequence) - DNA_ALPHABET
    if invalid:
        raise ValueError(f"Unsupported DNA characters: {sorted(invalid)}")
    counts = Counter(sequence)
    acgt_count = sum(counts[base] for base in "ACGT")
    metadata = {
        "bgc_id": path.stem,
        "source": str(path.resolve()),
        "source_sha256": file_sha256(path),
        "record_id": record.id,
        "sequence_sha256": sha256(sequence.encode("ascii")).hexdigest(),
        "base_count": len(sequence),
        "sequence_region": [0, len(sequence)],
        "orientation": "as stored in GenBank",
        "topology": record.annotations.get("topology", "unknown"),
        "organism": record.annotations.get("organism", "unknown"),
        "gc_fraction_acgt": (counts["G"] + counts["C"]) / acgt_count if acgt_count else None,
        "ambiguous_base_count": len(sequence) - acgt_count,
        "ambiguous_base_counts": {key: counts[key] for key in sorted(counts) if key not in "ACGT"},
    }
    return sequence, metadata
