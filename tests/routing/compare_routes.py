# SPDX-License-Identifier: GPL-3.0-or-later
"""Compare complete routed files and optional deterministic summaries, excluding timing."""
import argparse
import hashlib
import json
from pathlib import Path


def artifact_hashes(board):
    return {
        extension: hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        for extension in (".kicad_pcb", ".kicad_pro", ".kicad_dru")
        for path in (board.with_suffix(extension),)
    }


def compare(reference, candidate):
    before, after = artifact_hashes(reference), artifact_hashes(candidate)
    if before != after or before[".kicad_pcb"] is None:
        raise ValueError(f"Routed artifact mismatch: {reference} / {candidate}")
    return before


SUMMARY_FIELDS = (
    "routed", "connections", "tracks", "vias", "expansions", "pitch_mm", "failures",
    "variant", "variant_name", "copper_layers", "work", "work_all_variants",
    "time_limited", "open", "completion_pct", "style",
)


def summary_signature(summary):
    return {key: summary[key] for key in SUMMARY_FIELDS if key in summary}


def compare_summaries(reference, candidate):
    before = summary_signature(json.loads(reference.read_text()))
    after = summary_signature(json.loads(candidate.read_text()))
    if not before or before != after:
        raise ValueError(f"Deterministic summary mismatch: {reference} / {candidate}")
    if before.get("time_limited", False):
        raise ValueError("A time-limited route is not an exact-work comparison")
    return before


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--reference-summary", type=Path)
    parser.add_argument("--candidate-summary", type=Path)
    args = parser.parse_args()
    if bool(args.reference_summary) != bool(args.candidate_summary):
        parser.error("Provide both summary paths")
    result = {"artifacts": compare(args.reference, args.candidate)}
    if args.reference_summary:
        result["summary"] = compare_summaries(args.reference_summary, args.candidate_summary)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
