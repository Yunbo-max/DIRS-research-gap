#!/usr/bin/env python3
"""Create a deterministic evaluator packet without path identities."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", required=True, type=Path)
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()

    source = read_json(args.rollouts)
    rollouts = source["rollouts"]
    shuffled = list(rollouts)
    random.Random(args.seed).shuffle(shuffled)

    packet_drafts = []
    mapping: dict[str, str] = {}
    for index, rollout in enumerate(shuffled):
        anonymous_id = f"draft_{chr(ord('A') + index)}"
        mapping[anonymous_id] = rollout["path_id"]
        packet_drafts.append(
            {
                "draft_id": anonymous_id,
                "abstract": rollout["abstract"],
                "reported_word_count": rollout["word_count"],
            }
        )

    write_json(
        args.packet,
        {
            "schema_version": "dirs.anonymous_path_rollouts.v1",
            "seed_commitment": args.seed,
            "draft_count": len(packet_drafts),
            "path_identity_visible": False,
            "drafts": packet_drafts,
        },
    )
    write_json(
        args.mapping,
        {
            "schema_version": "dirs.private_rollout_mapping.v1",
            "seed": args.seed,
            "mapping": mapping,
            "must_not_be_read_by_evaluator": True,
        },
    )


if __name__ == "__main__":
    main()
