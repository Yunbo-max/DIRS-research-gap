#!/usr/bin/env python3
"""Map anonymous evaluator rewards back to private path identifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluations", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    evaluations = read_json(args.evaluations)
    mapping = read_json(args.mapping)["mapping"]
    rank_by_draft = {
        item["draft_id"]: item["rank"]
        for item in evaluations["strict_ranking"]
    }
    rows = []
    for evaluation in evaluations["evaluations"]:
        draft_id = evaluation["draft_id"]
        if draft_id not in mapping:
            raise ValueError(f"Missing private mapping for {draft_id}")
        rows.append(
            {
                "path_id": mapping[draft_id],
                "anonymous_draft_id": draft_id,
                "rank": rank_by_draft[draft_id],
                "overall_preference_score": evaluation[
                    "overall_preference_score"
                ],
                "confidence": evaluation["confidence"],
                "scores": evaluation["scores"],
                "hard_failures": evaluation["hard_failures"],
            }
        )
    rows.sort(key=lambda item: item["path_id"])
    if len({item["path_id"] for item in rows}) != len(rows):
        raise ValueError("Private mapping is not one-to-one")

    output = {
        "schema_version": "dirs.path_rewards.v1",
        "source_evaluation_mode": evaluations["evaluation_mode"],
        "reward_field": "overall_preference_score",
        "path_rewards": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
