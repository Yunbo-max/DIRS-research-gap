#!/usr/bin/env python3
"""Build a DIRS Case 1 abstract-only training run from a domain chip list.

The script creates the substrate for a blind abstract-writing test:
- parse chip paths from a domain topic file
- hold out one chip
- extract original abstracts for training chips only
- infer simple abstract DAG node/edge support priors
- save style priors and a held-out test card

It intentionally does not generate the held-out abstract.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import statistics
from pathlib import Path
from typing import Any


ABSTRACT_START = re.compile(r"^\s*(abstract|a\s*b\s*s\s*t\s*r\s*a\s*c\s*t|summary)\b", re.I)
ABSTRACT_STOP = re.compile(
    r"^\s*(1\s+introduction|1\.\s*introduction|introduction|keywords|"
    r"1\s+i\s*n\s*t\s*r\s*o\s*d\s*u\s*c\s*t\s*i\s*o\s*n|"
    r"i\s*n\s*t\s*r\s*o\s*d\s*u\s*c\s*t\s*i\s*o\s*n|"
    r"ccs concepts|index terms|contents|background|related work)\s*$",
    re.I,
)
INLINE_STOP = re.compile(
    r"\b(?:Figure|Table)\s+\d+\.\s|\b(?:\d+\.?\s*)?Introduction\b|"
    r"\bProject\s+Page\s*:|\bProject\s+page\s*:",
    re.I,
)
ARXIV_HEADER = re.compile(r"\barXiv:\d{4}\.\d{4,5}v\d+\s+\[[^\]]+\]\s+\d+\s+\w+\s+\d{4}\b", re.I)
TITLE_STOPWORDS = {
    "about",
    "after",
    "analysis",
    "and",
    "architecture",
    "attention",
    "based",
    "before",
    "beyond",
    "context",
    "data",
    "deep",
    "for",
    "from",
    "graph",
    "how",
    "image",
    "improved",
    "large",
    "learn",
    "learning",
    "model",
    "modeling",
    "models",
    "modern",
    "network",
    "networks",
    "paper",
    "prediction",
    "principles",
    "scalable",
    "sequence",
    "state",
    "systems",
    "the",
    "through",
    "time",
    "toward",
    "transformer",
    "transformers",
    "using",
    "with",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_chip_paths(domain_file: Path) -> list[Path]:
    text = domain_file.read_text(encoding="utf-8")
    return [Path(p) for p in re.findall(r"`(/tf/notebooks/[^`]+\.chip\.json)`", text)]


def parse_domain_name(domain_file: Path) -> str:
    text = domain_file.read_text(encoding="utf-8")
    for line in text.splitlines():
        m = re.match(r"^#\s+(.+?)\s*$", line)
        if m:
            return m.group(1).strip()
    return domain_file.stem.replace("_", " ").title()


def chip_title(chip: dict[str, Any]) -> str:
    return chip.get("chip_metadata", {}).get("title") or chip.get("title") or chip.get("chip_id", "unknown")


def chip_venue(chip: dict[str, Any]) -> str:
    return str(chip.get("chip_metadata", {}).get("venue") or "").upper()


def flatten_paths(value: Any) -> list[Path]:
    out: list[Path] = []
    if isinstance(value, str) and value.startswith("/tf/notebooks"):
        out.append(Path(value))
    elif isinstance(value, list):
        for item in value:
            out.extend(flatten_paths(item))
    elif isinstance(value, dict):
        for item in value.values():
            out.extend(flatten_paths(item))
    return out


def safe_identifier_strings(chip_path: Path, chip: dict[str, Any]) -> list[str]:
    chip_id = str(chip.get("chip_id", chip_path.stem.replace(".chip", "")))
    meta = chip.get("chip_metadata", {})
    external_id = str(meta.get("external_id") or "")
    parts = chip_id.split("_")

    ids = [chip_id]
    if len(parts) >= 2:
        ids.append("_".join(parts[:2]))
    if external_id:
        ids.extend([external_id, f"{parts[0]}_{external_id}" if parts else external_id])

    seen: set[str] = set()
    out: list[str] = []
    for item in ids:
        item = item.strip()
        if len(item) < 4 or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def path_matches_chip(path: Path, identifiers: list[str]) -> bool:
    name = path.name
    return any(identifier and identifier in name for identifier in identifiers)


def candidate_source_paths(chip_path: Path, chip: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    identifiers = safe_identifier_strings(chip_path, chip)
    paths.extend(
        p
        for p in flatten_paths(chip.get("source_coverage", {}))
        if path_matches_chip(p, identifiers)
    )

    roots = [
        chip_path.parent.parent / "text",
        chip_path.parent.parent / "pdf_text",
    ]

    for root in roots:
        if not root.exists():
            continue
        for p in root.iterdir():
            if not p.is_file():
                continue
            name = p.name
            if any(identifier and identifier in name for identifier in identifiers):
                paths.append(p)

    keep_ext = {".txt", ".html", ".xml"}
    seen: set[str] = set()
    cleaned: list[Path] = []
    for p in paths:
        if p.suffix.lower() not in keep_ext:
            continue
        key = str(p)
        if key not in seen and p.exists():
            seen.add(key)
            cleaned.append(p)

    # Prefer text extracted from PDFs/OpenReview/CVF over metadata XML.
    def rank(p: Path) -> tuple[int, int]:
        n = p.name.lower()
        if "openreview.txt" in n or "cvf.txt" in n or "arxiv" in n and p.suffix == ".txt":
            return (0, len(n))
        if p.suffix == ".txt":
            return (1, len(n))
        if p.suffix == ".html":
            return (2, len(n))
        return (3, len(n))

    return sorted(cleaned, key=rank)


def distinctive_title_tokens(title: str, chip_id: str) -> set[str]:
    title_tokens = {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", title)
        if token.lower() not in TITLE_STOPWORDS
    }
    chip_tokens = {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", chip_id)
        if token.lower() not in {"cvpr2026", "iclr2026", "icml2026", "chip", "json"}
    }
    return title_tokens | chip_tokens


def abstract_matches_title(abstract: str, chip: dict[str, Any]) -> bool:
    tokens = distinctive_title_tokens(chip_title(chip), str(chip.get("chip_id", "")))
    if not tokens:
        return True
    lower = abstract.lower()
    return any(token in lower for token in tokens)


def html_to_text(raw: str) -> str:
    raw = re.sub(r"(?is)<script.*?</script>", " ", raw)
    raw = re.sub(r"(?is)<style.*?</style>", " ", raw)
    raw = re.sub(r"(?i)<br\s*/?>", "\n", raw)
    raw = re.sub(r"(?i)</(p|div|section|h[1-6]|li|tr)>", "\n", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    return html.unescape(raw)


def normalize_lines(text: str) -> list[str]:
    lines = []
    for line in text.replace("\r", "\n").split("\n"):
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            lines.append(line)
    return lines


def strip_column_tail(line: str) -> str:
    """Keep the left text column when PDF extraction interleaves two columns."""
    parts = [part.strip() for part in re.split(r"\s{8,}", line.rstrip()) if part.strip()]
    if not parts:
        return ""
    for part in parts:
        if re.search(r"[A-Za-z]", part):
            return part
    return parts[0]


def normalize_pdf_lines(text: str) -> list[str]:
    lines = []
    for raw_line in text.replace("\r", "\n").split("\n"):
        line = strip_column_tail(raw_line)
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            lines.append(line)
    return lines


def is_noise_line(line: str) -> bool:
    lower = line.lower().strip()
    if not lower:
        return True
    if len(re.findall(r"[A-Za-z]", line)) < 3:
        return True
    if re.match(r"^(figure|table)\s+\d+\b", line, re.I):
        return True
    if re.match(r"^(image|video|3d|x|y|z|t|skip|compute|current|condition)\b", lower):
        return True
    if "cvpr paper is the open access version" in lower:
        return True
    if "final published version" in lower:
        return True
    if "corresponding author" in lower or "equal contribution" in lower:
        return True
    if "arxiv:" in lower:
        return True
    if "project page" in lower and word_count(line) < 12:
        return True
    if "…" in line and word_count(line) <= 5:
        return True
    return False


def looks_like_abstract_start(line: str) -> bool:
    if is_noise_line(line) or word_count(line) < 5:
        return False
    if re.match(
        r"^(We|This|The|Although|Despite|Recent|Large|Vision|Video|Visual|Sparse|Diffusion|"
        r"Specular|Articulated|Autoregressive|Modern|Existing|Transformer|Tokenization|"
        r"Language|Knowledge|Hyperparameter|Overcoming|Deploying|Efficient)\b",
        line,
    ):
        return True
    return bool(re.search(r"\b(we propose|we introduce|we present|this paper)\b", line, re.I))


def join_extracted_lines(lines: list[str]) -> str:
    text = ""
    for line in lines:
        if text.endswith("-"):
            text = text[:-1] + line.lstrip()
        else:
            text = f"{text} {line}".strip()
    return text


def sanitize_candidate(text: str) -> str:
    text = ARXIV_HEADER.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    for match in list(INLINE_STOP.finditer(text)):
        prefix = text[: match.start()].strip()
        if word_count(prefix) >= 40:
            text = prefix
            break
    text = re.sub(r"\b(?:This CVPR paper is the Open Access version|Except for this watermark).*", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip(" -*†‡")
    if text and not re.search(r"[.!?]$", text):
        last_sentence_end = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
        if last_sentence_end > 0 and word_count(text[: last_sentence_end + 1]) >= 40:
            text = text[: last_sentence_end + 1]
    return clean_abstract(text)


def collect_candidate_after(lines: list[str], start_idx: int) -> str | None:
    collected: list[str] = []
    started = False
    for line in lines[start_idx:]:
        if ABSTRACT_STOP.match(line):
            break
        if re.fullmatch(r"\d{2,6}", line) and word_count(join_extracted_lines(collected)) >= 40:
            break
        if re.match(r"^(?:\d+\.?\s*)?(Introduction|Related Work|Background|Method|Conclusion)\b", line, re.I):
            break
        if INLINE_STOP.search(line) and word_count(join_extracted_lines(collected)) >= 40:
            before = INLINE_STOP.split(line, maxsplit=1)[0].strip()
            if before and not is_noise_line(before):
                collected.append(before)
            break
        if not started:
            if not looks_like_abstract_start(line):
                continue
            started = True
        if is_noise_line(line):
            if re.match(r"^(figure|table)\s+\d+\b", line, re.I) and word_count(join_extracted_lines(collected)) >= 40:
                break
            continue
        collected.append(line)
        if word_count(join_extracted_lines(collected)) > 430:
            break
    abstract = sanitize_candidate(join_extracted_lines(collected))
    if 40 <= word_count(abstract) <= 450:
        return abstract
    return None


def front_matter_candidate(lines: list[str]) -> str | None:
    end = len(lines)
    for idx, line in enumerate(lines[:180]):
        if re.match(r"^Figure\s+1\b", line, re.I) or re.match(r"^1\.?\s+Introduction\b", line, re.I):
            end = idx
            break

    start = None
    for idx, line in enumerate(lines[:end]):
        if ABSTRACT_START.match(line):
            start = idx + 1
            break
    if start is None:
        for idx, line in enumerate(lines[:end]):
            if idx < 6:
                continue
            if looks_like_abstract_start(line):
                start = idx
                break
    if start is None:
        return None
    return collect_candidate_after(lines[:end], start)


def score_candidate(abstract: str, chip: dict[str, Any] | None = None) -> float:
    wc = word_count(abstract)
    if wc < 40 or wc > 450:
        return -1e9
    lower = abstract.lower()
    score = 0.0
    if chip is not None:
        tokens = distinctive_title_tokens(chip_title(chip), str(chip.get("chip_id", "")))
        score += 3.0 * sum(1 for token in tokens if token in lower)
    score += 2.0 if re.search(r"\b(we propose|we introduce|we present|this paper)\b", abstract, re.I) else 0.0
    score += 1.5 if has_numbers(abstract) else 0.0
    score += 1.0 if 120 <= wc <= 280 else 0.0
    score -= 3.0 * len(re.findall(r"\b(Figure|Table)\s+\d|arXiv:|Project Page|Introduction\b", abstract, re.I))
    score -= max(0, wc - 320) / 40.0
    return score


def extract_xml_summary(raw: str) -> str | None:
    m = re.search(r"(?is)<summary[^>]*>(.*?)</summary>", raw)
    if not m:
        return None
    summary = html_to_text(m.group(1))
    return clean_abstract(summary)


def clean_abstract(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^(abstract|a\s*b\s*s\s*t\s*r\s*a\s*c\s*t|summary)\s*[:.-]?\s*", "", text, flags=re.I)
    return text.strip()


def extract_abstract_from_text(raw: str, suffix: str, chip: dict[str, Any] | None = None) -> str | None:
    candidates: list[str] = []
    if suffix.lower() in {".html", ".xml"}:
        if suffix.lower() == ".xml":
            summary = extract_xml_summary(raw)
            if summary and 40 <= len(summary.split()) <= 450:
                candidates.append(summary)
        raw = html_to_text(raw)

    lines = normalize_pdf_lines(raw)
    front_candidate = front_matter_candidate(lines)
    if front_candidate:
        candidates.append(front_candidate)

    for i, line in enumerate(lines):
        if ABSTRACT_START.match(line):
            abstract = collect_candidate_after(lines, i + 1)
            if abstract:
                candidates.append(abstract)

    # Fallback for lines like "ABSTRACT This paper ..."
    joined = "\n".join(normalize_lines(raw)[:250])
    m = re.search(
        r"(?is)\babstract\b\s*[:.-]?\s*(.{200,3500}?)(?:\n\s*(?:1\s+introduction|introduction|keywords|ccs concepts)\b)",
        joined,
    )
    if m:
        abstract = sanitize_candidate(m.group(1))
        if 40 <= len(abstract.split()) <= 450:
            candidates.append(abstract)
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: score_candidate(candidate, chip))


def extract_abstract(chip_path: Path, chip: dict[str, Any]) -> tuple[str | None, str | None, list[str]]:
    tried: list[str] = []
    for p in candidate_source_paths(chip_path, chip):
        tried.append(str(p))
        raw = p.read_text(encoding="utf-8", errors="ignore")
        abstract = extract_abstract_from_text(raw, p.suffix, chip)
        if abstract and abstract_matches_title(abstract, chip):
            return abstract, str(p), tried
    return None, None, tried


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w.-]+\b", text))


def sentence_count(text: str) -> int:
    return max(1, len(re.findall(r"[.!?](?:\s|$)", text)))


def has_numbers(text: str) -> bool:
    return bool(re.search(r"\b\d+(?:\.\d+)?\s*(?:%|x|B|M|K|points?|tokens?|samples?|tasks?)?\b", text))


NODE_RULES = [
    ("R1_abstract_as_argument", "role", lambda chip, abs_text: True),
    ("G1_problem_gap", "gap", lambda chip, abs_text: bool(chip.get("problem_gap"))),
    ("C1_domain_context", "context", lambda chip, abs_text: bool(chip.get("chip_metadata", {}).get("domain_tags"))),
    ("O1_named_method_or_object", "object", lambda chip, abs_text: bool(chip.get("method_mechanism") or chip.get("implementation"))),
    ("M1_architecture_or_mechanism", "mechanism", lambda chip, abs_text: bool(chip.get("method_mechanism"))),
    ("M2_efficiency_or_theory_detail", "mechanism", lambda chip, abs_text: any(w in (abs_text or "").lower() for w in ["efficient", "efficiency", "theory", "theorem", "bound", "complexity", "linear", "quadratic"])),
    ("E1_evaluation_setup", "evidence", lambda chip, abs_text: bool(chip.get("evaluation_validation") or chip.get("experimental_setting"))),
    ("E2_result_outcome", "evidence", lambda chip, abs_text: bool(chip.get("result_outcome"))),
    ("E3_quantitative_anchor", "evidence", lambda chip, abs_text: has_numbers(abs_text or "")),
    ("I1_interpretation_or_tradeoff", "interpretation", lambda chip, abs_text: bool(chip.get("result_outcome") or chip.get("limitations"))),
    ("S1_bounded_takeaway", "scope", lambda chip, abs_text: True),
    ("P1_length_and_placement_prior", "style", lambda chip, abs_text: True),
]


def infer_nodes(chip: dict[str, Any], abstract: str) -> list[str]:
    return [node for node, _family, fn in NODE_RULES if fn(chip, abstract)]


def infer_edges(nodes: list[str]) -> list[str]:
    order = [node for node, _family, _fn in NODE_RULES if node in nodes]
    return [f"{a}->{b}" for a, b in zip(order, order[1:])]


def paper_signature(chip: dict[str, Any]) -> dict[str, Any]:
    meta = chip.get("chip_metadata", {})
    return {
        "chip_id": chip.get("chip_id"),
        "title": chip_title(chip),
        "venue": meta.get("venue"),
        "year": meta.get("year"),
        "domain_tags": meta.get("domain_tags", []),
        "has_problem_gap": bool(chip.get("problem_gap")),
        "has_method_mechanism": bool(chip.get("method_mechanism")),
        "has_evaluation": bool(chip.get("evaluation_validation") or chip.get("experimental_setting")),
        "has_results": bool(chip.get("result_outcome")),
        "has_limitations": bool(chip.get("limitations")),
    }


def support_table(items_by_paper: list[list[str]], total: int) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for items in items_by_paper:
        for item in set(items):
            counts[item] = counts.get(item, 0) + 1
    return [
        {"id": k, "support_count": v, "support_rate": round(v / total, 4)}
        for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]


def md_table(rows: list[dict[str, Any]], headers: list[str]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-file", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--holdout-chip-id", default="ICLR2026_HwCvaJOiCj_mamba3")
    args = parser.parse_args()

    domain_file = Path(args.domain_file)
    domain_name = parse_domain_name(domain_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chip_paths = parse_chip_paths(domain_file)
    papers: list[dict[str, Any]] = []
    holdout_private: dict[str, Any] | None = None
    for chip_path in chip_paths:
        chip = read_json(chip_path)
        abstract, abstract_path, tried = extract_abstract(chip_path, chip)
        is_holdout = chip.get("chip_id") == args.holdout_chip_id
        wc = word_count(abstract) if abstract else None
        paper = {
            "chip_id": chip.get("chip_id"),
            "title": chip_title(chip),
            "venue": chip_venue(chip),
            "chip_path": str(chip_path),
            "split": "holdout" if is_holdout else "train",
            "abstract_extraction_ok": abstract is not None,
            "abstract_extraction_status": "withheld_post_generation"
            if is_holdout
            else ("extracted" if abstract is not None else "missing"),
            "abstract_source_path": abstract_path,
            "abstract_word_count": wc,
            "abstract_sentence_count": sentence_count(abstract) if abstract else None,
            "source_paths_tried": tried[:8],
            "paper_signature": paper_signature(chip),
        }
        if not is_holdout and abstract:
            paper["abstract_text"] = abstract
            paper["selected_nodes"] = infer_nodes(chip, abstract)
            paper["selected_edges"] = infer_edges(paper["selected_nodes"])
        elif is_holdout:
            # Keep blind test clean: do not copy the held-out abstract text into the manifest.
            holdout_private = {
                "chip_id": chip.get("chip_id"),
                "title": chip_title(chip),
                "original_abstract_source_path": abstract_path,
                "original_abstract_sha256": hashlib.sha256(abstract.encode("utf-8")).hexdigest() if abstract else None,
                "original_abstract_word_count": wc,
                "source_paths_tried": tried,
                "read_after": "Read only after blind generation and verification are complete.",
            }
            paper["abstract_source_path"] = "hidden_until_after_generation"
            paper["abstract_word_count"] = None
            paper["abstract_sentence_count"] = None
            paper["source_paths_tried"] = ["hidden_until_after_generation"]
            paper["post_generation_only"] = {
                "stored_in": "holdout_private_after_generation.json",
            }
        papers.append(paper)

    train = [p for p in papers if p["split"] == "train" and p["abstract_extraction_ok"]]
    holdout_matches = [p for p in papers if p["split"] == "holdout"]
    if not holdout_matches:
        valid_ids = "\n".join(f"- {p['chip_id']}" for p in papers)
        raise SystemExit(f"holdout chip id not found: {args.holdout_chip_id}\nValid chip ids:\n{valid_ids}")
    holdout = holdout_matches[0]

    words = [p["abstract_word_count"] for p in train if p["abstract_word_count"]]
    sentences = [p["abstract_sentence_count"] for p in train if p["abstract_sentence_count"]]
    node_support = support_table([p["selected_nodes"] for p in train], len(train))
    edge_support = support_table([p["selected_edges"] for p in train], len(train))

    style_profile = {
        "domain": domain_name,
        "train_paper_count": len(train),
        "holdout_paper_count": 1,
        "abstract_word_count": {
            "min": min(words),
            "median": statistics.median(words),
            "mean": round(statistics.mean(words), 2),
            "max": max(words),
        },
        "abstract_sentence_count": {
            "median": statistics.median(sentences),
            "mean": round(statistics.mean(sentences), 2),
        },
        "recommended_target_words": int(round(statistics.median(words))),
        "recommended_band": [max(120, int(statistics.median(words) - 35)), int(statistics.median(words) + 35)],
    }

    manifest = {
        "run_name": out_dir.name,
        "created_for": "DIRS Case 1 abstract-only domain training",
        "domain_file": str(domain_file),
        "domain": domain_name,
        "total_chips": len(papers),
        "train_count": len([p for p in papers if p["split"] == "train"]),
        "train_with_abstract_count": len(train),
        "holdout_count": 1,
        "holdout_chip_id": holdout["chip_id"],
        "blind_rule": "Do not read heldout post_generation_only source before generation.",
        "papers": papers,
    }

    write_json(out_dir / "manifest.json", manifest)
    write_json(out_dir / "style_profile.json", style_profile)
    write_json(out_dir / "node_support_scores.json", node_support)
    write_json(out_dir / "edge_support_scores.json", edge_support)
    if holdout_private:
        write_json(out_dir / "holdout_private_after_generation.json", holdout_private)

    train_public = [
        {
            "chip_id": p["chip_id"],
            "title": p["title"],
            "venue": p["venue"],
            "chip_path": p["chip_path"],
            "abstract_word_count": p["abstract_word_count"],
            "selected_nodes": p["selected_nodes"],
            "selected_edges": p["selected_edges"],
        }
        for p in train
    ]
    write_json(out_dir / "training_trace.json", train_public)

    md = [
        "# DIRS Abstract Training Run",
        "",
        "Date: `2026-07-20`",
        "",
        f"Domain: `{domain_name}`",
        "",
        "## Split",
        "",
        f"- Total chips: `{len(papers)}`",
        f"- Training papers: `{len([p for p in papers if p['split'] == 'train'])}`",
        f"- Training papers with extracted abstracts: `{len(train)}`",
        f"- Held-out chip: `{holdout['chip_id']}`",
        f"- Held-out title: `{holdout['title']}`",
        "",
        "Blind rule: do not read the held-out original abstract until after generation.",
        "",
        "## Style Prior",
        "",
        "```json",
        json.dumps(style_profile, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Node Support",
        "",
        md_table(node_support, ["id", "support_count", "support_rate"]),
        "",
        "## Edge Support",
        "",
        md_table(edge_support, ["id", "support_count", "support_rate"]),
        "",
        "## Held-Out Test Card",
        "",
        "```yaml",
        f"chip_id: {holdout['chip_id']}",
        f"title: {holdout['title']}",
        f"chip_path: {holdout['chip_path']}",
        f"target_words_from_training_median: {style_profile['recommended_target_words']}",
        f"target_band: {style_profile['recommended_band']}",
        "original_abstract: hidden_until_after_generation",
        "```",
        "",
        "## Files",
        "",
        "```text",
        "manifest.json",
        "style_profile.json",
        "node_support_scores.json",
        "edge_support_scores.json",
        "training_trace.json",
        "holdout_test_card.md",
        "holdout_private_after_generation.json",
        "```",
    ]
    write_text(out_dir / "README.md", "\n".join(md) + "\n")

    holdout_md = [
        "# Held-Out Abstract Test Card",
        "",
        "Date: `2026-07-20`",
        "",
        f"Chip: `{holdout['chip_path']}`",
        "",
        f"Paper: `{holdout['title']}`",
        "",
        f"Domain: `{domain_name}`",
        "",
        "## Blind Inputs Allowed",
        "",
        "```text",
        "chip facts",
        "domain node/edge support priors",
        "style_profile.json",
        "case1_writing/01_abstract_writing.md",
        "```",
        "",
        "## Hidden Until After Generation",
        "",
        "```yaml",
        "private_file: holdout_private_after_generation.json",
        "rule: do_not_open_until_after_blind_generation_and_verification",
        "```",
        "",
        "## Target Length From Training Priors",
        "",
        "```yaml",
        f"target_words: {style_profile['recommended_target_words']}",
        f"target_band: {style_profile['recommended_band']}",
        "```",
        "",
        "## Test Procedure",
        "",
        "```text",
        "1. Read the chip only.",
        "2. Infer paper signature.",
        "3. Select connected abstract sub-DAG using node/edge support and chip compatibility.",
        "4. Generate abstract without reading hidden original.",
        "5. Verify evidence support, order, target length, and scope.",
        "6. Reveal original only for post-generation comparison.",
        "```",
    ]
    write_text(out_dir / "holdout_test_card.md", "\n".join(holdout_md) + "\n")

    print(f"wrote {out_dir}")
    print(f"train abstracts: {len(train)} / {len([p for p in papers if p['split']=='train'])}")
    print(f"holdout: {holdout['chip_id']} - {holdout['title']}")


if __name__ == "__main__":
    main()
