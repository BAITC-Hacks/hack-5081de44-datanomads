#!/usr/bin/env python3
"""Build a balanced, scenario-split synthetic classifier corpus for local demos."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import itertools
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.normalization.pii import scan_pii
from data.schemas.taxonomy import TOPIC_DEFINITIONS


SEED = 109
SCENARIO_BANK = ROOT / "data/sdg/classifier_scenarios.tsv"
CHALLENGE_BANK = ROOT / "data/sdg/classifier_challenges.tsv"
DEFAULT_OUTPUT = ROOT / "data/sdg/generated/classifier_v2"
SPLITS = ("train",) * 7 + ("validation",) + ("test",) * 2
TOPIC_IDS = {topic["id"] for topic in TOPIC_DEFINITIONS}
CHALLENGE_DECISIONS = {"UNKNOWN", "OTHER", "NEEDS_REVIEW"}
CHALLENGE_LANGUAGES = {"RU", "KZ", "MIXED"}
REQUIRED_BOUNDARIES = {
    frozenset(pair) for pair in (
        ("street_lighting", "electricity"),
        ("buildings", "street_lighting"),
        ("roads", "public_transport"),
        ("wastewater", "roads"),
        ("landscaping", "roads"),
    )
}
PHRASES = {
    "RU": {
        "openers": ("", "Здравствуйте. ", "Добрый день. ", "Нужна помощь. ", "Пишу с проблемой. "),
        "requests": (
            "", "Проверьте, пожалуйста.", "Когда это исправят?",
            "Прошу разобраться.",
        ),
    },
    "KZ": {
        "openers": ("", "Сәлеметсіз бе. ", "Қайырлы күн. ", "Көмек қажет. ", "Мәселе бойынша жазып отырмын. "),
        "requests": (
            "", "Тексеріп беріңізші.", "Қашан түзетіледі?",
            "Мәселені қарауыңызды сұраймын.",
        ),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_scenarios(path: Path) -> list[dict[str, str]]:
    expected_topics = {topic["id"] for topic in TOPIC_DEFINITIONS}
    counts: Counter[str] = Counter()
    scenarios = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["topic_id", "ru", "kk"]:
            raise ValueError("scenario bank must have topic_id, ru, kk columns")
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"invalid scenario structure at line {line_number}")
            topic = row["topic_id"]
            if topic not in expected_topics or not row["ru"].strip() or not row["kk"].strip():
                raise ValueError(f"invalid scenario at line {line_number}")
            if scan_pii(row["ru"]).detected or scan_pii(row["kk"]).detected:
                raise ValueError(f"sensitive-looking scenario at line {line_number}")
            counts[topic] += 1
            scenario_id = f"{topic}-{counts[topic]:02d}"
            scenarios.append({"scenario_id": scenario_id, "topic_id": topic, "RU": row["ru"].strip(), "KZ": row["kk"].strip()})
    if set(counts) != expected_topics or any(count != 10 for count in counts.values()):
        raise ValueError("scenario bank must contain exactly 10 scenarios for each of the 16 topics")
    return scenarios


def read_challenges(path: Path) -> list[dict[str, object]]:
    challenges: list[dict[str, object]] = []
    groups: dict[str, tuple[str, tuple[str, ...], set[str]]] = {}
    seen_texts: set[str] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["scenario_id", "proposed_decision", "candidate_topics", "language", "text"]:
            raise ValueError("challenge bank has unexpected columns")
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"invalid challenge structure at line {line_number}")
            scenario_id = row["scenario_id"]
            decision = row["proposed_decision"]
            language = row["language"]
            text = row["text"].strip()
            topics = tuple(row["candidate_topics"].split("|")) if row["candidate_topics"] else ()
            if (not re.fullmatch(r"[a-z][a-z0-9_]*", scenario_id) or
                    decision not in CHALLENGE_DECISIONS or language not in CHALLENGE_LANGUAGES or
                    not text or scan_pii(text).detected or text.casefold() in seen_texts or
                    (decision == "NEEDS_REVIEW" and (len(topics) != 2 or
                     len(set(topics)) != 2 or any(topic not in TOPIC_IDS for topic in topics))) or
                    (decision != "NEEDS_REVIEW" and topics)):
                raise ValueError(f"invalid challenge at line {line_number}")
            previous = groups.get(scenario_id)
            if previous is None:
                groups[scenario_id] = (decision, topics, {language})
            elif previous[:2] != (decision, topics) or language in previous[2]:
                raise ValueError(f"inconsistent challenge group at line {line_number}")
            else:
                previous[2].add(language)
            seen_texts.add(text.casefold())
            challenges.append({
                "id": f"{scenario_id}-{language.lower()}",
                "scenario_id": scenario_id,
                "language": language,
                "text": text,
                "proposed_decision": decision,
                "candidate_topics": list(topics),
                "synthetic": True,
                "review_status": "PENDING",
                "approved_for_training": False,
            })
    boundaries = {frozenset(topics) for decision, topics, _ in groups.values() if decision == "NEEDS_REVIEW"}
    if (not groups or any(languages != CHALLENGE_LANGUAGES for _, _, languages in groups.values()) or
            not REQUIRED_BOUNDARIES <= boundaries or
            not CHALLENGE_DECISIONS <= {decision for decision, _, _ in groups.values()}):
        raise ValueError("challenge bank lacks RU/KZ/MIXED or required decision and boundary coverage")
    return challenges


def render_variants(fact: str, language: str, scenario_id: str) -> list[str]:
    phrases = PHRASES[language]
    variants = set()
    for opener, request in itertools.product(phrases["openers"], phrases["requests"]):
        text = f"{opener}{fact}"
        if request:
            text += f" {request}"
        variants.add(text)
    ordered = sorted(variants)
    random.Random(f"{SEED}:{scenario_id}:{language}").shuffle(ordered)
    return ordered


def generate(output_dir: Path, bank_path: Path = SCENARIO_BANK,
             challenge_bank_path: Path = CHALLENGE_BANK) -> dict[str, object]:
    scenarios = read_scenarios(bank_path)
    challenges = read_challenges(challenge_bank_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    paths = {split: output_dir / f"{split}.jsonl" for split in ("train", "validation", "test")}
    counts: Counter[tuple[str, str, str]] = Counter()
    groups: dict[str, set[str]] = {split: set() for split in paths}
    seen_texts = set()
    with paths["train"].open("x", encoding="utf-8") as train, paths["validation"].open("x", encoding="utf-8") as validation, paths["test"].open("x", encoding="utf-8") as test:
        streams = {"train": train, "validation": validation, "test": test}
        for scenario in scenarios:
            topic_id = scenario["topic_id"]
            scenario_id = scenario["scenario_id"]
            issue_index = int(scenario_id.rsplit("-", 1)[1]) - 1
            split = SPLITS[issue_index]
            groups[split].add(scenario_id)
            for language in ("RU", "KZ"):
                variants = render_variants(scenario[language], language, scenario_id)
                for variant_index, text in enumerate(variants, start=1):
                    if text in seen_texts or scan_pii(text).detected:
                        raise ValueError(f"duplicate or sensitive-looking generated text for {scenario_id}")
                    seen_texts.add(text)
                    record = {
                        "id": f"{scenario_id}-{language.lower()}-{variant_index:03d}",
                        "scenario_id": scenario_id,
                        "split": split,
                        "language": language,
                        "topic_id": topic_id,
                        "text": text,
                        "synthetic": True,
                        "review_status": "PENDING",
                    }
                    streams[split].write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    counts[(split, topic_id, language)] += 1

    if groups["train"] & groups["validation"] or groups["train"] & groups["test"] or groups["validation"] & groups["test"]:
        raise ValueError("scenario groups overlap across splits")
    challenge_path = output_dir / "challenges.jsonl"
    with challenge_path.open("x", encoding="utf-8") as stream:
        for challenge in challenges:
            stream.write(json.dumps(challenge, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "dataset_version": "synthetic-classifier-v2",
        "synthetic": True,
        "purpose": "local classifier demo; not evidence of quality on real 109 appeals",
        "review_status": "PENDING",
        "generation_method": "deterministic_scenario_composition",
        "seed": SEED,
        "scenario_bank_sha256": sha256(bank_path),
        "challenge_bank_sha256": sha256(challenge_bank_path),
        "challenge_file": {"name": challenge_path.name, "sha256": sha256(challenge_path)},
        "challenge_count": len(challenges),
        "challenge_scenario_count": len({row["scenario_id"] for row in challenges}),
        "challenge_proposed_decision_counts": dict(sorted(
            Counter(row["proposed_decision"] for row in challenges).items()
        )),
        "scenario_count": len(scenarios),
        "record_count": len(seen_texts),
        "split_counts": {split: sum(count for (name, _, _), count in counts.items() if name == split) for split in paths},
        "scenario_counts_by_split": {split: len(group) for split, group in groups.items()},
        "topic_count": len(TOPIC_DEFINITIONS),
        "language_counts": {language: sum(count for (_, _, name), count in counts.items() if name == language) for language in PHRASES},
        "files": {split: {"name": path.name, "sha256": sha256(path)} for split, path in paths.items()},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = generate(args.output_dir)
    print(json.dumps({key: manifest[key] for key in (
        "dataset_version", "record_count", "split_counts", "scenario_count",
        "language_counts", "challenge_count",
    )}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
