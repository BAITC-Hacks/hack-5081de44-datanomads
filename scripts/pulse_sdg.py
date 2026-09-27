"""Generate a review-only synthetic 109 pilot with NeMo Data Designer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.normalization.pii import scan_pii

DEFAULT_SEEDS = ROOT / "data/sdg/pilot_scenarios.jsonl"
CATALOG = ROOT / "data/catalogs/almaty_2025_taxonomy_review.json"
PROMPT_VERSION = "pulse109-appeal.v1"
LANGUAGES = {"RU", "KZ", "MIXED"}
STYLES = {"short", "conversational", "neutral"}
REQUIRED_SEED_FIELDS = {
    "scenario_id",
    "topic_id",
    "subtopic_id",
    "source_category",
    "source_service",
    "facts_ru",
    "critical_facts",
    "forbidden_invented_facts",
    "source_provenance",
    "review_status",
}
OPTIONAL_SEED_FIELDS = {"object_type", "region_constraints", "time_context", "duplicate_group", "repeat_group"}
SEED_SOURCE_PROVENANCE = "SYNTHETIC_FROM_CANDIDATE_CATALOG_PAIR"
GENERATION_SEED_FIELDS = (
    "scenario_id", "topic_id", "subtopic_id", "source_category", "source_service", "facts_ru",
)
PROMPT = """Напиши ровно одно вымышленное обращение гражданина в службу 109.
Факты ситуации: {{ facts_ru }}
Язык: {{ language }}. RU — русский; KZ — естественный казахский; MIXED — естественное смешение русского и казахского.
Стиль: {{ style }}. short — коротко; conversational — разговорно; neutral — нейтрально.
Сохрани все существенные факты. Не добавляй причину, адрес, фамилию, телефон, ИИН, исполнителя,
срок решения, действия службы или другие непредоставленные сведения.
Не называй тему, подтип и служебные метки. Верни только текст обращения без кавычек и пояснений."""


def source_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def read_seeds(path: Path) -> dict[str, dict]:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    allowed = {
        (pair["source_category"], pair["source_service"]): pair
        for pair in catalog["pairs"]
        if pair["proposal_status"] == "CANDIDATE" and pair["suggested_subtopic_id"]
    }
    seeds: dict[str, dict] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            seed = json.loads(line)
            if (not isinstance(seed, dict) or not REQUIRED_SEED_FIELDS <= set(seed) or
                    set(seed) - REQUIRED_SEED_FIELDS - OPTIONAL_SEED_FIELDS):
                raise ValueError(f"seed line {line_number}: unexpected fields")
            if any(
                not isinstance(value, str) or not value.strip()
                for key, value in seed.items()
                if key not in {"critical_facts", "forbidden_invented_facts", "region_constraints"}
            ):
                raise ValueError(
                    f"seed line {line_number}: text fields must be nonempty"
                )
            if (seed["source_provenance"] != SEED_SOURCE_PROVENANCE or
                    seed["review_status"] != "PENDING"):
                raise ValueError(f"seed line {line_number}: invalid provenance or review status")
            for field in ("critical_facts", "forbidden_invented_facts"):
                values = seed[field]
                if (not isinstance(values, list) or not values or
                        any(not isinstance(value, str) or not value.strip() for value in values) or
                        len(set(values)) != len(values)):
                    raise ValueError(f"seed line {line_number}: invalid {field}")
            regions = seed.get("region_constraints", [])
            if (not isinstance(regions, list) or
                    any(not isinstance(value, str) or not value.strip() for value in regions) or
                    len(set(regions)) != len(regions)):
                raise ValueError(f"seed line {line_number}: invalid region_constraints")
            if (any(fact not in seed["facts_ru"] for fact in seed["critical_facts"]) or
                    any(seed[field] not in seed["facts_ru"] for field in ("object_type", "time_context")
                        if field in seed)):
                raise ValueError(f"seed line {line_number}: context is not present in facts_ru")
            scenario_id = seed["scenario_id"]
            if scenario_id in seeds:
                raise ValueError(f"seed line {line_number}: duplicate scenario_id")
            pair = allowed.get((seed["source_category"], seed["source_service"]))
            if pair is None or (
                pair["suggested_topic_id"],
                pair["suggested_subtopic_id"],
            ) != (seed["topic_id"], seed["subtopic_id"]):
                raise ValueError(
                    f"seed line {line_number}: pair is outside the clear catalog scope"
                )
            if any(scan_pii(value).detected for value in (
                    seed["facts_ru"], *seed["critical_facts"], *seed["forbidden_invented_facts"],
                    *regions, *(seed[field] for field in ("object_type", "time_context") if field in seed))):
                raise ValueError(f"seed line {line_number}: sensitive-looking value")
            seeds[scenario_id] = seed
    if not seeds:
        raise ValueError("seed file is empty")
    return seeds


def write_generation_seeds(seeds: dict[str, dict], path: Path) -> None:
    """Keep review metadata in source scenarios, not Data Designer's seed columns."""
    with path.open("x", encoding="utf-8") as stream:
        for seed in seeds.values():
            stream.write(json.dumps({key: seed[key] for key in GENERATION_SEED_FIELDS},
                                    ensure_ascii=False, sort_keys=True) + "\n")


def build_config(seed_path: Path, model_id: str, generator_seed: int):
    import data_designer.config as dd

    builder = dd.DataDesignerConfigBuilder(
        model_configs=[
            dd.ModelConfig(
                alias="local-generator",
                model=model_id,
                provider="local",
                inference_parameters=dd.ChatCompletionInferenceParams(
                    temperature=0.7,
                    top_p=0.9,
                    max_tokens=220,
                    max_parallel_requests=1,
                    timeout=180,
                    extra_body={"reasoning_effort": "none", "seed": generator_seed},
                ),
            )
        ]
    )
    builder.with_seed_dataset(dd.LocalFileSeedSource(path=str(seed_path)))
    builder.add_column(
        dd.SamplerColumnConfig(
            name="language",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(
                values=["RU", "KZ", "MIXED"], weights=[45, 45, 10]
            ),
        )
    )
    builder.add_column(
        dd.SamplerColumnConfig(
            name="style",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(
                values=["short", "conversational", "neutral"]
            ),
        )
    )
    builder.add_column(
        dd.LLMTextColumnConfig(
            name="appeal_text",
            model_alias="local-generator",
            prompt=PROMPT,
        )
    )
    return builder


def clean_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.strip().strip('"«»').split())
    if not 15 <= len(text) <= 700:
        return None
    if scan_pii(text).detected:
        return None
    if text.casefold().startswith(("вот текст", "конечно", "тема:", "категория:")):
        return None
    return text


def candidate_text_key(text: str) -> str:
    return unicodedata.normalize("NFC", text.casefold())


def export_candidates(
    rows: list[dict], seeds: dict[str, dict], output: Path, model_id: str,
    scenario_checksum: str, generator_seed: int,
) -> Counter:
    counts: Counter = Counter()
    seen_texts: set[str] = set()
    with output.open("x", encoding="utf-8") as stream:
        for row in rows:
            scenario_id = row.get("scenario_id")
            seed = seeds.get(scenario_id)
            if seed is None or any(
                row.get(key) != seed[key] for key in ("topic_id", "subtopic_id")
            ):
                counts["invalid_seed_or_label"] += 1
                continue
            language, style = row.get("language"), row.get("style")
            if language not in LANGUAGES or style not in STYLES:
                counts["invalid_sampler_value"] += 1
                continue
            text = clean_text(row.get("appeal_text"))
            if text is None:
                counts["invalid_text"] += 1
                continue
            duplicate_key = candidate_text_key(text)
            if duplicate_key in seen_texts:
                counts["exact_duplicate"] += 1
                continue
            seen_texts.add(duplicate_key)
            variant_digest = hashlib.sha256(
                f"{scenario_id}\0{language}\0{style}\0{text}".encode("utf-8")
            ).hexdigest()[:16]
            candidate = {
                "variant_id": f"{scenario_id}_{variant_digest}",
                "scenario_id": scenario_id,
                "split_group": scenario_id,
                "language": language,
                "style": style,
                "text": text,
                "topic_id": seed["topic_id"],
                "subtopic_id": seed["subtopic_id"],
                "synthetic": True,
                "generator_model": model_id,
                "prompt_version": PROMPT_VERSION,
                "generator_seed": generator_seed,
                "source_scenario_sha256": scenario_checksum,
                "review_status": "PENDING",
            }
            stream.write(json.dumps(candidate, ensure_ascii=False) + "\n")
            counts["pending_review"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-path", type=Path, default=DEFAULT_SEEDS)
    parser.add_argument("--check-seeds", action="store_true")
    parser.add_argument(
        "--model", help="Model name exposed by the local OpenAI-compatible server"
    )
    parser.add_argument("--endpoint", default="http://localhost:11434/v1")
    parser.add_argument("--num-records", type=int)
    parser.add_argument("--generator-seed", type=int, default=109)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()

    scenario_checksum = source_checksum(args.seed_path)
    seeds = read_seeds(args.seed_path)
    if source_checksum(args.seed_path) != scenario_checksum:
        raise ValueError("source scenarios changed while loading")
    if args.check_seeds:
        print(f"OK: {len(seeds)} clear synthetic scenarios")
        return
    if not args.model:
        parser.error("--model is required for generation")
    if args.generator_seed < 0:
        parser.error("--generator-seed must be nonnegative")
    num_records = len(seeds) if args.num_records is None else args.num_records
    if not 1 <= num_records <= 300:
        parser.error("--num-records must be between 1 and 300 for the pilot")

    os.environ.setdefault("NEMO_TELEMETRY_ENABLED", "false")
    os.environ.setdefault("PULSE_LOCAL_LLM_API_KEY", "ollama")
    try:
        import data_designer.config as dd
        from data_designer.interface import DataDesigner
    except ImportError as exc:
        parser.error(f"install data-designer in the ML environment: {exc}")

    run_dir = args.run_dir or ROOT / "data/sdg/runs" / datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    generation_seeds = run_dir / "generation_seeds.jsonl"
    write_generation_seeds(seeds, generation_seeds)
    provider = dd.ModelProvider(
        name="local",
        endpoint=args.endpoint,
        provider_type="openai",
        api_key="PULSE_LOCAL_LLM_API_KEY",
    )
    designer = DataDesigner(
        model_providers=[provider], artifact_path=run_dir / "artifacts"
    )
    config = build_config(generation_seeds.resolve(), args.model, args.generator_seed)
    designer.validate(config)
    result = designer.create(
        config, num_records=num_records, dataset_name="pulse109-pilot"
    )
    dataset = result.load_dataset()
    rows = dataset.to_dict(orient="records")
    if source_checksum(args.seed_path) != scenario_checksum:
        raise ValueError("source scenarios changed during generation")
    counts = export_candidates(rows, seeds, run_dir / "candidates.jsonl", args.model,
                               scenario_checksum, args.generator_seed)
    (run_dir / "summary.json").write_text(
        json.dumps({"counts": dict(counts), "source_scenario_sha256": scenario_checksum,
                    "generation_seed_sha256": source_checksum(generation_seeds),
                    "prompt_version": PROMPT_VERSION, "generator_model": args.model,
                    "generator_seed": args.generator_seed}, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Run: {run_dir}")
    print(f"Candidates pending human review: {counts['pending_review']} / {len(rows)}")


if __name__ == "__main__":
    main()
