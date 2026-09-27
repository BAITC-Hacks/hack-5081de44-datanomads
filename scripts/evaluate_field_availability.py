#!/usr/bin/env python3
"""Describe safe field availability in the two customer 109 CSV exports."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys


HEADERS = {
    "vko_109": {
        "application_number", "creation_date", "closing_date", "region", "district", "street",
        "full_name", "applicant_number", "application_type", "submittal_channel", "category",
        "service", "contractor", "com_exp", "result", "status", "operator",
    },
    "almaty_109": {
        "application_number", "creation_date", "closing_date", "com_exp", "result",
        "contractor", "status", "category", "service", "status_1", "submittal_channel",
    },
}

FACT_COLUMNS = {
    "source_record_id": ("application_number", "application_number"),
    "created_at": ("creation_date", "creation_date"),
    "closed_at": ("closing_date", "closing_date"),
    "raw_category": ("category", "category"),
    "raw_service": ("service", "service"),
    "raw_contractor": ("contractor", "contractor"),
    "raw_status": ("status", "status"),
    "secondary_status": (None, "status_1"),
    "raw_result": ("result", "result"),
    "channel": ("submittal_channel", "submittal_channel"),
    "region_hint": ("region", None),
    "district_hint": ("district", None),
    "street_hint": ("street", None),
    "application_type": ("application_type", None),
    "original_appeal_text": (None, None),
    "language_label": (None, None),
    "priority_label": (None, None),
    "object_id": (None, None),
    "verified_resolution": (None, None),
}

SEMANTICS = {
    "source_record_id": "SOURCE_LOCAL_IDENTIFIER",
    "created_at": "PARSEABLE_LOCAL_DATETIME_TIMEZONE_UNVERIFIED",
    "closed_at": "DATE_FIELD_NOT_VERIFIED_RESOLUTION",
    "raw_category": "UNVERIFIED_CANONICAL_TOPIC",
    "raw_service": "UNVERIFIED_SERVICE_ROLE",
    "raw_contractor": "UNVERIFIED_EXECUTOR_ROLE",
    "raw_status": "UNVERIFIED_STATUS_MEANING",
    "secondary_status": "UNVERIFIED_STATUS_MEANING",
    "raw_result": "UNVERIFIED_OUTCOME_MEANING",
    "channel": "UNVERIFIED_CHANNEL_CODEBOOK",
    "region_hint": "CITY_OR_DISTRICT_NOT_CANONICAL_REGION",
    "district_hint": "UNVERIFIED_GEO_UNIT",
    "street_hint": "ADDRESS_LIKE_PII_PRESENCE_ONLY",
    "application_type": "UNVERIFIED_APPLICATION_TYPE",
    "original_appeal_text": "ABSENT_CONFIRMED_BY_CUSTOMER",
    "language_label": "ABSENT",
    "priority_label": "ABSENT",
    "object_id": "ABSENT",
    "verified_resolution": "ABSENT",
}

WITHIN_SOURCE = {
    "source_record_id": "ID_EQUALITY_ONLY",
    "created_at": "LOCAL_DATETIME_ORDER_ONLY",
    "closed_at": "LOCAL_DATETIME_ORDER_ONLY",
    "raw_category": "RAW_VALUE_EQUALITY_ONLY",
    "raw_service": "RAW_VALUE_EQUALITY_ONLY",
    "raw_contractor": "RAW_VALUE_EQUALITY_ONLY",
    "raw_status": "RAW_VALUE_EQUALITY_ONLY",
    "secondary_status": "RAW_VALUE_EQUALITY_ONLY",
    "raw_result": "RAW_VALUE_EQUALITY_ONLY",
    "channel": "RAW_VALUE_EQUALITY_ONLY",
    "region_hint": "RAW_VALUE_EQUALITY_ONLY",
    "district_hint": "RAW_VALUE_EQUALITY_ONLY",
    "street_hint": "PRESENCE_ONLY",
    "application_type": "RAW_VALUE_EQUALITY_ONLY",
}

JOINT_FACTS = {
    "created_and_category": ("created_at", "raw_category"),
    "category_and_service": ("raw_category", "raw_service"),
    "closed_and_result": ("closed_at", "raw_result"),
    "location_hints": ("region_hint", "district_hint", "street_hint"),
}

CROSS_SOURCE = {
    "source_record_id": "NO_SOURCE_LOCAL_IDS",
    "created_at": "DATE_ONLY_TIMEZONE_UNVERIFIED",
    "closed_at": "DATE_ONLY_TIMEZONE_AND_OUTCOME_UNVERIFIED",
    "raw_category": "NO_UNMAPPED_TAXONOMIES",
    "raw_service": "NO_DIFFERENT_OR_UNVERIFIED_FIELD_ROLE",
    "raw_contractor": "NO_EXECUTOR_SEMANTICS_UNVERIFIED",
    "raw_status": "NO_STATUS_SEMANTICS_UNVERIFIED",
    "secondary_status": "NO_ABSENT_IN_VKO",
    "raw_result": "NO_OUTCOME_SEMANTICS_UNVERIFIED",
    "channel": "NO_CODEBOOK_UNVERIFIED",
    "region_hint": "NO_ABSENT_IN_ALMATY_AND_NONCANONICAL_IN_VKO",
    "district_hint": "NO_ABSENT_IN_ALMATY",
    "street_hint": "NO_PII_AND_ABSENT_IN_ALMATY",
    "application_type": "NO_ABSENT_IN_ALMATY",
    "original_appeal_text": "UNAVAILABLE_IN_BOTH",
    "language_label": "UNAVAILABLE_IN_BOTH",
    "priority_label": "UNAVAILABLE_IN_BOTH",
    "object_id": "UNAVAILABLE_IN_BOTH",
    "verified_resolution": "UNAVAILABLE_IN_BOTH",
}


def _source_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _date_present(value: str) -> bool:
    if not value:
        return False
    try:
        datetime.strptime(value, "%d.%m.%Y %H:%M:%S")
    except ValueError:
        return False
    return True


def scan_profile(path: Path, profile: str) -> dict:
    if profile not in HEADERS:
        raise ValueError("unknown customer source profile")
    index = 0 if profile == "vko_109" else 1
    columns = {fact: choices[index] for fact, choices in FACT_COLUMNS.items()}
    presence: Counter = Counter()
    joint: Counter = Counter()
    identifiers = set()
    row_count = malformed_count = duplicate_ids = invalid_creation_dates = 0
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if len(reader.fieldnames or []) != len(HEADERS[profile]) or set(reader.fieldnames or []) != HEADERS[profile]:
            raise ValueError("customer CSV schema differs from the reviewed profile")
        for row in reader:
            row_count += 1
            if None in row or any(value is None for value in row.values()):
                malformed_count += 1
                continue
            observed = set()
            for fact, column in columns.items():
                if column is None:
                    continue
                value = row[column].strip()
                if not value:
                    continue
                if fact in {"created_at", "closed_at"} and not _date_present(value):
                    if fact == "created_at":
                        invalid_creation_dates += 1
                    continue
                observed.add(fact)
                presence[fact] += 1
            identifier = row["application_number"].strip()
            if identifier:
                if identifier in identifiers:
                    duplicate_ids += 1
                identifiers.add(identifier)
            for name, needed in JOINT_FACTS.items():
                if set(needed).issubset(observed):
                    joint[name] += 1
    fields = {}
    for fact, column in columns.items():
        count = presence[fact]
        semantic_status = (SEMANTICS[fact] if column is not None or fact == "original_appeal_text"
                           else "ABSENT")
        if profile == "almaty_109" and fact == "raw_service":
            semantic_status = "OBSERVED_ISSUE_TYPE_NOT_VERIFIED_EXECUTOR"
        fields[fact] = {
            "source_column": column,
            "present_count": count,
            "missing_or_invalid_count": row_count - count,
            "presence_rate": round(count / row_count, 6) if row_count else None,
            "semantic_status": semantic_status,
            "within_source_comparison": WITHIN_SOURCE.get(fact, "UNAVAILABLE") if column else "UNAVAILABLE",
        }
    return {
        "profile_id": profile,
        "source_sha256": _source_checksum(path),
        "schema_fingerprint_sha256": "sha256:" + hashlib.sha256("\n".join(sorted(HEADERS[profile])).encode()).hexdigest(),
        "record_count": row_count,
        "malformed_row_count": malformed_count,
        "duplicate_source_id_count": duplicate_ids,
        "invalid_creation_date_count": invalid_creation_dates,
        "fields": fields,
        "joint_presence": {name: joint[name] for name in JOINT_FACTS},
    }


def build_report(vko: Path, almaty: Path) -> dict:
    profiles = [scan_profile(vko, "vko_109"), scan_profile(almaty, "almaty_109")]
    return {
        "report_version": "customer-field-availability.v1",
        "scope": "two_customer_109_exports_structural_metadata_only",
        "profiles": profiles,
        "cross_source_value_comparison": CROSS_SOURCE,
        "excluded_columns": {
            "com_exp": "CUSTOMER_CONFIRMED_NOT_ORIGINAL_APPEAL_TEXT",
            "full_name": "PII_EXCLUDED_WHERE_PRESENT",
            "applicant_number": "PII_EXCLUDED_WHERE_PRESENT",
            "operator": "OPERATOR_IDENTIFIER_EXCLUDED_WHERE_PRESENT",
        },
        "required_follow_up_questions": [],
        "recommended_routing_or_priority_rules": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vko", type=Path, required=True)
    parser.add_argument("--almaty", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError("field availability report already exists")
        report = build_report(args.vko, args.almaty)
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
    except Exception as error:
        print(json.dumps({"status": "FAILED", "error": "FIELD_AVAILABILITY_FAILED",
                          "error_type": type(error).__name__}), file=sys.stderr)
        return 2
    print(json.dumps({"status": "COMPLETED", "profiles": [profile["profile_id"] for profile in report["profiles"]]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
