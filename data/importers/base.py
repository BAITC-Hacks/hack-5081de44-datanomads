"""Common CSV/JSON reader and quality-gate behavior."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import unicodedata
import zipfile
import xml.etree.ElementTree as ET
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from data.normalization.pipeline import NormalizationResult, normalize_row
from data.quarantine.records import QuarantineRecord
from data.schemas.taxonomy import canonical_source_system
from data.schemas.unified_ticket import SchemaValidationError


@dataclass
class ImportResult:
    source_system: str
    profile_version: str = "synthetic-aliases.v1"
    profile_status: str = "SYNTHETIC_TEST_ONLY"
    tickets: List[Any] = field(default_factory=list)
    quarantine: List[QuarantineRecord] = field(default_factory=list)
    duplicate_external_ids: List[str] = field(default_factory=list)
    schema_fingerprints: List[str] = field(default_factory=list)

    @property
    def valid_count(self) -> int:
        return len(self.tickets)

    @property
    def quarantine_count(self) -> int:
        return len(self.quarantine)

    def to_summary(self) -> Mapping[str, Any]:
        return {
            "source_system": self.source_system,
            "profile_version": self.profile_version,
            "profile_status": self.profile_status,
            "schema_fingerprints": sorted(set(self.schema_fingerprints)),
            "valid_count": self.valid_count,
            "quarantine_count": self.quarantine_count,
            "duplicate_external_ids": len(self.duplicate_external_ids),
        }


class SourceImporter:
    """Base importer with source-specific aliases supplied by subclasses."""

    source_system: str = "unknown"
    profile_version: str = "synthetic-aliases.v1"
    profile_status: str = "SYNTHETIC_TEST_ONLY"
    csv_delimiter: str = ","
    required_headers = frozenset({"external_ticket_id", "region_id", "created_at", "original_text"})
    # These are test hypotheses, not verified production export schemas.
    # Canonical field -> accepted exact source headers (ignoring case and outer whitespace).
    field_aliases: Mapping[str, Sequence[str]] = {
        "external_ticket_id": ("external_ticket_id", "ticket_id", "id", "appeal_id", "request_id"),
        "region_id": ("region_id", "region", "oblast", "область", "регион"),
        "created_at": ("created_at", "created_date", "created", "date", "дата создания", "дата"),
        "original_text": ("original_text", "text", "description", "message", "appeal_text", "обращение", "текст"),
        "language": ("language", "lang", "язык", "тіл"),
        "topic_raw": ("topic_raw", "topic", "direction", "category", "тема", "направление"),
        "service_raw": ("service_raw", "service", "executor", "organization", "исполнитель", "служба"),
        "priority": ("priority", "urgency", "приоритет"),
        "status": ("status", "state", "статус"),
        "district": ("district", "район"),
        "address": ("address", "full_address", "адрес", "мекенжай"),
        "channel": ("channel", "source_channel", "канал"),
        "closed_at": ("closed_at", "closed_date", "дата закрытия"),
        "deadline_at": ("deadline_at", "deadline", "срок"),
        "resolution_text": ("resolution_text", "resolution", "решение"),
        "official_response": ("official_response", "response", "ответ"),
        "coordinates": ("coordinates", "geo", "координаты"),
        "attachments": ("attachments", "attachment", "вложения"),
    }

    def __init__(self, source_system: Optional[str] = None):
        if source_system:
            canonical = canonical_source_system(source_system)
            if canonical:
                self.source_system = canonical
            else:
                self.source_system = source_system

    @staticmethod
    def _header_key(value: object) -> str:
        return unicodedata.normalize("NFKC", str(value or "")).strip().casefold()

    def _header_map(self, headers: Sequence[str]) -> Mapping[str, str]:
        normalized: Dict[str, str] = {}
        for header in headers:
            key = self._header_key(header)
            if not key or key in normalized:
                raise SchemaValidationError("UNKNOWN_SCHEMA", "empty or duplicate source header")
            normalized[key] = header
        result: Dict[str, str] = {}
        for canonical, aliases in self.field_aliases.items():
            matches = {normalized[self._header_key(alias)] for alias in aliases if self._header_key(alias) in normalized}
            if len(matches) > 1:
                raise SchemaValidationError("UNKNOWN_SCHEMA", f"ambiguous source headers for {canonical}")
            if matches:
                result[canonical] = next(iter(matches))
        if not self.required_headers.issubset(result):
            raise SchemaValidationError("UNKNOWN_SCHEMA", "source headers do not identify all required fields")
        if len(set(result.values())) != len(result):
            raise SchemaValidationError("UNKNOWN_SCHEMA", "one source header maps to multiple fields")
        return result

    def schema_fingerprint(self, headers: Sequence[str]) -> str:
        signature = {
            "source_system": self.source_system,
            "profile_version": self.profile_version,
            "headers": sorted(self._header_key(header) for header in headers),
        }
        return hashlib.sha256(json.dumps(signature, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

    def canonicalize_row(self, raw_row: Mapping[str, Any], header_map: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
        mapping = header_map if header_map is not None else self._header_map(list(raw_row.keys()))
        return {canonical: raw_row.get(source_key) for canonical, source_key in mapping.items()}

    def import_rows(self, rows: Iterable[Mapping[str, Any]], *, start_row: int = 1,
                    row_numbers: Iterable[int] | None = None) -> ImportResult:
        result = ImportResult(source_system=self.source_system, profile_version=self.profile_version, profile_status=self.profile_status)
        seen_ids = set()
        numbered_rows = (enumerate(rows, start=start_row) if row_numbers is None else
                         zip(row_numbers, rows, strict=True))
        for row_number, raw_row in numbered_rows:
            try:
                header_map = self._header_map(list(raw_row.keys()))
                canonical_row = self.canonicalize_row(raw_row, header_map)
            except SchemaValidationError as error:
                result.quarantine.append(
                    QuarantineRecord(
                        source_system=self.source_system,
                        row_number=row_number,
                        reason=error.code,
                        detail=str(error),
                        row=raw_row if isinstance(raw_row, Mapping) else {"row": str(raw_row)},
                    )
                )
                continue
            fingerprint = self.schema_fingerprint(list(raw_row.keys()))
            if fingerprint not in result.schema_fingerprints:
                result.schema_fingerprints.append(fingerprint)
            normalized: NormalizationResult = normalize_row(
                canonical_row,
                source_system=self.source_system,
                row_number=row_number,
            )
            if normalized.ticket is not None:
                external_id = normalized.ticket.external_ticket_id
                if external_id in seen_ids:
                    result.duplicate_external_ids.append(external_id)
                    result.quarantine.append(
                        QuarantineRecord(
                            source_system=self.source_system,
                            row_number=row_number,
                            reason="INVALID_VALUE",
                            detail="duplicate external_ticket_id",
                            field="external_ticket_id",
                            row=raw_row,
                        )
                    )
                    continue
                seen_ids.add(external_id)
                result.tickets.append(normalized.ticket)
            elif normalized.quarantine is not None:
                result.quarantine.append(normalized.quarantine)
        return result

    def _read_csv(self, path: Path) -> Tuple[List[Tuple[int, Mapping[str, Any]]], List[QuarantineRecord]]:
        rows: List[Tuple[int, Mapping[str, Any]]] = []
        errors: List[QuarantineRecord] = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t" if path.suffix.lower() == ".tsv" else self.csv_delimiter, strict=True)
            try:
                headers = next(reader)
            except StopIteration:
                return rows, [
                    QuarantineRecord(
                        self.source_system,
                        1,
                        "BAD_CSV_STRUCTURE",
                        "CSV has no header row",
                        {},
                    )
                ]
            except (csv.Error, UnicodeError):
                return rows, [QuarantineRecord(self.source_system, 1, "BAD_CSV_STRUCTURE", "invalid CSV header", {})]
            if not headers or any(not str(header).strip() for header in headers):
                return rows, [
                    QuarantineRecord(
                        self.source_system,
                        1,
                        "BAD_CSV_STRUCTURE",
                        "CSV contains an empty header",
                        {str(i): value for i, value in enumerate(headers)},
                    )
                ]
            try:
                self._header_map(headers)
            except SchemaValidationError as error:
                return rows, [
                    QuarantineRecord(
                        self.source_system,
                        1,
                        "UNKNOWN_SCHEMA",
                        str(error),
                        {str(i): value for i, value in enumerate(headers)},
                    )
                ]
            while True:
                row_number = reader.line_num + 1
                try:
                    values = next(reader)
                except StopIteration:
                    break
                except (csv.Error, UnicodeError):
                    return [], [QuarantineRecord(self.source_system, row_number, "BAD_CSV_STRUCTURE",
                                                 "CSV parsing failed; entire file rejected", {})]
                if len(values) != len(headers):
                    errors.append(QuarantineRecord(self.source_system, row_number, "BAD_CSV_STRUCTURE", f"expected {len(headers)} columns, got {len(values)}", {str(i): value for i, value in enumerate(values)}))
                else:
                    rows.append((row_number, dict(zip(headers, values))))
        return rows, errors

    def _read_json(self, path: Path) -> Tuple[List[Tuple[int, Mapping[str, Any]]], List[QuarantineRecord]]:
        rows: List[Tuple[int, Mapping[str, Any]]] = []
        errors: List[QuarantineRecord] = []
        text = path.read_text(encoding="utf-8-sig")
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                values = parsed
            elif isinstance(parsed, Mapping) and isinstance(parsed.get("tickets"), list):
                # Accept deterministic dataset manifests as well as a plain
                # object-per-file export; manifest metadata never becomes a
                # ticket row.
                values = parsed["tickets"]
            else:
                values = [parsed]
            numbered_values = list(enumerate(values, start=1))
        except json.JSONDecodeError:
            numbered_values = []
            for row_number, line in enumerate(text.splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    numbered_values.append((row_number, json.loads(line)))
                except json.JSONDecodeError as exc:
                    errors.append(
                        QuarantineRecord(
                            self.source_system,
                            row_number,
                            "UNKNOWN_SCHEMA",
                            f"invalid JSON row: {exc.msg}",
                            {"line": line},
                        )
                    )
        for row_number, value in numbered_values:
            if not isinstance(value, Mapping):
                errors.append(
                    QuarantineRecord(
                        self.source_system,
                        row_number,
                        "UNKNOWN_SCHEMA",
                        "JSON row must be an object",
                        {"value": str(value)},
                    )
                )
            else:
                rows.append((row_number, value))
        return rows, errors

    def _read_xlsx(self, path: Path) -> Tuple[List[Tuple[int, Mapping[str, Any]]], List[QuarantineRecord]]:
        """Read the first worksheet using only the XLSX ZIP/XML contract.

        Government exports are often simple tabular workbooks.  Keeping this
        parser dependency-free makes the ingestion CLI usable in the same
        restricted environment as the validation tests; formulas and rich
        formatting are intentionally ignored.
        """

        namespace = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        errors: List[QuarantineRecord] = []
        try:
            with zipfile.ZipFile(path) as archive:
                shared: List[str] = []
                if "xl/sharedStrings.xml" in archive.namelist():
                    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
                    for item in root.findall("main:si", namespace):
                        shared.append("".join(node.text or "" for node in item.iter() if node.tag.endswith("}t")))
                sheet_name = "xl/worksheets/sheet1.xml"
                if sheet_name not in archive.namelist():
                    raise ValueError("workbook has no first worksheet")
                root = ET.fromstring(archive.read(sheet_name))
                matrix: List[Tuple[int, List[str]]] = []
                for ordinal, row in enumerate(root.findall(".//main:sheetData/main:row", namespace), start=1):
                    row_number = int(row.attrib.get("r", ordinal))
                    values: List[str] = []
                    for cell in row.findall("main:c", namespace):
                        cell_type = cell.attrib.get("t")
                        value = cell.find("main:v", namespace)
                        inline = cell.find("main:is/main:t", namespace)
                        text = inline.text if inline is not None else (value.text if value is not None else "")
                        if cell_type == "s" and text:
                            text = shared[int(text)] if int(text) < len(shared) else ""
                        values.append(text or "")
                    matrix.append((row_number, values))
                if not matrix:
                    return [], [QuarantineRecord(self.source_system, 1, "BAD_CSV_STRUCTURE", "XLSX has no rows", {})]
                headers = matrix[0][1]
                if not headers or any(not str(header).strip() for header in headers):
                    return [], [QuarantineRecord(self.source_system, 1, "BAD_CSV_STRUCTURE", "XLSX contains an empty header", {str(i): value for i, value in enumerate(headers)})]
                rows: List[Tuple[int, Mapping[str, Any]]] = []
                for row_number, values in matrix[1:]:
                    if len(values) != len(headers):
                        errors.append(QuarantineRecord(self.source_system, row_number, "BAD_CSV_STRUCTURE", f"expected {len(headers)} columns, got {len(values)}", {str(i): value for i, value in enumerate(values)}))
                        continue
                    rows.append((row_number, dict(zip(headers, values))))
                return rows, errors
        except (OSError, zipfile.BadZipFile, ET.ParseError, ValueError, IndexError) as error:
            return [], [QuarantineRecord(self.source_system, 1, "UNKNOWN_SCHEMA", f"invalid XLSX: {error}", {"path": str(path)})]

    def import_file(self, path: Path | str) -> ImportResult:
        """Import CSV, JSON array or JSONL and retain every bad-row reason."""

        source_path = Path(path)
        suffix = source_path.suffix.lower()
        if suffix in {".json", ".jsonl", ".ndjson"}:
            numbered_rows, parse_errors = self._read_json(source_path)
        elif suffix in {".csv", ".tsv"}:
            numbered_rows, parse_errors = self._read_csv(source_path)
        elif suffix in {".xlsx", ".xlsm"}:
            numbered_rows, parse_errors = self._read_xlsx(source_path)
        else:
            return ImportResult(
                source_system=self.source_system,
                profile_version=self.profile_version,
                profile_status=self.profile_status,
                quarantine=[
                    QuarantineRecord(
                        self.source_system,
                        1,
                        "UNKNOWN_SCHEMA",
                        f"unsupported file extension: {suffix}",
                        {"path": str(source_path)},
                    )
                ],
            )
        result = self.import_rows((row for _, row in numbered_rows),
                                  row_numbers=(number for number, _ in numbered_rows))
        result.quarantine = sorted(parse_errors + result.quarantine, key=lambda row: row.row_number)
        return result


__all__ = ["ImportResult", "SourceImporter"]
