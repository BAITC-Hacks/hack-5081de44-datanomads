"""Common CSV/JSON reader and quality-gate behavior."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import json
from pathlib import Path
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
    tickets: List[Any] = field(default_factory=list)
    quarantine: List[QuarantineRecord] = field(default_factory=list)
    duplicate_external_ids: List[str] = field(default_factory=list)

    @property
    def valid_count(self) -> int:
        return len(self.tickets)

    @property
    def quarantine_count(self) -> int:
        return len(self.quarantine)

    def to_summary(self) -> Mapping[str, Any]:
        return {
            "source_system": self.source_system,
            "valid_count": self.valid_count,
            "quarantine_count": self.quarantine_count,
            "duplicate_external_ids": len(self.duplicate_external_ids),
        }


class SourceImporter:
    """Base importer with source-specific aliases supplied by subclasses."""

    source_system: str = "unknown"
    # Canonical field -> accepted source headers.  Header matching is
    # case/spacing/punctuation insensitive.
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
        import re
        import unicodedata

        text = unicodedata.normalize("NFKC", str(value or "")).strip().lower()
        text = text.replace("ё", "е")
        return re.sub(r"[^\w]+", "_", text, flags=re.UNICODE).strip("_")

    def _header_map(self, headers: Sequence[str]) -> Mapping[str, str]:
        normalized = {self._header_key(header): header for header in headers}
        result: Dict[str, str] = {}
        for canonical, aliases in self.field_aliases.items():
            for alias in aliases:
                candidate = normalized.get(self._header_key(alias))
                if candidate is not None:
                    result[canonical] = candidate
                    break
        return result

    def canonicalize_row(self, raw_row: Mapping[str, Any], header_map: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
        mapping = header_map or self._header_map(list(raw_row.keys()))
        if not mapping:
            raise SchemaValidationError("UNKNOWN_SCHEMA", "no known columns in source row")
        return {canonical: raw_row.get(source_key) for canonical, source_key in mapping.items()}

    def import_rows(self, rows: Iterable[Mapping[str, Any]], *, start_row: int = 1) -> ImportResult:
        result = ImportResult(source_system=self.source_system)
        seen_ids = set()
        for row_number, raw_row in enumerate(rows, start=start_row):
            try:
                canonical_row = self.canonicalize_row(raw_row)
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
                            detail=f"duplicate external_ticket_id: {external_id}",
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

    def _read_csv(self, path: Path) -> Tuple[List[Mapping[str, Any]], List[QuarantineRecord]]:
        rows: List[Mapping[str, Any]] = []
        errors: List[QuarantineRecord] = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t" if path.suffix.lower() == ".tsv" else ",")
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
            header_map = self._header_map(headers)
            if not header_map:
                return rows, [
                    QuarantineRecord(
                        self.source_system,
                        1,
                        "UNKNOWN_SCHEMA",
                        "no supported source columns found",
                        {str(i): value for i, value in enumerate(headers)},
                    )
                ]
            for row_number, values in enumerate(reader, start=2):
                if len(values) != len(headers):
                    errors.append(
                        QuarantineRecord(
                            self.source_system,
                            row_number,
                            "BAD_CSV_STRUCTURE",
                            f"expected {len(headers)} columns, got {len(values)}",
                            {str(i): value for i, value in enumerate(values)},
                        )
                    )
                    continue
                rows.append(dict(zip(headers, values)))
        return rows, errors

    def _read_json(self, path: Path) -> Tuple[List[Mapping[str, Any]], List[QuarantineRecord]]:
        rows: List[Mapping[str, Any]] = []
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
        except json.JSONDecodeError:
            values = []
            for row_number, line in enumerate(text.splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    values.append(json.loads(line))
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
        for row_number, value in enumerate(values, start=1):
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
                rows.append(value)
        return rows, errors

    def _read_xlsx(self, path: Path) -> Tuple[List[Mapping[str, Any]], List[QuarantineRecord]]:
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
                matrix: List[List[str]] = []
                for row in root.findall(".//main:sheetData/main:row", namespace):
                    values: List[str] = []
                    for cell in row.findall("main:c", namespace):
                        cell_type = cell.attrib.get("t")
                        value = cell.find("main:v", namespace)
                        inline = cell.find("main:is/main:t", namespace)
                        text = inline.text if inline is not None else (value.text if value is not None else "")
                        if cell_type == "s" and text:
                            text = shared[int(text)] if int(text) < len(shared) else ""
                        values.append(text or "")
                    matrix.append(values)
                if not matrix:
                    return [], [QuarantineRecord(self.source_system, 1, "BAD_CSV_STRUCTURE", "XLSX has no rows", {})]
                headers = matrix[0]
                if not headers or any(not str(header).strip() for header in headers):
                    return [], [QuarantineRecord(self.source_system, 1, "BAD_CSV_STRUCTURE", "XLSX contains an empty header", {str(i): value for i, value in enumerate(headers)})]
                rows: List[Mapping[str, Any]] = []
                for row_number, values in enumerate(matrix[1:], start=2):
                    if len(values) != len(headers):
                        errors.append(QuarantineRecord(self.source_system, row_number, "BAD_CSV_STRUCTURE", f"expected {len(headers)} columns, got {len(values)}", {str(i): value for i, value in enumerate(values)}))
                        continue
                    rows.append(dict(zip(headers, values)))
                return rows, errors
        except (OSError, zipfile.BadZipFile, ET.ParseError, ValueError, IndexError) as error:
            return [], [QuarantineRecord(self.source_system, 1, "UNKNOWN_SCHEMA", f"invalid XLSX: {error}", {"path": str(path)})]

    def import_file(self, path: Path | str) -> ImportResult:
        """Import CSV, JSON array or JSONL and retain every bad-row reason."""

        source_path = Path(path)
        suffix = source_path.suffix.lower()
        if suffix in {".json", ".jsonl", ".ndjson"}:
            rows, parse_errors = self._read_json(source_path)
        elif suffix in {".csv", ".tsv"}:
            rows, parse_errors = self._read_csv(source_path)
        elif suffix in {".xlsx", ".xlsm"}:
            rows, parse_errors = self._read_xlsx(source_path)
        else:
            return ImportResult(
                source_system=self.source_system,
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
        result = self.import_rows(rows, start_row=1 if suffix in {".json", ".jsonl", ".ndjson"} else 2)
        result.quarantine = parse_errors + result.quarantine
        return result


__all__ = ["ImportResult", "SourceImporter"]
