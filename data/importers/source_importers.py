"""Small source-specific adapters.

Each adapter owns aliases for the export it understands.  The normalization
and privacy rules remain shared so source differences cannot bypass the safe
UnifiedTicket boundary.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from .base import SourceImporter


class IKOMEK109Importer(SourceImporter):
    source_system = "ikomek109"
    field_aliases = {
        **SourceImporter.field_aliases,
        "external_ticket_id": ("ticket_id", "id", "requestNumber", "номер обращения", "external_ticket_id"),
        "created_at": ("created_at", "createdDate", "dateCreated", "дата создания"),
        "original_text": ("original_text", "appealText", "description", "обращение"),
        "topic_raw": ("topic_raw", "direction", "serviceDirection", "направление"),
        "service_raw": ("service_raw", "executor", "responsibleOrganization", "исполнитель"),
    }


class ASKOMEK109Importer(SourceImporter):
    source_system = "as_komek109"
    csv_delimiter = ";"
    field_aliases = {
        **SourceImporter.field_aliases,
        "external_ticket_id": ("external_ticket_id", "nomer_obrasheniya", "номер обращения", "id"),
        "created_at": ("created_at", "data_sozdaniya", "дата создания"),
        "original_text": ("original_text", "opisanie", "текст обращения", "обращение"),
        "topic_raw": ("topic_raw", "vid_rabot", "категория", "направление"),
    }


class AIKEYImporter(SourceImporter):
    source_system = "aikey"
    field_aliases = {
        **SourceImporter.field_aliases,
        "external_ticket_id": ("external_ticket_id", "appealId", "appeal_id", "id"),
        "created_at": ("created_at", "registeredAt", "createdAt", "дата"),
        "original_text": ("original_text", "messageText", "message", "text"),
        "topic_raw": ("topic_raw", "categoryName", "category", "тема"),
    }


class OpenCityImporter(SourceImporter):
    source_system = "open_city"
    field_aliases = {
        **SourceImporter.field_aliases,
        "external_ticket_id": ("external_ticket_id", "request_id", "requestId", "id"),
        "created_at": ("created_at", "date_created", "created", "дата"),
        "original_text": ("original_text", "request_text", "description", "текст"),
        "region_id": ("region_id", "city", "region", "город"),
    }


class ESEPSUImporter(SourceImporter):
    source_system = "e_sep_su"
    field_aliases = {
        **SourceImporter.field_aliases,
        "external_ticket_id": ("external_ticket_id", "case_number", "caseNo", "id"),
        "created_at": ("created_at", "registration_date", "registered", "дата регистрации"),
        "original_text": ("original_text", "case_text", "description", "обращение"),
    }


class EKC109Importer(SourceImporter):
    source_system = "ekc109"
    field_aliases = {
        **SourceImporter.field_aliases,
        "external_ticket_id": ("external_ticket_id", "card_id", "cardId", "id"),
        "created_at": ("created_at", "opened_at", "openDate", "дата открытия"),
        "original_text": ("original_text", "complaint", "complaintText", "текст"),
    }


class RDJardemImporter(SourceImporter):
    source_system = "rdjardem3"
    field_aliases = {
        **SourceImporter.field_aliases,
        "external_ticket_id": ("external_ticket_id", "ticketNo", "ticket_number", "id"),
        "created_at": ("created_at", "created_on", "createdAt", "дата"),
        "original_text": ("original_text", "appeal", "appeal_body", "обращение"),
    }


_IMPORTERS = {
    "ikomek109": IKOMEK109Importer,
    "as_komek109": ASKOMEK109Importer,
    "aikey": AIKEYImporter,
    "open_city": OpenCityImporter,
    "e_sep_su": ESEPSUImporter,
    "ekc109": EKC109Importer,
    "rdjardem3": RDJardemImporter,
}


def get_importer(source_system: str) -> SourceImporter:
    from data.schemas.taxonomy import canonical_source_system

    canonical = canonical_source_system(source_system) or source_system
    importer_type = _IMPORTERS.get(canonical)
    if importer_type is None:
        raise ValueError(f"unsupported source_system: {source_system}")
    return importer_type()


__all__ = [
    "IKOMEK109Importer",
    "ASKOMEK109Importer",
    "AIKEYImporter",
    "OpenCityImporter",
    "ESEPSUImporter",
    "EKC109Importer",
    "RDJardemImporter",
    "get_importer",
]
