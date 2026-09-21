"""Stable region, source and topic identifiers for the data contract.

The names are deliberately kept separate from source labels.  A source may
continue to send a local spelling in ``topic_raw`` while the normalized row
uses one of the canonical identifiers below.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, Iterable, List, Mapping, Optional, Tuple


def _fold(value: object) -> str:
    """Return a comparison form that works for RU/KZ labels."""

    text = unicodedata.normalize("NFKC", str(value or "")).strip().lower()
    text = text.replace("ё", "е")
    return re.sub(r"[^\w]+", " ", text, flags=re.UNICODE).strip()


CANONICAL_SOURCE_SYSTEMS: Tuple[str, ...] = (
    "ikomek109",
    "as_komek109",
    "aikey",
    "open_city",
    "e_sep_su",
    "ekc109",
    "rdjardem3",
)

SOURCE_DISPLAY_NAMES: Mapping[str, str] = {
    "ikomek109": "iKOMEK109",
    "as_komek109": "АС Комек 109",
    "aikey": "AIKEY",
    "open_city": "Открытый город",
    "e_sep_su": "E-SEP.SU",
    "ekc109": "ЕКЦ-109",
    "rdjardem3": "RDJardem3.0",
}

_SOURCE_ALIASES: Dict[str, str] = {}
for _canonical, _aliases in {
    "ikomek109": ("ikomek109", "iKOMEK109", "i komek", "икоменк109"),
    "as_komek109": (
        "as_komek109",
        "as komek 109",
        "ас комек 109",
        "аскомек109",
    ),
    "aikey": ("aikey", "ai key", "айки", "айкей"),
    "open_city": ("open_city", "open city", "открытый город", "ашык кала"),
    "e_sep_su": ("e_sep_su", "e-sep.su", "e sep su", "есепсу"),
    "ekc109": ("ekc109", "екц-109", "екц 109", "екц109"),
    "rdjardem3": ("rdjardem3", "rdjardem3.0", "рджаrdем3", "рд жәрдем"),
}.items():
    for _alias in _aliases:
        _SOURCE_ALIASES[_fold(_alias)] = _canonical


def canonical_source_system(value: object) -> Optional[str]:
    """Resolve a source label, returning ``None`` for an unknown source."""

    folded = _fold(value)
    return _SOURCE_ALIASES.get(folded)


REGION_DEFINITIONS: Tuple[Mapping[str, str], ...] = (
    {
        "id": "KZ-ABAY",
        "name_ru": "Абайская область",
        "name_kk": "Абай облысы",
        "name_en": "Abai Region",
        "aliases": "абай абайская область абай облысы abay",
    },
    {
        "id": "KZ-AKMOLA",
        "name_ru": "Акмолинская область",
        "name_kk": "Ақмола облысы",
        "name_en": "Akmola Region",
        "aliases": "акмола акмолинская область ақмола облысы akmola",
    },
    {
        "id": "KZ-AKTOBE",
        "name_ru": "Актюбинская область",
        "name_kk": "Ақтөбе облысы",
        "name_en": "Aktobe Region",
        "aliases": "актобе актюбинская область ақтөбе облысы aktobe",
    },
    {
        "id": "KZ-ALMATY-REGION",
        "name_ru": "Алматинская область",
        "name_kk": "Алматы облысы",
        "name_en": "Almaty Region",
        "aliases": "алматинская область алматы облысы алматыская область almaty region",
    },
    {
        "id": "KZ-ATYRAU",
        "name_ru": "Атырауская область",
        "name_kk": "Атырау облысы",
        "name_en": "Atyrau Region",
        "aliases": "атырау атырауская область atyrau",
    },
    {
        "id": "KZ-EAST-KAZAKHSTAN",
        "name_ru": "Восточно-Казахстанская область",
        "name_kk": "Шығыс Қазақстан облысы",
        "name_en": "East Kazakhstan Region",
        "aliases": "вко восточно казахстанская область шығыс қазақстан east kazakhstan",
    },
    {
        "id": "KZ-ZHAMBYL",
        "name_ru": "Жамбылская область",
        "name_kk": "Жамбыл облысы",
        "name_en": "Zhambyl Region",
        "aliases": "жамбыл жамбылская область жамбыл облысы zhambyl",
    },
    {
        "id": "KZ-ZHETISU",
        "name_ru": "Область Жетісу",
        "name_kk": "Жетісу облысы",
        "name_en": "Zhetisu Region",
        "aliases": "жетису жетісу область жетісу облысы zhetisu",
    },
    {
        "id": "KZ-WEST-KAZAKHSTAN",
        "name_ru": "Западно-Казахстанская область",
        "name_kk": "Батыс Қазақстан облысы",
        "name_en": "West Kazakhstan Region",
        "aliases": "зко западно казахстанская область батыс қазақстан west kazakhstan",
    },
    {
        "id": "KZ-KARAGANDA",
        "name_ru": "Карагандинская область",
        "name_kk": "Қарағанды облысы",
        "name_en": "Karaganda Region",
        "aliases": "караганда карагандинская область қарағанды облысы karaganda",
    },
    {
        "id": "KZ-KOSTANAY",
        "name_ru": "Костанайская область",
        "name_kk": "Қостанай облысы",
        "name_en": "Kostanay Region",
        "aliases": "костанай костанайская область қостанай облысы kostanay",
    },
    {
        "id": "KZ-KYZYLORDA",
        "name_ru": "Кызылординская область",
        "name_kk": "Қызылорда облысы",
        "name_en": "Kyzylorda Region",
        "aliases": "кызылорда кызылординская область қызылорда облысы kyzylorda",
    },
    {
        "id": "KZ-MANGYSTAU",
        "name_ru": "Мангистауская область",
        "name_kk": "Маңғыстау облысы",
        "name_en": "Mangystau Region",
        "aliases": "мангистау мангистауская область маңғыстау облысы mangystau",
    },
    {
        "id": "KZ-PAVLODAR",
        "name_ru": "Павлодарская область",
        "name_kk": "Павлодар облысы",
        "name_en": "Pavlodar Region",
        "aliases": "павлодар павлодарская область павлодар облысы pavlodar",
    },
    {
        "id": "KZ-NORTH-KAZAKHSTAN",
        "name_ru": "Северо-Казахстанская область",
        "name_kk": "Солтүстік Қазақстан облысы",
        "name_en": "North Kazakhstan Region",
        "aliases": "северо казахстанская область солтүстік қазақстан sko north kazakhstan",
    },
    {
        "id": "KZ-TURKESTAN",
        "name_ru": "Туркестанская область",
        "name_kk": "Түркістан облысы",
        "name_en": "Turkistan Region",
        "aliases": "туркестан туркестанская область түркістан облысы turkistan",
    },
    {
        "id": "KZ-ULYTAU",
        "name_ru": "Улытауская область",
        "name_kk": "Ұлытау облысы",
        "name_en": "Ulytau Region",
        "aliases": "улытау улытауская область ұлытау облысы ulytau",
    },
    {
        "id": "KZ-ASTANA",
        "name_ru": "город Астана",
        "name_kk": "Астана қаласы",
        "name_en": "Astana City",
        "aliases": "астана город астана астана қаласы нур султан нур султан city astana",
    },
    {
        "id": "KZ-ALMATY",
        "name_ru": "город Алматы",
        "name_kk": "Алматы қаласы",
        "name_en": "Almaty City",
        "aliases": "город алматы алматы қаласы алматы city almaty",
    },
    {
        "id": "KZ-SH YMKENT".replace(" ", ""),
        "name_ru": "город Шымкент",
        "name_kk": "Шымкент қаласы",
        "name_en": "Shymkent City",
        "aliases": "шымкент город шымкент шымкент қаласы шимкент shymkent",
    },
)

_REGION_ALIASES: Dict[str, str] = {}
for _region in REGION_DEFINITIONS:
    _REGION_ALIASES[_fold(_region["id"])] = _region["id"]
    _REGION_ALIASES[_fold(_region["name_ru"])] = _region["id"]
    _REGION_ALIASES[_fold(_region["name_kk"])] = _region["id"]
    _REGION_ALIASES[_fold(_region["name_en"])] = _region["id"]
    for _alias in _region["aliases"].split():
        # The aliases string is also allowed to contain multi-word phrases;
        # register the complete phrase below and individual words are useful
        # for compact source exports.
        _REGION_ALIASES[_fold(_alias)] = _region["id"]
    _REGION_ALIASES[_fold(_region["aliases"])] = _region["id"]


def canonical_region_id(value: object) -> Optional[str]:
    """Resolve a region label to one of the 20 stable IDs."""

    folded = _fold(value)
    if folded in _REGION_ALIASES:
        return _REGION_ALIASES[folded]
    # Source exports frequently add a suffix such as "район" or "обл.".
    for alias, region_id in _REGION_ALIASES.items():
        if len(alias) > 4 and (alias in folded or folded in alias):
            return region_id
    return None


TOPIC_DEFINITIONS: Tuple[Mapping[str, str], ...] = (
    {
        "id": "water_supply",
        "name_ru": "Водоснабжение",
        "name_kk": "Сумен жабдықтау",
        "aliases": "водоснабжение вода водопровод сумен жабдықтау су құбыры",
    },
    {
        "id": "wastewater",
        "name_ru": "Канализация и водоотведение",
        "name_kk": "Кәріз және су бұру",
        "aliases": "канализация водоотведение кәріз су бұру",
    },
    {
        "id": "electricity",
        "name_ru": "Электроснабжение",
        "name_kk": "Электрмен жабдықтау",
        "aliases": "электричество электроснабжение свет электр электрмен жабдықтау",
    },
    {
        "id": "street_lighting",
        "name_ru": "Наружное освещение",
        "name_kk": "Көшені жарықтандыру",
        "aliases": "освещение фонарь уличный свет наружное освещение көшені жарықтандыру",
    },
    {
        "id": "heating",
        "name_ru": "Теплоснабжение и отопление",
        "name_kk": "Жылумен жабдықтау және жылыту",
        "aliases": "отопление теплоснабжение тепло батарея жылу жылыту",
    },
    {
        "id": "gas_supply",
        "name_ru": "Газоснабжение",
        "name_kk": "Газбен жабдықтау",
        "aliases": "газ газоснабжение утечка газа газбен жабдықтау",
    },
    {
        "id": "roads",
        "name_ru": "Дороги и дорожная инфраструктура",
        "name_kk": "Жолдар және жол инфрақұрылымы",
        "aliases": "дорога дороги яма асфальт дорожная инфраструктура жол шұңқыр",
    },
    {
        "id": "public_transport",
        "name_ru": "Общественный транспорт",
        "name_kk": "Қоғамдық көлік",
        "aliases": "автобус транспорт общественный маршрут аялдама қоғамдық көлік",
    },
    {
        "id": "waste_management",
        "name_ru": "ТБО и санитарная очистка",
        "name_kk": "Қалдықтарды басқару және тазалық",
        "aliases": "мусор отходы тбо контейнер санитарная очистка қоқыс қалдық",
    },
    {
        "id": "landscaping",
        "name_ru": "Благоустройство и озеленение",
        "name_kk": "Көріктендіру және көгалдандыру",
        "aliases": "благоустройство озеленение двор парк ағаш көріктендіру",
    },
    {
        "id": "buildings",
        "name_ru": "Здания, сооружения и лифты",
        "name_kk": "Ғимараттар, құрылыстар және лифтілер",
        "aliases": "здание дом лифт сооружение подъезд ғимарат лифт",
    },
    {
        "id": "healthcare",
        "name_ru": "Здравоохранение",
        "name_kk": "Денсаулық сақтау",
        "aliases": "медицина поликлиника больница врач здравоохранение денсаулық",
    },
    {
        "id": "veterinary",
        "name_ru": "Ветеринария и безнадзорные животные",
        "name_kk": "Ветеринария және қараусыз жануарлар",
        "aliases": "животные собака кошка ветеринария безнадзорные жануар ветеринария",
    },
    {
        "id": "environment",
        "name_ru": "Экология",
        "name_kk": "Экология",
        "aliases": "экология загрязнение выброс дым қоршаған орта",
    },
    {
        "id": "education",
        "name_ru": "Образование",
        "name_kk": "Білім беру",
        "aliases": "школа детский сад образование мектеп балабақша білім",
    },
    {
        "id": "telecom",
        "name_ru": "Связь и интернет",
        "name_kk": "Байланыс және интернет",
        "aliases": "интернет связь мобильная связь телефон байланыс",
    },
)

_TOPIC_ALIASES: Dict[str, str] = {}
for _topic in TOPIC_DEFINITIONS:
    _TOPIC_ALIASES[_fold(_topic["id"])] = _topic["id"]
    _TOPIC_ALIASES[_fold(_topic["name_ru"])] = _topic["id"]
    _TOPIC_ALIASES[_fold(_topic["name_kk"])] = _topic["id"]
    for _alias in _topic["aliases"].split():
        _TOPIC_ALIASES[_fold(_alias)] = _topic["id"]
    _TOPIC_ALIASES[_fold(_topic["aliases"])] = _topic["id"]


def canonical_topic_id(value: object) -> str:
    """Map a raw direction to a canonical topic or ``unknown``."""

    folded = _fold(value)
    if not folded:
        return "unknown"
    if folded in _TOPIC_ALIASES:
        return _TOPIC_ALIASES[folded]
    for alias, topic_id in _TOPIC_ALIASES.items():
        if len(alias) > 3 and (alias in folded or folded in alias):
            return topic_id
    return "unknown"


def canonical_language(value: object, text: str = "") -> str:
    """Normalize source language labels and deterministically infer RU/KZ."""

    folded = _fold(value)
    if folded in {"ru", "rus", "рус", "русский", "russian", "ru ru"}:
        return "RU"
    if folded in {"kz", "kk", "kaz", "қазақ", "қазақша", "казахский", "kazakh"}:
        return "KZ"
    sample = str(text or "").lower()
    if re.search(r"[әғқңөұүһі]", sample):
        return "KZ"
    if re.search(r"[а-яё]", sample):
        return "RU"
    return "UNKNOWN"


def canonical_priority(value: object) -> Optional[str]:
    folded = _fold(value)
    if not folded:
        return None
    if folded in {"critical", "критический", "аварийный", "срочно", "4", "критично"}:
        return "CRITICAL"
    if folded in {"high", "высокий", "высокая", "жоғары", "3"}:
        return "HIGH"
    if folded in {"medium", "normal", "средний", "обычный", "орташа", "2", "1"}:
        return "MEDIUM"
    if folded in {"low", "низкий", "низкая", "төмен", "0"}:
        return "LOW"
    return "OTHER"


def canonical_status(value: object) -> str:
    folded = _fold(value)
    if not folded:
        return "UNKNOWN"
    if folded in {"open", "new", "новое", "новый", "открыто", "ашық"}:
        return "OPEN"
    if folded in {"in progress", "в работе", "работа", "исполняется", "орындауда"}:
        return "IN_PROGRESS"
    if folded in {"resolved", "решено", "исполнено", "шешілді"}:
        return "RESOLVED"
    if folded in {"closed", "закрыто", "закрыт", "жабық"}:
        return "CLOSED"
    if folded in {"cancelled", "canceled", "отменено", "отклонено"}:
        return "CANCELLED"
    return "OTHER"


def topic_by_id(topic_id: str) -> Optional[Mapping[str, str]]:
    for topic in TOPIC_DEFINITIONS:
        if topic["id"] == topic_id:
            return topic
    return None


def region_by_id(region_id: str) -> Optional[Mapping[str, str]]:
    for region in REGION_DEFINITIONS:
        if region["id"] == region_id:
            return region
    return None


__all__ = [
    "CANONICAL_SOURCE_SYSTEMS",
    "SOURCE_DISPLAY_NAMES",
    "REGION_DEFINITIONS",
    "TOPIC_DEFINITIONS",
    "canonical_language",
    "canonical_priority",
    "canonical_region_id",
    "canonical_source_system",
    "canonical_status",
    "canonical_topic_id",
    "region_by_id",
    "topic_by_id",
]
