"""Stable taxonomy and model constants used by the demo service.

The production taxonomy is expected to be learned from the normalized 109
dataset.  These labels are deliberately explicit and versioned so a demo
client can rely on a stable contract while the real dataset is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Topic:
    topic_id: str
    name_ru: str
    name_kz: str
    keywords: tuple[str, ...]


TOPICS: tuple[Topic, ...] = (
    Topic(
        "water_supply",
        "Водоснабжение",
        "Сумен жабдықтау",
        ("вода", "вод", "воды", "водопровод", "водоснабж", "горяч", "ауызсу", "су құбыры", "су жоқ"),
    ),
    Topic(
        "wastewater",
        "Канализация и водоотведение",
        "Кәріз және су бұру",
        ("канализац", "сток", "водоотвед", "кәріз", "ағынды су"),
    ),
    Topic(
        "electricity",
        "Электроснабжение",
        "Электрмен жабдықтау",
        ("электр", "свет", "электроснабж", "жарық", "тоқ", "электр қуаты"),
    ),
    Topic(
        "street_lighting",
        "Наружное освещение",
        "Көшені жарықтандыру",
        ("фонарь", "освещен", "уличн свет", "лампа", "жарықтандыру", "көше", "көшеде", "көше шамы", "жарық жоқ"),
    ),
    Topic(
        "heating",
        "Теплоснабжение и отопление",
        "Жылумен жабдықтау",
        ("отоплен", "батаре", "теплоснабж", "тепло", "жылыту", "жылу"),
    ),
    Topic(
        "gas_supply",
        "Газоснабжение",
        "Газбен жабдықтау",
        ("газ", "газоснабж", "газ иісі", "газдың иісі"),
    ),
    Topic(
        "roads",
        "Дороги и дорожная инфраструктура",
        "Жолдар және жол инфрақұрылымы",
        ("дорог", "ям", "асфальт", "тротуар", "жол", "шұңқыр", "жаяу жүргінші"),
    ),
    Topic(
        "public_transport",
        "Общественный транспорт",
        "Қоғамдық көлік",
        ("автобус", "маршрут", "останов", "такси", "көлік", "аялдама"),
    ),
    Topic(
        "waste",
        "ТБО и санитарная очистка",
        "Қатты тұрмыстық қалдықтар",
        ("мусор", "тбо", "свалк", "контейнер", "қоқыс", "қалдық", "санитар"),
    ),
    Topic(
        "landscaping",
        "Благоустройство и озеленение",
        "Көгалдандыру және абаттандыру",
        ("благоустрой", "озелен", "дерев", "газон", "аула", "көгал", "абаттандыру"),
    ),
    Topic(
        "buildings",
        "Здания, сооружения и лифты",
        "Ғимараттар және лифтілер",
        ("дом", "подъезд", "крыша", "лифт", "здан", "үй", "шатыр"),
    ),
    Topic(
        "healthcare",
        "Здравоохранение",
        "Денсаулық сақтау",
        ("больниц", "поликлиник", "врач", "медицин", "денсаулық", "дәрігер"),
    ),
    Topic(
        "veterinary",
        "Ветеринария и безнадзорные животные",
        "Ветеринария және қараусыз жануарлар",
        ("собак", "кошк", "животн", "ветеринар", "ит", "мысық", "жануар"),
    ),
    Topic(
        "ecology",
        "Экология",
        "Экология",
        ("эколог", "выброс", "загрязн", "воздух", "қоршаған орта", "ластану"),
    ),
    Topic(
        "education",
        "Образование",
        "Білім беру",
        ("школ", "детсад", "образован", "училищ", "мектеп", "балабақша"),
    ),
    Topic(
        "telecom",
        "Связь и интернет",
        "Байланыс және интернет",
        ("интернет", "связь", "телефон", "мобильн", "байланыс", "ұялы"),
    ),
    Topic(
        "other",
        "Другое / требуется уточнение",
        "Басқа / нақтылау қажет",
        (),
    ),
)

TOPIC_BY_ID = {topic.topic_id: topic for topic in TOPICS}

MODEL_VERSIONS: dict[str, str] = {
    "classifier": "classifier-demo-2026-09-21-001",
    "embedder": "embedder-demo-2026-09-21-001",
    "forecast": "forecast-statsforecast-seasonal-naive-2026-09-24-001",
    "anomaly": "anomaly-robust-zscore-2026-09-21-001",
}

DEMO_IMPLEMENTATIONS: dict[str, str] = {
    "classifier": "DETERMINISTIC_KEYWORD_BASELINE",
    "embedder": "DETERMINISTIC_HASHING_EMBEDDING",
    "forecast": "SEASONAL_NAIVE",
    "anomaly": "ROLLING_MEDIAN_MAD",
}

SUPPORTED_LANGUAGES = ("RU", "KZ", "MIXED", "UNKNOWN")
