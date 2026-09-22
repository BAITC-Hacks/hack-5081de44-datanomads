-- P0 operational rules and lifecycle state.
-- Rules are data, not Rust/React constants, so a later model replacement does
-- not change routing, priorities or approved operator copy.

CREATE TABLE IF NOT EXISTS routing_rules (
    id BIGSERIAL PRIMARY KEY,
    topic_id TEXT REFERENCES topics(id),
    region_id TEXT REFERENCES regions(id),
    service_id TEXT NOT NULL REFERENCES services(id),
    source TEXT NOT NULL CHECK (source IN ('OFFICIAL', 'LABEL_HISTORY', 'MANUAL')),
    reason TEXT NOT NULL,
    precedence INTEGER NOT NULL DEFAULT 100,
    version INTEGER NOT NULL DEFAULT 1,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_routing_rules_lookup
    ON routing_rules (topic_id, region_id, active, precedence);

CREATE TABLE IF NOT EXISTS priority_rules (
    id BIGSERIAL PRIMARY KEY,
    topic_id TEXT REFERENCES topics(id),
    region_id TEXT REFERENCES regions(id),
    priority TEXT NOT NULL CHECK (priority IN ('low', 'medium', 'high', 'critical')),
    source TEXT NOT NULL CHECK (source IN ('OFFICIAL', 'LABEL_HISTORY', 'MANUAL')),
    reason TEXT NOT NULL,
    precedence INTEGER NOT NULL DEFAULT 100,
    version INTEGER NOT NULL DEFAULT 1,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_priority_rules_lookup
    ON priority_rules (topic_id, region_id, active, precedence);

ALTER TABLE alerts ADD COLUMN IF NOT EXISTS detail JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE alerts ADD COLUMN IF NOT EXISTS closed_at TIMESTAMPTZ;
ALTER TABLE alerts ADD COLUMN IF NOT EXISTS closed_by TEXT;
ALTER TABLE learning_cycles ADD COLUMN IF NOT EXISTS decision_note TEXT;
ALTER TABLE model_versions DROP CONSTRAINT IF EXISTS model_versions_status_check;
ALTER TABLE model_versions ADD CONSTRAINT model_versions_status_check CHECK (status IN ('CANDIDATE', 'SHADOW', 'PRODUCTION', 'REJECTED', 'ARCHIVED'));

CREATE INDEX IF NOT EXISTS idx_tickets_external_id
    ON tickets (external_ticket_id);

-- Official service catalog.  Existing service_other remains the explicit
-- fallback when a source label cannot be mapped.
INSERT INTO services (id, name_ru, name_kk) VALUES
    ('service_water', 'Водоканал', 'Су арнасы'),
    ('service_wastewater', 'Канализационная служба', 'Кәріз қызметі'),
    ('service_electricity', 'Электросети', 'Электр желілері'),
    ('service_lighting', 'Городское освещение', 'Қалалық жарықтандыру'),
    ('service_heating', 'Теплосети', 'Жылу желілері'),
    ('service_gas', 'Газовая служба', 'Газ қызметі'),
    ('service_roads', 'Городская инфраструктура', 'Қалалық инфрақұрылым'),
    ('service_transport', 'Управление транспорта', 'Көлік басқармасы'),
    ('service_waste', 'Санитарная очистка', 'Санитарлық тазалық'),
    ('service_buildings', 'Жилищная инспекция', 'Тұрғын үй инспекциясы'),
    ('service_healthcare', 'Управление здравоохранения', 'Денсаулық сақтау басқармасы'),
    ('service_veterinary', 'Ветеринарная служба', 'Ветеринариялық қызмет'),
    ('service_environment', 'Управление экологии', 'Экология басқармасы'),
    ('service_education', 'Управление образования', 'Білім басқармасы'),
    ('service_telecom', 'Цифровой акимат', 'Цифрлық әкімдік')
ON CONFLICT (id) DO UPDATE SET name_ru = EXCLUDED.name_ru, name_kk = EXCLUDED.name_kk;

INSERT INTO routing_rules (topic_id, service_id, source, reason, precedence)
SELECT topic_id, service_id, 'OFFICIAL', 'Официальное соответствие темы ответственной службе', 10
FROM (VALUES
    ('water_supply', 'service_water'),
    ('wastewater', 'service_wastewater'),
    ('electricity', 'service_electricity'),
    ('street_lighting', 'service_lighting'),
    ('heating', 'service_heating'),
    ('gas_supply', 'service_gas'),
    ('roads', 'service_roads'),
    ('public_transport', 'service_transport'),
    ('waste_management', 'service_waste'),
    ('landscaping', 'service_roads'),
    ('buildings', 'service_buildings'),
    ('healthcare', 'service_healthcare'),
    ('veterinary', 'service_veterinary'),
    ('environment', 'service_environment'),
    ('education', 'service_education'),
    ('telecom', 'service_telecom')
) AS rules(topic_id, service_id)
WHERE NOT EXISTS (
    SELECT 1 FROM routing_rules existing
    WHERE existing.topic_id = rules.topic_id
      AND existing.region_id IS NULL
      AND existing.source = 'OFFICIAL'
      AND existing.active
);

INSERT INTO priority_rules (topic_id, priority, source, reason, precedence)
SELECT topic_id, priority, 'OFFICIAL', reason, 10
FROM (VALUES
    ('water_supply', 'high', 'Перебой водоснабжения требует ускоренной обработки'),
    ('wastewater', 'high', 'Аварийная канализация требует ускоренной обработки'),
    ('electricity', 'high', 'Перебой электроснабжения требует ускоренной обработки'),
    ('street_lighting', 'high', 'Наружное освещение влияет на безопасность'),
    ('heating', 'high', 'Отопление требует ускоренной обработки'),
    ('gas_supply', 'critical', 'Газовая безопасность требует критического приоритета'),
    ('roads', 'medium', 'Дорожные обращения обрабатываются в плановом порядке'),
    ('public_transport', 'medium', 'Транспортные обращения обрабатываются в плановом порядке'),
    ('healthcare', 'high', 'Здравоохранение требует ускоренной обработки'),
    ('veterinary', 'medium', 'Ветеринарные обращения обрабатываются в плановом порядке'),
    ('environment', 'medium', 'Экологические обращения требуют контроля'),
    ('education', 'medium', 'Образовательные обращения обрабатываются в плановом порядке'),
    ('telecom', 'medium', 'Цифровые обращения обрабатываются в плановом порядке'),
    ('buildings', 'medium', 'Здания и сооружения требуют плановой проверки')
) AS rules(topic_id, priority, reason)
WHERE NOT EXISTS (
    SELECT 1 FROM priority_rules existing
    WHERE existing.topic_id = rules.topic_id
      AND existing.region_id IS NULL
      AND existing.source = 'OFFICIAL'
      AND existing.active
);

INSERT INTO response_templates (template_key, language, topic_id, service_id, body, approved, version)
SELECT 'default-' || topic_id, language, topic_id, service_id, body, TRUE, 1
FROM (VALUES
    ('RU', 'water_supply', 'service_water', 'Ваше обращение по водоснабжению зарегистрировано и направлено в Водоканал.'),
    ('KZ', 'water_supply', 'service_water', 'Сумен жабдықтау туралы өтінішіңіз тіркеліп, Су арнасына жіберілді.'),
    ('RU', 'wastewater', 'service_wastewater', 'Обращение по водоотведению зарегистрировано и направлено в канализационную службу.'),
    ('KZ', 'wastewater', 'service_wastewater', 'Су бұру туралы өтінішіңіз тіркеліп, тиісті қызметке жіберілді.'),
    ('RU', 'electricity', 'service_electricity', 'Обращение по электроснабжению зарегистрировано и направлено в электросети.'),
    ('KZ', 'electricity', 'service_electricity', 'Электрмен жабдықтау туралы өтінішіңіз тіркеліп, электр желілеріне жіберілді.'),
    ('RU', 'street_lighting', 'service_lighting', 'Обращение по городскому освещению зарегистрировано и направлено ответственным специалистам.'),
    ('KZ', 'street_lighting', 'service_lighting', 'Қалалық жарықтандыру туралы өтінішіңіз тіркеліп, жауапты мамандарға жіберілді.'),
    ('RU', 'heating', 'service_heating', 'Обращение по отоплению зарегистрировано и направлено в теплосети.'),
    ('KZ', 'heating', 'service_heating', 'Жылыту туралы өтінішіңіз тіркеліп, жылу желілеріне жіберілді.'),
    ('RU', 'gas_supply', 'service_gas', 'Обращение по газоснабжению зарегистрировано и направлено в газовую службу.'),
    ('KZ', 'gas_supply', 'service_gas', 'Газбен жабдықтау туралы өтінішіңіз тіркеліп, газ қызметіне жіберілді.'),
    ('RU', 'roads', 'service_roads', 'Обращение по дорогам зарегистрировано и направлено в городскую инфраструктуру.'),
    ('KZ', 'roads', 'service_roads', 'Жолдар туралы өтінішіңіз тіркеліп, қалалық инфрақұрылым қызметіне жіберілді.'),
    ('RU', 'public_transport', 'service_transport', 'Обращение по общественному транспорту зарегистрировано и направлено в управление транспорта.'),
    ('KZ', 'public_transport', 'service_transport', 'Қоғамдық көлік туралы өтінішіңіз тіркеліп, көлік басқармасына жіберілді.'),
    ('RU', 'waste_management', 'service_waste', 'Обращение по санитарной очистке зарегистрировано и направлено ответственным специалистам.'),
    ('KZ', 'waste_management', 'service_waste', 'Тазалық туралы өтінішіңіз тіркеліп, жауапты мамандарға жіберілді.'),
    ('RU', 'buildings', 'service_buildings', 'Обращение по зданиям зарегистрировано и направлено в жилищную инспекцию.'),
    ('KZ', 'buildings', 'service_buildings', 'Ғимараттар туралы өтінішіңіз тіркеліп, тұрғын үй инспекциясына жіберілді.'),
    ('RU', 'healthcare', 'service_healthcare', 'Обращение по здравоохранению зарегистрировано и направлено в управление здравоохранения.'),
    ('KZ', 'healthcare', 'service_healthcare', 'Денсаулық сақтау туралы өтінішіңіз тіркеліп, денсаулық сақтау басқармасына жіберілді.'),
    ('RU', 'veterinary', 'service_veterinary', 'Обращение по ветеринарии зарегистрировано и направлено в ветеринарную службу.'),
    ('KZ', 'veterinary', 'service_veterinary', 'Ветеринария туралы өтінішіңіз тіркеліп, ветеринариялық қызметке жіберілді.'),
    ('RU', 'environment', 'service_environment', 'Обращение по экологии зарегистрировано и направлено в управление экологии.'),
    ('KZ', 'environment', 'service_environment', 'Экология туралы өтінішіңіз тіркеліп, экология басқармасына жіберілді.'),
    ('RU', 'education', 'service_education', 'Обращение по образованию зарегистрировано и направлено в управление образования.'),
    ('KZ', 'education', 'service_education', 'Білім беру туралы өтінішіңіз тіркеліп, білім басқармасына жіберілді.'),
    ('RU', 'telecom', 'service_telecom', 'Обращение по связи зарегистрировано и направлено в цифровой акимат.'),
    ('KZ', 'telecom', 'service_telecom', 'Байланыс туралы өтінішіңіз тіркеліп, цифрлық әкімдікке жіберілді.')
) AS templates(language, topic_id, service_id, body)
WHERE NOT EXISTS (
    SELECT 1 FROM response_templates existing
    WHERE existing.template_key = 'default-' || templates.topic_id
      AND existing.language = templates.language
      AND existing.version = 1
);
