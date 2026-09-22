-- Every taxonomy label must have an approved operator template before it can
-- be shown in the assist/response workflow.  Keep this as a data migration so
-- template copy can be reviewed and versioned independently of the API.

INSERT INTO response_templates (template_key, language, topic_id, service_id, body, approved, version)
SELECT 'default-' || topic_id, language, topic_id, service_id, body, TRUE, 1
FROM (VALUES
    ('RU', 'landscaping', 'service_roads', 'Обращение по благоустройству и озеленению зарегистрировано и направлено в городскую инфраструктуру.'),
    ('KZ', 'landscaping', 'service_roads', 'Көріктендіру және көгалдандыру туралы өтінішіңіз тіркеліп, қалалық инфрақұрылым қызметіне жіберілді.'),
    ('RU', 'unknown', 'service_other', 'Обращение зарегистрировано и будет направлено ответственному специалисту после уточнения темы.'),
    ('KZ', 'unknown', 'service_other', 'Өтінішіңіз тіркелді және тақырыбы нақтыланғаннан кейін жауапты маманға жіберіледі.')
) AS templates(language, topic_id, service_id, body)
WHERE NOT EXISTS (
    SELECT 1
    FROM response_templates existing
    WHERE existing.template_key = 'default-' || templates.topic_id
      AND existing.language = templates.language
      AND existing.version = 1
);
