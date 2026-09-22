-- Normalize the seeded demo defaults without rewriting earlier migrations.
-- Migration 004 is already part of the migration history; these rows are
-- intentionally downgraded until authoritative 109 rules/templates arrive.

UPDATE routing_rules
SET source = 'MANUAL',
    reason = 'Демонстрационное сопоставление; официальные правила 109 не предоставлены',
    precedence = 100
WHERE source = 'OFFICIAL'
  AND reason = 'Официальное соответствие темы ответственной службе';

UPDATE priority_rules
SET source = 'MANUAL',
    reason = 'Демонстрационное правило приоритета; официальные правила 109 не предоставлены',
    precedence = 100
WHERE source = 'OFFICIAL'
  AND topic_id IN (
      'water_supply', 'wastewater', 'electricity', 'street_lighting',
      'heating', 'gas_supply', 'roads', 'public_transport',
      'healthcare', 'veterinary', 'environment', 'education',
      'telecom', 'buildings'
  );

UPDATE response_templates
SET approved = FALSE
WHERE template_key LIKE 'default-%'
  AND version = 1;
