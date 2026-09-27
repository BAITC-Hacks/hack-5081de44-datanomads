# Routing и priority: статус ground truth

`scripts/data_audit.py --source <system> --synthetic|--real` теперь включает
`label_ground_truth` для `service_raw` и `priority`. Статус `UNSUITABLE`
означает synthetic source либо отсутствие значений в конкретной выгрузке.
`UNVERIFIED` означает, что значения есть, но их смысл не подтверждён.
Ни один из этих статусов не разрешает supervised training.

| Источник доказательства | Routing label | Priority label | Что известно |
| --- | --- | --- | --- |
| Synthetic exports семи source profiles | `UNSUITABLE` | `UNSUITABLE` | Данные вымышлены; поля проверяют import contract, а не решения операторов. |
| CSV 109 ВКО, описанный в [аудите](vko-109-data-audit.md) | `UNVERIFIED` | `UNVERIFIED` | `service` и `contractor` не подтверждены как фактический конечный исполнитель или первоначальное назначение. История переназначений и policy для priority не проверены. |
| CSV Алматинской области, описанный в [разборе taxonomy](almaty-2025-taxonomy-review.md) | `UNSUITABLE` для `service`; `UNVERIFIED` для `contractor` | `UNVERIFIED` | `service` обозначает тип вопроса, а не исполнителя. Семантика `contractor` и priority не подтверждена. |

Перед сменой статуса на `VERIFIED` владелец источника должен подтвердить
для routing, является ли поле первоначальным назначением или фактическим
конечным исполнителем, наличие истории переназначений и возможность получить
first-pass ground truth. Для priority нужны определение уровней и доказательство
того, что значения отражают подтверждённое решение, а не служебный default.
Решение фиксируется отдельно с reviewer, датой, версией источника и checksum.
До такой проверки service model и priority head не обучаются; demo MANUAL
mappings остаются только правилами прототипа.
