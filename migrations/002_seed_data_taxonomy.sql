-- Initial taxonomy.  These are candidate labels, not a claim that every
-- source has all labels; unmapped source directions remain topic_id=unknown.

INSERT INTO regions (id, name_ru, name_kk, name_en) VALUES
('KZ-ABAY', 'Абайская область', 'Абай облысы', 'Abai Region'),
('KZ-AKMOLA', 'Акмолинская область', 'Ақмола облысы', 'Akmola Region'),
('KZ-AKTOBE', 'Актюбинская область', 'Ақтөбе облысы', 'Aktobe Region'),
('KZ-ALMATY-REGION', 'Алматинская область', 'Алматы облысы', 'Almaty Region'),
('KZ-ATYRAU', 'Атырауская область', 'Атырау облысы', 'Atyrau Region'),
('KZ-EAST-KAZAKHSTAN', 'Восточно-Казахстанская область', 'Шығыс Қазақстан облысы', 'East Kazakhstan Region'),
('KZ-ZHAMBYL', 'Жамбылская область', 'Жамбыл облысы', 'Zhambyl Region'),
('KZ-ZHETISU', 'Область Жетісу', 'Жетісу облысы', 'Zhetisu Region'),
('KZ-WEST-KAZAKHSTAN', 'Западно-Казахстанская область', 'Батыс Қазақстан облысы', 'West Kazakhstan Region'),
('KZ-KARAGANDA', 'Карагандинская область', 'Қарағанды облысы', 'Karaganda Region'),
('KZ-KOSTANAY', 'Костанайская область', 'Қостанай облысы', 'Kostanay Region'),
('KZ-KYZYLORDA', 'Кызылординская область', 'Қызылорда облысы', 'Kyzylorda Region'),
('KZ-MANGYSTAU', 'Мангистауская область', 'Маңғыстау облысы', 'Mangystau Region'),
('KZ-PAVLODAR', 'Павлодарская область', 'Павлодар облысы', 'Pavlodar Region'),
('KZ-NORTH-KAZAKHSTAN', 'Северо-Казахстанская область', 'Солтүстік Қазақстан облысы', 'North Kazakhstan Region'),
('KZ-TURKESTAN', 'Туркестанская область', 'Түркістан облысы', 'Turkistan Region'),
('KZ-ULYTAU', 'Улытауская область', 'Ұлытау облысы', 'Ulytau Region'),
('KZ-ASTANA', 'город Астана', 'Астана қаласы', 'Astana City'),
('KZ-ALMATY', 'город Алматы', 'Алматы қаласы', 'Almaty City'),
('KZ-SHYMKENT', 'город Шымкент', 'Шымкент қаласы', 'Shymkent City')
ON CONFLICT (id) DO UPDATE SET name_ru = EXCLUDED.name_ru, name_kk = EXCLUDED.name_kk, name_en = EXCLUDED.name_en;

INSERT INTO topics (id, name_ru, name_kk) VALUES
('unknown', 'Не определено', 'Анықталмаған'),
('water_supply', 'Водоснабжение', 'Сумен жабдықтау'),
('wastewater', 'Канализация и водоотведение', 'Кәріз және су бұру'),
('electricity', 'Электроснабжение', 'Электрмен жабдықтау'),
('street_lighting', 'Наружное освещение', 'Көшені жарықтандыру'),
('heating', 'Теплоснабжение и отопление', 'Жылумен жабдықтау және жылыту'),
('gas_supply', 'Газоснабжение', 'Газбен жабдықтау'),
('roads', 'Дороги и дорожная инфраструктура', 'Жолдар және жол инфрақұрылымы'),
('public_transport', 'Общественный транспорт', 'Қоғамдық көлік'),
('waste_management', 'ТБО и санитарная очистка', 'Қалдықтарды басқару және тазалық'),
('landscaping', 'Благоустройство и озеленение', 'Көріктендіру және көгалдандыру'),
('buildings', 'Здания, сооружения и лифты', 'Ғимараттар, құрылыстар және лифтілер'),
('healthcare', 'Здравоохранение', 'Денсаулық сақтау'),
('veterinary', 'Ветеринария и безнадзорные животные', 'Ветеринария және қараусыз жануарлар'),
('environment', 'Экология', 'Экология'),
('education', 'Образование', 'Білім беру'),
('telecom', 'Связь и интернет', 'Байланыс және интернет')
ON CONFLICT (id) DO UPDATE SET name_ru = EXCLUDED.name_ru, name_kk = EXCLUDED.name_kk;

INSERT INTO services (id, name_ru, name_kk) VALUES
('service_other', 'Другая служба', 'Басқа қызмет')
ON CONFLICT (id) DO UPDATE SET name_ru = EXCLUDED.name_ru, name_kk = EXCLUDED.name_kk;
