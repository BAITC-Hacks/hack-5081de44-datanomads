import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { machineKazakh } from './i18n/kk'
import { reviewedKazakh } from './i18n/kkReviewed'

export type Locale = 'ru' | 'kk'
export type Theme = 'light' | 'dark'

const LOCALE_KEY = 'pulse109.locale'
const THEME_KEY = 'pulse109.theme'

const kazakh: Record<string, string> = {
  ...machineKazakh,
  ...reviewedKazakh,
  'Оператор': 'Оператор',
  'Руководитель': 'Басшы',
  'Проверяющий ML': 'ML тексерушісі',
  'Рабочее место': 'Жұмыс орны',
  'Центр ситуации': 'Ахуал орталығы',
  'Администрирование': 'Әкімшілік',
  'Входящие обращения': 'Келіп түскен өтініштер',
  'Обзор потока': 'Ағынға шолу',
  'Регионы': 'Өңірлер',
  'Темы обращений': 'Өтініш тақырыптары',
  'Временная динамика': 'Уақыт бойынша өзгеріс',
  'Оповещения': 'Хабарламалар',
  'Прогноз': 'Болжам',
  'Отчёты': 'Есептер',
  'Цикл обучения': 'Оқыту циклі',
  'Статус моделей': 'Модельдердің күйі',
  'Журнал аудита': 'Аудит журналы',
  'Входящие': 'Келіп түскендер',
  'Обзор': 'Шолу',
  'Темы': 'Тақырыптар',
  'Контроль моделей': 'Модельдерді бақылау',
  'Что происходит с обращениями прямо сейчас': 'Өтініштермен қазір не болып жатыр',
  'Решения оператора, которым можно доверять': 'Операторға арналған тексерілетін шешімдер',
  'Где меняется нагрузка и появляется риск': 'Жүктеме мен тәуекел өзгеретін өңірлер',
  'Распределение спроса по таксономии Pulse': 'Сұраныстың Pulse таксономиясы бойынша таралуы',
  'Ритм обращений за последние 30 дней': 'Соңғы 30 күндегі өтініштер ағыны',
  'Сигналы, которые требуют внимания команды': 'Команда назарын қажет ететін сигналдар',
  'Ожидаемая нагрузка на ближайшие дни': 'Алдағы күндердегі күтілетін жүктеме',
  'Срезы для руководителей и рабочих встреч': 'Басшылар мен жұмыс кездесулеріне арналған кесінділер',
  'Как обратная связь становится улучшением модели': 'Кері байланыс модельді қалай жақсартады',
  'Версии, метрики и решение о продвижении': 'Нұсқалар, метрикалар және енгізу шешімі',
  'Кто, когда и с каким объектом выполнял действие': 'Кім, қашан және қандай нысанмен әрекет жасады',
  'Открыть меню': 'Мәзірді ашу',
  'Закрыть меню': 'Мәзірді жабу',
  'Основная навигация': 'Негізгі навигация',
  'Профиль': 'Профиль',
  'Настройки профиля': 'Профиль баптаулары',
  'Язык': 'Тіл',
  'Русский': 'Орысша',
  'Қазақша': 'Қазақша',
  'Оформление': 'Безендіру',
  'Светлая тема': 'Жарық тақырып',
  'Тёмная тема': 'Қараңғы тақырып',
  'Выйти': 'Шығу',
  'API подключён': 'API қосылған',
  'API недоступен': 'API қолжетімсіз',
  'Демо-данные': 'Демо деректер',
  'Демо-данные · не являются операционной статистикой заказчика': 'Демо деректер · тапсырыс берушінің операциялық статистикасы емес',
  'Core API недоступен': 'Core API қолжетімсіз',
  'Период': 'Кезең',
  '7 дней': '7 күн',
  '30 дней': '30 күн',
  '90 дней': '90 күн',
  'Регион': 'Өңір',
  'Все регионы': 'Барлық өңірлер',
  'Тема': 'Тақырып',
  'Все темы': 'Барлық тақырыптар',
  'Служба': 'Қызмет',
  'Все службы': 'Барлық қызметтер',
  'Статус': 'Күй',
  'Все статусы': 'Барлық күйлер',
  'Район': 'Аудан',
  'Все районы': 'Барлық аудандар',
  'Канал': 'Арна',
  'Все каналы': 'Барлық арналар',
  'Сбросить': 'Қалпына келтіру',
  'Фильтры аналитики': 'Талдау сүзгілері',
  'Закрыть уведомление': 'Хабарламаны жабу',
  'Загрузка данных': 'Деректер жүктелуде',
  'Повторить': 'Қайталау',
  'Выберите тестовый аккаунт': 'Сынақ аккаунтын таңдаңыз',
  'Для входа пароль не нужен. Выберите роль, чтобы увидеть соответствующее рабочее место.': 'Кіру үшін құпиясөз қажет емес. Тиісті жұмыс орнын көру үшін рөлді таңдаңыз.',
  'Работа с обращениями, рекомендациями и решениями': 'Өтініштермен, ұсыныстармен және шешімдермен жұмыс',
  'Аналитика, сигналы, прогнозы и отчёты': 'Талдау, сигналдар, болжамдар және есептер',
  'Циклы обучения, оценка и выпуск моделей': 'Оқыту циклдері, бағалау және модельдерді шығару',
  'Открыть рабочее место': 'Жұмыс орнын ашу',
  'Тестовый режим · действия записываются от выбранной роли': 'Сынақ режимі · әрекеттер таңдалған рөл атынан жазылады',
}

let activeLocale: Locale = 'ru'

export function localeTag(): 'ru-RU' | 'kk-KZ' {
  return activeLocale === 'kk' ? 'kk-KZ' : 'ru-RU'
}

export function formatUiDateTime(value: string): string {
  const date = new Date(value)
  return Number.isNaN(date.valueOf()) ? value : new Intl.DateTimeFormat(localeTag(), { dateStyle: 'short', timeStyle: 'short' }).format(date)
}

export function translateUi(text: string): string {
  if (activeLocale !== 'kk') return text
  const spikeTitle = /^Всплеск обращений: (.+)$/.exec(text)
  if (spikeTitle) return `${kazakh['Всплеск обращений']}: ${translateUi(spikeTitle[1])}`
  const monitoringEnd = /^Период наблюдения завершится (.+)\. Итог будет определён по полным интервалам после окончания срока\.$/.exec(text)
  if (monitoringEnd) return `Бақылау кезеңі ${monitoringEnd[1]} аяқталады. Нәтиже мерзім біткеннен кейінгі толық аралықтар бойынша анықталады.`
  const robustThreshold = /^Robust z (.+) превысил порог (.+)\.$/.exec(text)
  if (robustThreshold) return `Robust z ${robustThreshold[1]} ${robustThreshold[2]} шегінен асты.`
  const ratioThreshold = /^Количество выше baseline в (.+)× \(порог (.+)×\)\.$/.exec(text)
  if (ratioThreshold) return `Өтініш саны базалық деңгейден ${ratioThreshold[1]} есе жоғары (шек: ${ratioThreshold[2]} есе).`
  const relativePeak = /^(.+)× от среднего$/.exec(text)
  if (relativePeak) return `орташа мәннен ${relativePeak[1]} есе жоғары`
  return kazakh[text] ?? text
}

interface UiSettings {
  locale: Locale
  setLocale: (locale: Locale) => void
  theme: Theme
  setTheme: (theme: Theme) => void
  t: (text: string) => string
  number: (value: number) => string
  dateTime: (value: string | Date) => string
}

const UiSettingsContext = createContext<UiSettings | null>(null)

function storedLocale(): Locale {
  return window.localStorage.getItem(LOCALE_KEY) === 'kk' ? 'kk' : 'ru'
}

function storedTheme(): Theme {
  const saved = window.localStorage.getItem(THEME_KEY)
  if (saved === 'dark' || saved === 'light') return saved
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

export function UiSettingsProvider({ children }: { children: ReactNode }) {
  const [locale, setLocale] = useState<Locale>(storedLocale)
  const [theme, setTheme] = useState<Theme>(storedTheme)
  activeLocale = locale

  useEffect(() => {
    window.localStorage.setItem(LOCALE_KEY, locale)
    document.documentElement.lang = locale === 'kk' ? 'kk' : 'ru'
    document.title = locale === 'kk' ? 'Pulse 109 · Ахуал орталығы' : 'Pulse 109 · Ситуация'
    document.querySelector('meta[name="description"]')?.setAttribute(
      'content',
      locale === 'kk'
        ? 'Pulse 109 — оператордың жұмыс орны және ахуал орталығы'
        : 'Pulse 109 — рабочее место оператора и центр ситуации',
    )
  }, [locale])

  useEffect(() => {
    window.localStorage.setItem(THEME_KEY, theme)
    document.documentElement.dataset.theme = theme
    document.documentElement.style.colorScheme = theme
    const meta = document.querySelector('meta[name="theme-color"]')
    meta?.setAttribute('content', theme === 'dark' ? '#151c2b' : '#f5f7fb')
  }, [theme])

  const value = useMemo<UiSettings>(() => ({
    locale, setLocale, theme, setTheme,
    t: translateUi,
    number: (value) => new Intl.NumberFormat(locale === 'kk' ? 'kk-KZ' : 'ru-RU').format(value),
    dateTime: (value) => new Intl.DateTimeFormat(locale === 'kk' ? 'kk-KZ' : 'ru-RU', { dateStyle: 'medium', timeStyle: 'short' }).format(new Date(value)),
  }), [locale, theme])

  return <UiSettingsContext.Provider value={value}>{children}</UiSettingsContext.Provider>
}

export function useUiSettings(): UiSettings {
  const context = useContext(UiSettingsContext)
  if (!context) throw new Error('UiSettingsProvider is required')
  return context
}
