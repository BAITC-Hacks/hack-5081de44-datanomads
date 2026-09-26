import type { PreviewLanguage } from './types'

export function mapLanguage(value: string | null | undefined): PreviewLanguage {
  switch (value?.trim().toUpperCase()) {
    case 'RU':
    case 'RUS':
      return 'RU'
    case 'KZ':
    case 'KK':
    case 'KAZ':
      return 'KZ'
    case 'MIXED':
      return 'MIXED'
    case 'UNKNOWN':
    default:
      return 'UNKNOWN'
  }
}

export function languageLabel(language: PreviewLanguage): string {
  switch (language) {
    case 'RU':
      return 'Русский (RU)'
    case 'KZ':
      return 'Казахский (KZ)'
    case 'MIXED':
      return 'Смешанный (MIXED)'
    case 'UNKNOWN':
      return 'Не определён (UNKNOWN)'
  }
}

export function languageReviewNotice(language: PreviewLanguage): string | undefined {
  if (language === 'MIXED') return 'Обнаружен смешанный язык. Проверьте обращение вручную.'
  if (language === 'UNKNOWN') return 'Язык не определён. Проверьте язык вручную.'
  return undefined
}
