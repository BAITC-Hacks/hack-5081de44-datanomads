import type { ClassificationConfidenceState } from './types'

export function normalizeConfidenceState(
  value: string | undefined,
  confidence: number,
  available = true,
): ClassificationConfidenceState {
  if (!available) return 'UNAVAILABLE'

  switch (value?.trim().toUpperCase()) {
    case 'CONFIDENT':
    case 'HIGH':
      return 'CONFIDENT'
    case 'UNCERTAIN':
    case 'MEDIUM':
      return 'UNCERTAIN'
    case 'LOW_CONFIDENCE':
    case 'LOW':
      return 'LOW_CONFIDENCE'
    case undefined:
    case '':
      if (confidence >= 0.85) return 'CONFIDENT'
      if (confidence >= 0.7) return 'UNCERTAIN'
      return 'LOW_CONFIDENCE'
    default:
      return 'UNAVAILABLE'
  }
}

export function confidenceStateLabel(state: ClassificationConfidenceState): string {
  switch (state) {
    case 'CONFIDENT':
      return 'Высокая'
    case 'UNCERTAIN':
      return 'Средняя'
    case 'LOW_CONFIDENCE':
      return 'Низкая'
    case 'UNAVAILABLE':
      return 'нет данных'
  }
}

export function confidenceStateNotice(state: ClassificationConfidenceState): string {
  switch (state) {
    case 'CONFIDENT':
      return 'Уверенная рекомендация модели'
    case 'UNCERTAIN':
      return 'Рекомендацию нужно проверить'
    case 'LOW_CONFIDENCE':
      return 'Низкая уверенность — выберите тему вручную'
    case 'UNAVAILABLE':
      return 'Классификация недоступна — выберите тему вручную'
  }
}

export function classificationAlternatives<T extends { topic_id: string }>(
  state: ClassificationConfidenceState,
  predictedTopicId: string,
  alternatives: T[],
): T[] {
  if (state !== 'UNCERTAIN') return []
  return alternatives.filter((alternative) => alternative.topic_id !== predictedTopicId).slice(0, 2)
}
