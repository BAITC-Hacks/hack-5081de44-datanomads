export type DemoRole = 'OPERATOR' | 'MANAGER' | 'ML_REVIEWER'

const ROLE_STORAGE_KEY = 'pulse109.demo-role'

export const roleHome: Record<DemoRole, string> = {
  OPERATOR: '/operator',
  MANAGER: '/situation/overview',
  ML_REVIEWER: '/situation/learning',
}

export function readDemoRole(): DemoRole | null {
  const value = window.sessionStorage.getItem(ROLE_STORAGE_KEY)
  return value === 'OPERATOR' || value === 'MANAGER' || value === 'ML_REVIEWER' ? value : null
}

export function saveDemoRole(role: DemoRole | null): void {
  if (role) window.sessionStorage.setItem(ROLE_STORAGE_KEY, role)
  else window.sessionStorage.removeItem(ROLE_STORAGE_KEY)
}

export function canOpenRoute(role: DemoRole, route: string): boolean {
  if (role === 'OPERATOR') return route === '/operator'
  if (role === 'ML_REVIEWER') return route === '/situation/learning' || route === '/situation/models'
  return route.startsWith('/situation/') && route !== '/situation/learning' && route !== '/situation/models'
}
