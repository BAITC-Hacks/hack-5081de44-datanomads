export type IconName = 'inbox' | 'pulse' | 'grid' | 'map' | 'tag' | 'trend' | 'bell' | 'forecast' | 'file' | 'cycle' | 'model' | 'search' | 'filter' | 'settings' | 'help' | 'chevron' | 'arrow' | 'check' | 'edit' | 'external' | 'download' | 'more' | 'clock' | 'close' | 'user' | 'sun' | 'moon' | 'logout'

const icons: Record<IconName, string> = {
  inbox: 'M4 5h16v14H4z M4 8h16 M8 12h3',
  pulse: 'M3 12h3l2-6 4 12 2-6h7',
  grid: 'M4 4h6v6H4z M14 4h6v6h-6z M4 14h6v6H4z M14 14h6v6h-6z',
  map: 'M4 6l5-2 6 2 5-2v14l-5 2-6-2-5 2z M9 4v14 M15 6v14',
  tag: 'M4 5v6l9 9 7-7-9-9H4z M8 8h.01',
  trend: 'M4 18l5-5 3 3 8-9 M15 7h5v5',
  bell: 'M6 17h12l-1.5-2v-4a4.5 4.5 0 0 0-9 0v4z M10 20h4',
  forecast: 'M4 17l4-5 3 2 5-7 4 3 M4 20h16 M4 4v16',
  file: 'M6 3h8l4 4v14H6z M14 3v5h4 M9 13h6 M9 17h4',
  cycle: 'M5 8a7 7 0 0 1 12-2l2 2 M19 16a7 7 0 0 1-12 2l-2-2 M19 6v4h-4 M5 18v-4h4',
  model: 'M5 4h14v16H5z M8 8h8 M8 12h8 M8 16h5',
  search: 'M11 18a7 7 0 1 1 0-14 7 7 0 0 1 0 14z M16 16l4 4',
  filter: 'M4 7h16 M7 12h10 M10 17h4',
  settings: 'M12 15.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7z M19 13v-2l-2-.6a7 7 0 0 0-.7-1.7l.9-1.8-1.4-1.4-1.8.9a7 7 0 0 0-1.7-.7L12.7 4h-2l-.6 1.7a7 7 0 0 0-1.7.7l-1.8-.9-1.4 1.4.9 1.8a7 7 0 0 0-.7 1.7L4 11v2l1.7.6a7 7 0 0 0 .7 1.7l-.9 1.8 1.4 1.4 1.8-.9a7 7 0 0 0 1.7.7l.6 1.7h2l.6-1.7a7 7 0 0 0 1.7-.7l1.8.9 1.4-1.4-.9-1.8a7 7 0 0 0-.7-1.7z',
  help: 'M9.5 9a2.5 2.5 0 1 1 4.1 1.9c-.9.7-1.6 1.2-1.6 2.6 M12 17h.01 M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18z',
  chevron: 'M7 10l5 5 5-5',
  arrow: 'M5 12h14 M13 6l6 6-6 6',
  check: 'M5 12l4 4L19 6',
  edit: 'M4 20h4L19 9l-4-4L4 16z M13 6l4 4',
  external: 'M14 5h5v5 M19 5l-8 8 M18 13v5H5V5h5',
  download: 'M12 4v11 M8 11l4 4 4-4 M5 20h14',
  more: 'M6 12h.01 M12 12h.01 M18 12h.01',
  clock: 'M12 7v5l3 2 M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18z',
  close: 'M6 6l12 12 M18 6L6 18',
  user: 'M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8z M4.5 20a7.5 7.5 0 0 1 15 0',
  sun: 'M12 3v2 M12 19v2 M3 12h2 M19 12h2 M5.6 5.6 7 7 M17 17l1.4 1.4 M5.6 18.4 7 17 M17 7l1.4-1.4 M12 16a4 4 0 1 0 0-8 4 4 0 0 0 0 8z',
  moon: 'M20 15.5A8 8 0 0 1 8.5 4 8 8 0 1 0 20 15.5z',
  logout: 'M9 4H5v16h4 M13 8l4 4-4 4 M17 12H8',
}

export function Icon({ name, size = 18 }: { name: IconName; size?: number }) {
  return <svg aria-hidden="true" className="icon" width={size} height={size} style={{ width: size, height: size }} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d={icons[name]} /></svg>
}
