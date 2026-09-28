import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import { UiSettingsProvider } from './uiSettings'
import './styles.css'
import './styles/pulse-template.css'
import './styles/app-adapters.css'
import './styles/polish.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <UiSettingsProvider><App /></UiSettingsProvider>
  </StrictMode>,
)
