import './monacoSetup'
import '@mantine/core/styles.css'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter } from 'react-router-dom'
import { MantineProvider, localStorageColorSchemeManager } from '@mantine/core'
import './index.css'
import App from './App.tsx'
import { initTableDensity } from './density'
import LegacyThemeSync from './components/LegacyThemeSync'
import { shunyaCssVariablesResolver } from './mantine/cssVariablesResolver'
import { shunyaMantineTheme } from './mantine/theme'
import { SHUNYA_THEME_STORAGE_KEY, resolveInitialTheme } from './theme'

initTableDensity()

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      retry: 1,
    },
  },
})

const colorSchemeManager = localStorageColorSchemeManager({
  key: SHUNYA_THEME_STORAGE_KEY,
})

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <MantineProvider
      theme={shunyaMantineTheme}
      cssVariablesResolver={shunyaCssVariablesResolver}
      defaultColorScheme={resolveInitialTheme()}
      colorSchemeManager={colorSchemeManager}
    >
      <LegacyThemeSync />
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <App />
        </BrowserRouter>
      </QueryClientProvider>
    </MantineProvider>
  </StrictMode>,
)
