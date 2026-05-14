import { Navigate, Route, Routes, useParams } from 'react-router-dom'
import AppShell from './components/AppShell'
import './layout.css'
import AlphaStudioLayout, {
  AlphaStudioWorkspace,
  StudioAlphaCreate,
  StudioAlphaHub,
} from './pages/AlphaStudioPage'
import BacktestDetailPage from './pages/BacktestDetailPage'
import BacktestNewPage from './pages/BacktestNewPage'
import BacktestsListPage from './pages/BacktestsListPage'
import DataSummaryPage from './pages/DataSummaryPage'
import ExecutionHubPage from './pages/ExecutionHubPage'
import ExecutionTracerPage from './pages/ExecutionTracerPage'
import HomePage from './pages/HomePage'
import InstrumentDetailPage from './pages/InstrumentDetailPage'
import LiveCockpitPage from './pages/LiveCockpitPage'
import PortfoliosListPage from './pages/PortfoliosListPage'
import PortfolioWorkspacePage from './pages/PortfolioWorkspacePage'
import RiskCommandCenterPage from './pages/RiskCommandCenterPage'
import SearchResultsPage from './pages/SearchResultsPage'
import SettingsPage from './pages/SettingsPage'
import TradeAccountPage from './pages/TradeAccountPage'
import UniverseDetailPage from './pages/UniverseDetailPage'
import UniverseListPage from './pages/UniverseListPage'

function RedirectLegacyAlphaToStudio() {
  const { alphaId } = useParams<{ alphaId: string }>()
  return <Navigate to={`/studio/${encodeURIComponent(alphaId!)}`} replace />
}

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<AppShell />}>
        <Route index element={<HomePage />} />
        <Route path="search" element={<SearchResultsPage />} />
        <Route path="instruments/:symbol" element={<InstrumentDetailPage />} />
        <Route path="studio" element={<AlphaStudioLayout />}>
          <Route index element={<StudioAlphaHub />} />
          <Route path="new" element={<StudioAlphaCreate />} />
          <Route path=":alphaId" element={<AlphaStudioWorkspace />} />
        </Route>
        <Route path="alphas" element={<Navigate to="/studio" replace />} />
        <Route path="alphas/new" element={<Navigate to="/studio/new" replace />} />
        <Route path="alphas/:alphaId/studio" element={<RedirectLegacyAlphaToStudio />} />
        <Route path="alphas/:alphaId" element={<RedirectLegacyAlphaToStudio />} />
        <Route path="backtests" element={<BacktestsListPage />} />
        <Route path="backtests/new" element={<BacktestNewPage />} />
        <Route path="backtests/:jobId" element={<BacktestDetailPage />} />
        <Route path="data" element={<DataSummaryPage />} />
        <Route path="universe" element={<UniverseListPage />} />
        <Route path="universe/:id" element={<UniverseDetailPage />} />
        <Route path="settings" element={<SettingsPage />} />
        <Route path="portfolios" element={<PortfoliosListPage />} />
        <Route path="portfolios/:id" element={<PortfolioWorkspacePage />} />
        <Route path="live" element={<LiveCockpitPage />} />
        <Route path="execution/:parentId" element={<ExecutionTracerPage />} />
        <Route path="execution" element={<ExecutionHubPage />} />
        <Route path="risk" element={<RiskCommandCenterPage />} />
        <Route path="trade/account" element={<TradeAccountPage />} />
      </Route>
    </Routes>
  )
}
