import { Navigate } from 'react-router-dom'

/** @deprecated Use `/studio`. Kept for direct imports during transition. */
export default function AlphasListPage() {
  return <Navigate to="/studio" replace />
}
