import { Navigate } from 'react-router-dom'

/** @deprecated Use `/studio/new`. Kept for direct imports during transition. */
export default function AlphaNewPage() {
  return <Navigate to="/studio/new" replace />
}
