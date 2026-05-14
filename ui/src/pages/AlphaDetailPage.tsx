import { Navigate, useParams } from 'react-router-dom'

/** @deprecated Legacy route; App redirects `/alphas/:id` to `/studio/:id`. Kept for direct imports during transition. */
export default function AlphaDetailPage() {
  const { alphaId } = useParams<{ alphaId: string }>()
  if (!alphaId) return <Navigate to="/studio" replace />
  return <Navigate to={`/studio/${encodeURIComponent(alphaId)}`} replace />
}
