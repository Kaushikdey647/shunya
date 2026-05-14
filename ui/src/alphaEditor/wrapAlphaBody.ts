/**
 * Must stay in sync with ``shunya/algorithm/alpha_source_wrap.py``:
 * canonical prefix and header line count.
 */
export const ALPHA_SOURCE_HEADER_LINE_COUNT = 6

export const ALPHA_SOURCE_CANONICAL_PREFIX = `import jax.numpy as jnp

def alpha(ctx) -> jnp.ndarray:
    ts = ctx.ts
    cs = ctx.cs
    fun = ctx.fun
`

/** Full module string from editor-visible function body (no import/def). */
export function wrapAlphaBody(body: string): string {
  const normalized = body.replace(/\r\n/g, '\n').replace(/\n$/, '')
  const trimmed = normalized.trimEnd()
  if (!trimmed.trim()) {
    return `${ALPHA_SOURCE_CANONICAL_PREFIX}    pass\n`
  }
  const indented = trimmed
    .split('\n')
    .map((line) => (line.length ? `    ${line}` : ''))
    .join('\n')
  return `${ALPHA_SOURCE_CANONICAL_PREFIX}${indented}\n`
}

function dedentBodyBlock(block: string): string {
  const lines = block.replace(/\r\n/g, '\n').split('\n')
  const out: string[] = []
  for (const line of lines) {
    if (line.startsWith('    ')) {
      out.push(line.slice(4))
    } else {
      out.push(line)
    }
  }
  return out.join('\n').replace(/\n+$/, '').replace(/^\n+/, '')
}

/**
 * Extract body text from stored full ``source_code`` (canonical or legacy
 * ``def alpha`` module). Falls back to whole-file dedent if no alpha function.
 */
export function unwrapAlphaSource(source: string): string {
  const text = source.replace(/\r\n/g, '\n')
  if (text.startsWith(ALPHA_SOURCE_CANONICAL_PREFIX)) {
    let rest = text.slice(ALPHA_SOURCE_CANONICAL_PREFIX.length)
    if (rest.startsWith('\n')) {
      rest = rest.slice(1)
    }
    return dedentBodyBlock(rest)
  }
  return unwrapLegacyAlphaSource(text)
}

/** Best-effort legacy unwrap without full AST in the browser. */
function unwrapLegacyAlphaSource(text: string): string {
  const lines = text.split('\n')
  const defRe = /^\s*def\s+alpha\s*\(\s*ctx\s*\)/
  let defIdx = -1
  for (let i = 0; i < lines.length; i++) {
    if (defRe.test(lines[i]!)) {
      defIdx = i
      break
    }
  }
  if (defIdx < 0) {
    return text.trim()
  }
  const bodyLines = lines.slice(defIdx + 1)
  if (bodyLines.length === 0) {
    return ''
  }
  const nonEmpty = bodyLines.find((l) => l.trim().length > 0)
  if (!nonEmpty) {
    return ''
  }
  const baseIndent = nonEmpty.match(/^\s*/)![0].length
  const stripped = bodyLines.map((line) => {
    if (!line.trim()) {
      return ''
    }
    if (line.length >= baseIndent) {
      return line.slice(baseIndent)
    }
    return line.trimStart()
  })
  let start = 0
  if (
    stripped.length >= 3 &&
    /^\s*ts\s*=\s*ctx\.ts\s*$/.test(stripped[0]!) &&
    /^\s*cs\s*=\s*ctx\.cs\s*$/.test(stripped[1]!) &&
    /^\s*fun\s*=\s*ctx\.fun\s*$/.test(stripped[2]!)
  ) {
    start = 3
  }
  return stripped.slice(start).join('\n').replace(/\n+$/, '')
}

export function bodyLineToWrappedLine(bodyLine1Based: number): number {
  return ALPHA_SOURCE_HEADER_LINE_COUNT + bodyLine1Based
}

export function wrappedLineToBodyLine(wrappedLine1Based: number): number | null {
  if (wrappedLine1Based <= ALPHA_SOURCE_HEADER_LINE_COUNT) {
    return null
  }
  return wrappedLine1Based - ALPHA_SOURCE_HEADER_LINE_COUNT
}
