import type * as Monaco from 'monaco-editor'

const FRAGMENT_RE = /[A-Za-z0-9_.]/

function isFragmentChar(ch: string): boolean {
  return ch.length === 1 && FRAGMENT_RE.test(ch)
}

/**
 * Maximal DSL prefix ending at the cursor on this line: e.g. `ctx.`, `ts.me`, `cs`.
 * Used so completion replaces the fragment instead of inserting after a zero-width
 * "word" (fixes `ctx.` + `ctx.close` → `ctx.ctx.close`).
 */
export function getAlphaDslFragmentContext(
  model: Monaco.editor.ITextModel,
  position: Monaco.Position,
): { fragment: string; replaceRange: Monaco.IRange } {
  const line = model.getLineContent(position.lineNumber)
  const col0 = Math.max(0, position.column - 1)
  let i = col0 - 1
  while (i >= 0 && isFragmentChar(line[i]!)) {
    i -= 1
  }
  const startCol0 = i + 1
  const fragment = line.slice(startCol0, col0)
  const replaceRange: Monaco.IRange = {
    startLineNumber: position.lineNumber,
    startColumn: startCol0 + 1,
    endLineNumber: position.lineNumber,
    endColumn: position.column,
  }
  return { fragment, replaceRange }
}
