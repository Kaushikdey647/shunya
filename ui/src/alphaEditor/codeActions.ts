import type * as Monaco from 'monaco-editor'

/** Must match `AlphaSourceEditor` command registration. */
export const ALPHA_COPY_DIAGNOSTIC_CMD = 'alpha-studio.copyDiagnostic'

const MARKER_OWNERS = ['pyright', 'alpha-ai'] as const

function rangeIntersectsMarker(
  r: Monaco.IRange,
  m: Pick<Monaco.editor.IMarkerData, 'startLineNumber' | 'startColumn' | 'endLineNumber' | 'endColumn'>,
): boolean {
  const rLo = r.startLineNumber * 1_000_000 + r.startColumn
  const rHi = r.endLineNumber * 1_000_000 + r.endColumn
  const mLo = m.startLineNumber * 1_000_000 + m.startColumn
  const mHi = m.endLineNumber * 1_000_000 + m.endColumn
  return rLo <= mHi && mLo <= rHi
}

/** Pyright / Pylance often append a did-you-mean clause. */
function extractDidYouMean(message: string): string | null {
  const patterns = [/Did you mean "([^"]+)"/i, /Did you mean '([^']+)'/i, /Did you mean `([^`]+)`/i]
  for (const re of patterns) {
    const x = message.match(re)
    if (x?.[1]) return x[1]
  }
  return null
}

export function registerAlphaLintCodeActions(monaco: typeof import('monaco-editor')): {
  dispose: () => void
} {
  return monaco.languages.registerCodeActionProvider('python', {
    provideCodeActions(model, range, _context) {
      const markers = monaco.editor
        .getModelMarkers({ resource: model.uri })
        .filter(
          (m) =>
            MARKER_OWNERS.includes(m.owner as (typeof MARKER_OWNERS)[number]) &&
            rangeIntersectsMarker(range, m),
        )

      const actions: Monaco.languages.CodeAction[] = []

      for (const m of markers) {
        const msg = m.message ?? ''
        const replacement = extractDidYouMean(msg)
        if (replacement) {
          actions.push({
            title: `Replace with "${replacement}"`,
            kind: 'quickfix',
            diagnostics: [m],
            edit: {
              edits: [
                {
                  resource: model.uri,
                  versionId: undefined,
                  textEdit: {
                    range: new monaco.Range(
                      m.startLineNumber,
                      m.startColumn,
                      m.endLineNumber,
                      m.endColumn,
                    ),
                    text: replacement,
                  },
                },
              ],
            },
          })
        }
        actions.push({
          title: 'Copy diagnostic message',
          kind: 'quickfix',
          diagnostics: [m],
          command: {
            id: ALPHA_COPY_DIAGNOSTIC_CMD,
            title: 'Copy',
            arguments: [msg],
          },
        })
      }

      return {
        actions,
        dispose: () => {},
      }
    },
  })
}
