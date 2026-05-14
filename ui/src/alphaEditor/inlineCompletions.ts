import type * as Monaco from 'monaco-editor'

import { ALPHA_DSL_INLINE_ENTRIES } from './alphaDslCatalog'
import { getAlphaDslFragmentContext } from './completionContext'

const MAX_ITEMS = 5

/**
 * Grey ghost completions (inline suggest). Uses same catalog as dropdown completions.
 * Requires Monaco build with inline completions API (0.48+).
 */
export function registerAlphaInlineCompletions(monaco: typeof import('monaco-editor')): {
  dispose: () => void
} {
  const langs = monaco.languages as typeof monaco.languages & {
    registerInlineCompletionsProvider?: (
      selector: Monaco.languages.LanguageSelector,
      provider: unknown,
    ) => Monaco.IDisposable
  }

  if (typeof langs.registerInlineCompletionsProvider !== 'function') {
    return { dispose: () => {} }
  }

  const provider = langs.registerInlineCompletionsProvider('python', {
    provideInlineCompletions: (model, position, _context, token) => {
      if (token.isCancellationRequested) {
        return { items: [] }
      }
      const { fragment: tail } = getAlphaDslFragmentContext(model, position)
      if (!tail) {
        return { items: [] }
      }

      const matches: { insertText: string; score: number }[] = []
      for (const e of ALPHA_DSL_INLINE_ENTRIES) {
        if (e.insertText.includes('$')) continue
        if (!e.insertText.startsWith(tail) || e.insertText.length <= tail.length) continue
        const suffix = e.insertText.slice(tail.length)
        if (!suffix) continue
        matches.push({ insertText: suffix, score: e.insertText.length })
      }
      matches.sort((a, b) => b.score - a.score)
      const items: Monaco.languages.InlineCompletion[] = matches.slice(0, MAX_ITEMS).map((m) => ({
        insertText: m.insertText,
        range: new monaco.Range(
          position.lineNumber,
          position.column,
          position.lineNumber,
          position.column,
        ),
      }))

      return { items }
    },
    disposeInlineCompletions: () => {},
  })

  return provider
}
