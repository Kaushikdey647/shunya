import type * as Monaco from 'monaco-editor'

import { ALPHA_DSL_INLINE_ENTRIES } from './alphaDslCatalog'
import { getAlphaDslFragmentContext } from './completionContext'

type CompletionWithoutRange = Omit<Monaco.languages.CompletionItem, 'range'>

const SNIPPET = (m: typeof Monaco) => m.languages.CompletionItemInsertTextRule.InsertAsSnippet

function kindForLabel(m: typeof Monaco, label: string): Monaco.languages.CompletionItemKind {
  const k = m.languages.CompletionItemKind
  if (label.startsWith('jnp.')) return k.Function
  if (label.startsWith('ctx.')) return k.Property
  if (label.startsWith('fun.')) return k.Property
  if (label.startsWith('ts.') || label.startsWith('cs.')) return k.Method
  return k.Text
}

const buildItems = (m: typeof Monaco) => {
  const k = m.languages.CompletionItemKind
  const mk = (
    label: string,
    insertText: string,
    kind: Monaco.languages.CompletionItemKind,
    detail: string,
    doc?: string,
    opts?: { sortText?: string; insertTextRules?: Monaco.languages.CompletionItemInsertTextRule },
  ): CompletionWithoutRange => ({
    label,
    kind,
    insertText,
    detail,
    documentation: doc,
    sortText: opts?.sortText ?? label,
    insertTextRules: opts?.insertTextRules,
  })

  const fromCatalog: CompletionWithoutRange[] = ALPHA_DSL_INLINE_ENTRIES.map((e) =>
    mk(e.label, e.insertText, kindForLabel(m, e.label), e.detail, e.detail),
  )

  return [
    ...fromCatalog,
    mk(
      'snippet_rank_close',
      'return cs.rank(ctx.close)',
      k.Snippet,
      'Snippet: cross-section rank of close',
      'Cross-section rank of latest close.',
      {
        sortText: '0snippet',
        insertTextRules: SNIPPET(m),
      },
    ),
    mk(
      'snippet_ts_mean',
      'return cs.rank(ts.mean(ctx.close, ${1:20}))',
      k.Snippet,
      'Snippet: rolling mean then cross-section rank',
      'Rolling mean of close then rank latest snapshot.',
      {
        sortText: '1snippet',
        insertTextRules: SNIPPET(m),
      },
    ),
  ]
}

export function registerAlphaCompletions(monaco: typeof import('monaco-editor')): {
  dispose: () => void
} {
  const suggestions = buildItems(monaco)
  const provider = monaco.languages.registerCompletionItemProvider('python', {
    triggerCharacters: ['.', '(', ','],
    provideCompletionItems: (model, position) => {
      const { fragment, replaceRange } = getAlphaDslFragmentContext(model, position)
      const filtered = suggestions.filter((s) => {
        if (!fragment) return true
        return s.insertText.startsWith(fragment)
      })
      return {
        suggestions: filtered.map((s) => ({
          ...s,
          range: replaceRange,
        })),
      }
    },
  })
  return provider
}
