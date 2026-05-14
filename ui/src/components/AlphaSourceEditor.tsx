import Editor, { useMonaco } from '@monaco-editor/react'
import type { editor } from 'monaco-editor'
import { Paper, useComputedColorScheme } from '@mantine/core'
import { useCallback, useEffect, useRef, useState } from 'react'
import type { AlphaAssistIssue } from '../api/types'
import { postAlphaAssistBody, postAlphaLintBody } from '../api/endpoints'
import { registerAlphaCompletions } from '../alphaEditor/completions'
import { ALPHA_COPY_DIAGNOSTIC_CMD, registerAlphaLintCodeActions } from '../alphaEditor/codeActions'
import { registerAlphaInlineCompletions } from '../alphaEditor/inlineCompletions'

export type AlphaEditorMeta = {
  name?: string | null
  description?: string | null
}

type Props = {
  value: string
  onChange: (v: string) => void
  readOnly?: boolean
  /** Editor height, e.g. 58vh */
  height?: string
  /** When set, passed to AI assist for context. */
  alphaMeta?: AlphaEditorMeta
  /** Pyright diagnostics via API (default true). */
  enableLint?: boolean
  /** Ollama review markers (default false; enable when API has Ollama configured). */
  enableAssist?: boolean
  /** Increment to force an immediate assist-body refresh (e.g. after Fix). */
  assistNonce?: number
  /** Latest assist issues (includes optional corrected_body for Fix). */
  onAssistIssues?: (issues: AlphaAssistIssue[]) => void
}

const LINT_DEBOUNCE_MS = 700
const ASSIST_DEBOUNCE_MS = 2800

export default function AlphaSourceEditor({
  value,
  onChange,
  readOnly = false,
  height = '58vh',
  alphaMeta,
  enableLint = true,
  enableAssist = false,
  assistNonce = 0,
  onAssistIssues,
}: Props) {
  const monaco = useMonaco()
  const regRef = useRef<{ dispose: () => void } | null>(null)
  const editorRef = useRef<editor.IStandaloneCodeEditor | null>(null)
  const [editorGen, setEditorGen] = useState(0)
  const colorScheme = useComputedColorScheme('light')
  const editorTheme = colorScheme === 'dark' ? 'vs-dark' : 'light'

  const clearMarkers = useCallback(() => {
    const ed = editorRef.current
    const m = ed?.getModel()
    if (!m || !monaco) return
    monaco.editor.setModelMarkers(m, 'pyright', [])
    monaco.editor.setModelMarkers(m, 'alpha-ai', [])
  }, [monaco])

  useEffect(() => {
    if (!monaco) return
    regRef.current?.dispose()
    const d1 = registerAlphaCompletions(monaco)
    const d2 = registerAlphaInlineCompletions(monaco)
    const d3 = registerAlphaLintCodeActions(monaco)
    const d4 = monaco.editor.registerCommand(ALPHA_COPY_DIAGNOSTIC_CMD, (_accessor, msg?: unknown) => {
      if (typeof msg === 'string' && msg.length > 0) void navigator.clipboard.writeText(msg)
    })
    regRef.current = {
      dispose: () => {
        d1.dispose()
        d2.dispose()
        d3.dispose()
        d4.dispose()
      },
    }
    return () => {
      regRef.current?.dispose()
      regRef.current = null
    }
  }, [monaco])

  useEffect(() => {
    return () => {
      clearMarkers()
    }
  }, [clearMarkers])

  useEffect(() => {
    if (!monaco || !enableLint) return
    const ed = editorRef.current
    const model = ed?.getModel()
    if (!model || editorGen === 0) return

    let cancelled = false
    const t = window.setTimeout(() => {
      void (async () => {
        try {
          const res = await postAlphaLintBody({ source_body: value })
          if (cancelled || editorRef.current?.getModel() !== model) return
          const markers = res.diagnostics.map((d) => ({
            severity:
              d.severity === 'error'
                ? monaco.MarkerSeverity.Error
                : d.severity === 'warning'
                  ? monaco.MarkerSeverity.Warning
                  : monaco.MarkerSeverity.Info,
            message: d.message,
            startLineNumber: d.startLineNumber,
            startColumn: d.startColumn,
            endLineNumber: d.endLineNumber,
            endColumn: d.endColumn,
          }))
          monaco.editor.setModelMarkers(model, 'pyright', markers)
        } catch {
          if (!cancelled && model === editorRef.current?.getModel()) {
            monaco.editor.setModelMarkers(model, 'pyright', [])
          }
        }
      })()
    }, LINT_DEBOUNCE_MS)
    return () => {
      cancelled = true
      window.clearTimeout(t)
    }
  }, [value, monaco, enableLint, editorGen])

  useEffect(() => {
    if (!monaco || !enableAssist) return
    const ed = editorRef.current
    const model = ed?.getModel()
    if (!model || editorGen === 0) return

    let cancelled = false
    const t = window.setTimeout(() => {
      void (async () => {
        try {
          const res = await postAlphaAssistBody({
            source_body: value,
            alpha_name: alphaMeta?.name ?? undefined,
            alpha_description: alphaMeta?.description ?? undefined,
          })
          if (cancelled || editorRef.current?.getModel() !== model) return
          const issues = res.issues?.length ? res.issues : []
          onAssistIssues?.(issues)
          const markers = (issues.length ? issues : res.markers).map((m) => ({
            severity:
              m.severity === 'error' || m.severity === 'warning'
                ? monaco.MarkerSeverity.Warning
                : monaco.MarkerSeverity.Info,
            message: m.message,
            startLineNumber: m.startLineNumber,
            startColumn: m.startColumn,
            endLineNumber: m.endLineNumber,
            endColumn: m.endColumn,
          }))
          monaco.editor.setModelMarkers(model, 'alpha-ai', markers)
        } catch {
          if (!cancelled && model === editorRef.current?.getModel()) {
            monaco.editor.setModelMarkers(model, 'alpha-ai', [])
            onAssistIssues?.([])
          }
        }
      })()
    }, ASSIST_DEBOUNCE_MS)
    return () => {
      cancelled = true
      window.clearTimeout(t)
    }
  }, [
    value,
    monaco,
    enableAssist,
    alphaMeta?.name,
    alphaMeta?.description,
    editorGen,
    assistNonce,
    onAssistIssues,
  ])

  const onMount = useCallback((ed: editor.IStandaloneCodeEditor) => {
    editorRef.current = ed
    setEditorGen((n) => n + 1)
  }, [])

  return (
    <Paper p={0} radius="sm" withBorder style={{ overflow: 'hidden' }}>
      <Editor
        key={editorTheme}
        height={height}
        defaultLanguage="python"
        theme={editorTheme}
        value={value}
        onChange={(v) => onChange(v ?? '')}
        onMount={onMount}
        options={{
          readOnly,
          fontSize: 14,
          minimap: { enabled: true },
          scrollBeyondLastLine: false,
          wordWrap: 'on',
          tabSize: 4,
          fixedOverflowWidgets: true,
          quickSuggestions: { other: true, comments: false, strings: true },
          suggestOnTriggerCharacters: true,
          inlineSuggest: { enabled: true },
        }}
      />
    </Paper>
  )
}
