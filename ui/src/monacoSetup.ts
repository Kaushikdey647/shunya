/**
 * Monaco web workers for Vite (avoids vite-plugin-monaco-editor, which breaks on
 * Node 22+ when clearing its cache: fs.rmdirSync no longer accepts `recursive`).
 */
import EditorWorker from 'monaco-editor/esm/vs/editor/editor.worker.js?worker'
import JsonWorker from 'monaco-editor/esm/vs/language/json/json.worker.js?worker'
import TsWorker from 'monaco-editor/esm/vs/language/typescript/ts.worker.js?worker'
import CssWorker from 'monaco-editor/esm/vs/language/css/css.worker.js?worker'
import HtmlWorker from 'monaco-editor/esm/vs/language/html/html.worker.js?worker'

self.MonacoEnvironment = {
  getWorker(_workerId, label) {
    switch (label) {
      case 'json':
        return new JsonWorker()
      case 'css':
      case 'scss':
      case 'less':
        return new CssWorker()
      case 'html':
      case 'handlebars':
      case 'razor':
        return new HtmlWorker()
      case 'typescript':
      case 'javascript':
        return new TsWorker()
      default:
        return new EditorWorker()
    }
  },
}
