/// <reference types="vite/client" />

interface Window {
  MonacoEnvironment?: { getWorker: (workerId: string, label: string) => Worker }
}

interface ImportMetaEnv {
  readonly VITE_API_BASE?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
