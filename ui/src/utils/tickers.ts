export function parseTickerList(comma: string): string[] {
  return comma
    .split(',')
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
}
