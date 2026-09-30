/** Splits an LLM's free-text response into individual items, stripping common list decoration (numbering,
 * bullets) a model tends to add even when told not to — every metric here asks for "one per line, no numbering"
 * but judges the response defensively rather than assuming perfect compliance. */
export function parseLines(text: string): string[] {
  return text
    .split('\n')
    .map((line) => line.replace(/^\s*(?:[-*•]|\d+[.)])\s*/, '').trim())
    .filter((line) => line.length > 0)
}

/** A yes/no judgment defaults to `false` on anything ambiguous — under-crediting an unclear response is the
 * safer failure mode for a metric whose whole point is not to overstate how well-grounded something is. */
export function parseYesNo(text: string): boolean {
  return /^\s*yes\b/i.test(text)
}
