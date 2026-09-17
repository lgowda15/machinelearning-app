const COLOR_VAR_PATTERN = /var\((--color-[a-z0-9-]+)\)/gi;

/**
 * Replaces every `var(--color-*)` occurrence in a captured `<svg>`'s
 * attributes and inline `style` strings with its resolved literal value.
 * svg2pdf.js parses SVG attributes itself rather than using the browser's
 * rendering pipeline, so it can't resolve CSS custom properties -- this
 * must happen before the SVG is handed to it (docs/plans/pdf-report.md's
 * "Chart colors" design decision).
 */
export function resolveCssVars(svg: SVGElement, styleSource: Element = document.documentElement): void {
  const computed = getComputedStyle(styleSource);
  const resolved = new Map<string, string>();

  function resolveValue(varName: string): string {
    const cached = resolved.get(varName);
    if (cached !== undefined) return cached;
    const value = computed.getPropertyValue(varName).trim();
    resolved.set(varName, value);
    return value;
  }

  function rewrite(value: string): string {
    return value.replace(COLOR_VAR_PATTERN, (match, varName: string) => resolveValue(varName) || match);
  }

  function walk(el: Element): void {
    for (const attr of Array.from(el.attributes)) {
      if (attr.value.includes("var(--color-")) {
        el.setAttribute(attr.name, rewrite(attr.value));
      }
    }

    for (const child of Array.from(el.children)) {
      walk(child);
    }
  }

  walk(svg);
}
