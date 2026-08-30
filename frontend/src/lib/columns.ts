/**
 * Screen 6's client-side column check (frontend.md: "On column mismatch,
 * name the mismatched columns explicitly before any request is sent").
 * The backend rejects a mismatch too (Stage 6,
 * app/core/preprocessing.py's FittedPreprocessors.transform) -- this is an
 * earlier, friendlier surfacing of the same rule, not a replacement for it.
 */

export interface ColumnMismatch {
  missing: string[];
  unexpected: string[];
}

/** Splits a CSV's first line into column names. Header-only, not a full CSV
 * parser -- good enough to name a mismatch before upload; the backend's own
 * parse (pandas) is still the source of truth once the file is sent. */
export function parseCsvHeader(text: string): string[] {
  const firstLine = text.split(/\r\n|\r|\n/, 1)[0] ?? "";
  if (firstLine.trim() === "") return [];
  return firstLine.split(",").map((column) => column.trim().replace(/^"|"$/g, ""));
}

/** Null when the columns match exactly (order-independent, since the
 * backend's preprocessing looks columns up by name, not position). */
export function diffColumns(expected: string[], actual: string[]): ColumnMismatch | null {
  const expectedSet = new Set(expected);
  const actualSet = new Set(actual);
  const missing = expected.filter((column) => !actualSet.has(column));
  const unexpected = actual.filter((column) => !expectedSet.has(column));
  if (missing.length === 0 && unexpected.length === 0) return null;
  return { missing, unexpected };
}

/** Quotes a single CSV field when it contains a comma, quote or newline --
 * the inverse of parseCsvHeader's quote-stripping. Used to build the
 * single-row CSV screen 6's manual-entry mode assembles in the browser
 * (frontend.md §6: "posting it through the existing upload-based predict
 * endpoint unchanged"). */
function csvField(value: string): string {
  if (/[",\n\r]/.test(value)) return `"${value.replace(/"/g, '""')}"`;
  return value;
}

/** Builds a one-row CSV (header + single data row) from an ordered list of
 * column names and a value per name, in the same column order the CSV-mode
 * upload would use. */
export function buildSingleRowCsv(columns: string[], values: Record<string, string>): string {
  const header = columns.map(csvField).join(",");
  const row = columns.map((column) => csvField(values[column] ?? "")).join(",");
  return `${header}\r\n${row}\r\n`;
}
