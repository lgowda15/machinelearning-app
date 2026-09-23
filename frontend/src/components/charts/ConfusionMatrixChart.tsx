interface ConfusionMatrixChartProps {
  confusionMatrix: number[][];
  labels: string[];
}

const SCALE_CLASSES = [
  "bg-scale-1 text-ink",
  "bg-scale-2 text-ink",
  "bg-scale-3 text-ink",
  "bg-scale-4 text-white",
  "bg-scale-5 text-white",
] as const;

/** A cell's magnitude relative to its own row's max -- normalized per row
 * (not globally) so an imbalanced class with few samples still shows which
 * of its predictions dominates, rather than reading as uniformly faint next
 * to a much larger class's row. A count of 0 always gets the lightest step. */
function scaleClassForCell(count: number, rowMax: number): (typeof SCALE_CLASSES)[number] {
  if (count <= 0 || rowMax <= 0) return SCALE_CLASSES[0];
  const step = Math.min(SCALE_CLASSES.length - 1, Math.ceil((count / rowMax) * SCALE_CLASSES.length) - 1);
  return SCALE_CLASSES[step];
}

/** Screen 5's classifier chart (frontend.md). Rows are actual classes,
 * columns predicted. Cell colour is the sequential scale (--color-scale-1
 * .. -5, light to dark), normalized per row, so it encodes how a given
 * actual class's predictions were distributed -- not a categorical/type
 * colour. The diagonal (a correct call) stays visually distinct by weight,
 * not colour, since colour is already spent on magnitude. */
export function ConfusionMatrixChart({ confusionMatrix, labels }: ConfusionMatrixChartProps) {
  const total = confusionMatrix.reduce((sum, row) => sum + row.reduce((a, b) => a + b, 0), 0);
  const correct = confusionMatrix.reduce((sum, row, i) => sum + (row[i] ?? 0), 0);

  return (
    <div>
      <div className="overflow-x-auto rounded-panel border border-rule bg-surface">
        <table className="w-full text-center text-sm">
          <thead>
            <tr className="border-b border-rule text-xs uppercase tracking-wide text-muted">
              <th className="px-3 py-2 text-left font-medium">Actual \ Predicted</th>
              {labels.map((label) => (
                <th key={label} className="px-3 py-2 font-mono font-medium">
                  {label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {confusionMatrix.map((row, i) => {
              const rowMax = Math.max(...row, 0);
              return (
                <tr key={labels[i]} className="border-b border-rule last:border-b-0">
                  <th scope="row" className="px-3 py-2 text-left font-mono text-xs text-muted">
                    {labels[i]}
                  </th>
                  {row.map((count, j) => (
                    <td
                      key={labels[j]}
                      className={
                        "px-3 py-2 font-mono " +
                        scaleClassForCell(count, rowMax) +
                        (i === j ? " font-semibold" : "")
                      }
                    >
                      {count}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-1 font-mono text-xs text-muted">
        {correct} of {total} test samples correctly classified.
      </p>
    </div>
  );
}
