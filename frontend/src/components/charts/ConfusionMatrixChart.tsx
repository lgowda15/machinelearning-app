interface ConfusionMatrixChartProps {
  confusionMatrix: number[][];
  labels: string[];
}

/** Screen 5's classifier chart (frontend.md). Rows are actual classes,
 * columns predicted -- the diagonal (a correct call) is the one place this
 * table spends colour, in --signal -- which is also the classifier's
 * model-type accent (frontend.md's colour coding defines the two as the
 * same hex), so this doubles as "the thing in focus" and "this model's
 * type colour" at once, rather than a heatmap gradient ("no gradients"). */
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
            {confusionMatrix.map((row, i) => (
              <tr key={labels[i]} className="border-b border-rule last:border-b-0">
                <th scope="row" className="px-3 py-2 text-left font-mono text-xs text-muted">
                  {labels[i]}
                </th>
                {row.map((count, j) => (
                  <td
                    key={labels[j]}
                    className={"px-3 py-2 font-mono " + (i === j ? "font-medium text-signal" : "text-ink")}
                  >
                    {count}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-1 font-mono text-xs text-muted">
        {correct} of {total} test samples correctly classified.
      </p>
    </div>
  );
}
