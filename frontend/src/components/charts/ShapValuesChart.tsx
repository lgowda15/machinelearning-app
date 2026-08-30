import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { AXIS_TICK, TOOLTIP_STYLE } from "./chartTheme";
import { typeColorVar } from "../../lib/modelType";
import type { ShapValues } from "../../types/visualizationData";

interface ShapValuesChartProps {
  shapValues: ShapValues;
  /** From the same result's `feature_importance` keys, or `metrics.labels`
   * for classes -- `shap_values` itself carries no names (see
   * types/visualizationData.ts), just the raw per-sample arrays. */
  featureNames: string[];
  classLabels: string[];
}

/** Screen 5's model-specific visual for group_02 (Random Forest, XGBoost)
 * (CLAUDE.md "Known gap", DATA_FLOW_GUIDE.md §5.3). A per-feature mean
 * absolute SHAP value, the standard SHAP "global importance" bar summary
 * -- distinct from `feature_importance` (impurity-based, one number per
 * feature) which already renders above this via FeatureImportanceChart;
 * this one is model-explanation-based and, for a multiclass model, shows
 * one bar per class per feature rather than collapsing classes away.
 * Classifier blue (--signal) throughout, per frontend.md's colour coding
 * -- group_02 ships only classifiers -- with per-class bars distinguished
 * by opacity rather than a new hue, since frontend.md reserves colour
 * assignment for model type, not for series within one model's own chart.
 */
export function ShapValuesChart({ shapValues, featureNames, classLabels }: ShapValuesChartProps) {
  const is3D = Array.isArray(shapValues[0]?.[0]);
  const nFeatures = is3D ? (shapValues as number[][][])[0].length : (shapValues as number[][])[0].length;
  const nClasses = is3D ? (shapValues as number[][][])[0][0].length : 1;
  const nSamples = shapValues.length;

  // Mean |SHAP value| per feature, per class, across every sample.
  const meanAbs: number[][] = Array.from({ length: nFeatures }, () => new Array(nClasses).fill(0));
  for (const sample of shapValues) {
    for (let f = 0; f < nFeatures; f++) {
      if (is3D) {
        const perClass = (sample as number[][])[f];
        for (let c = 0; c < nClasses; c++) meanAbs[f][c] += Math.abs(perClass[c]);
      } else {
        meanAbs[f][0] += Math.abs((sample as number[])[f]);
      }
    }
  }
  for (const row of meanAbs) {
    for (let c = 0; c < nClasses; c++) row[c] /= nSamples;
  }

  const rows = meanAbs
    .map((perClass, f) => {
      const row: Record<string, string | number> = { feature: featureNames[f] ?? `feature_${f}` };
      let total = 0;
      perClass.forEach((value, c) => {
        row[`class_${c}`] = value;
        total += value;
      });
      row.__total = total;
      return row;
    })
    .sort((a, b) => (b.__total as number) - (a.__total as number));

  const topFeature = rows[0];
  const color = typeColorVar("classifier");

  return (
    <div>
      <div className="h-56">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 16, left: 4, bottom: 4 }}>
            <CartesianGrid stroke="var(--color-rule)" horizontal={false} />
            <XAxis type="number" tick={AXIS_TICK} />
            <YAxis type="category" dataKey="feature" tick={AXIS_TICK} width={90} />
            <Tooltip {...TOOLTIP_STYLE} cursor={{ fill: "var(--color-rule)", opacity: 0.3 }} />
            {nClasses > 1 && <Legend wrapperStyle={{ fontSize: 10, fontFamily: "var(--font-mono)" }} />}
            {Array.from({ length: nClasses }, (_, c) => (
              <Bar
                key={c}
                dataKey={`class_${c}`}
                name={classLabels[c] ?? `class ${c}`}
                fill={color}
                fillOpacity={nClasses > 1 ? 0.35 + 0.65 * (c / Math.max(nClasses - 1, 1)) : 1}
                radius={0}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="mt-1 font-mono text-xs text-muted">
        Mean |SHAP value| across {nSamples} test samples
        {nClasses > 1 ? `, ${nClasses} classes` : ""}. Top feature: {topFeature?.feature as string}.
      </p>
    </div>
  );
}
