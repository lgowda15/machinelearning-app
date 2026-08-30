import { typeColorVar } from "../../lib/modelType";
import type { LinkageMatrix } from "../../types/visualizationData";

interface DendrogramChartProps {
  linkageMatrix: LinkageMatrix;
}

const PLOT_HEIGHT = 220;
const LEAF_SPACING = 18;
const MARGIN = { top: 8, right: 16, bottom: 20, left: 48 };
/** Above this many leaves, per-leaf tick labels would overlap into
 * illegibility, so they're dropped in favour of the sample-count caption. */
const MAX_LABELLED_LEAVES = 40;

/** Screen 5's model-specific visual for group_09's Hierarchical Clustering
 * (CLAUDE.md "Known gap", DATA_FLOW_GUIDE.md §5.3). Not one of Recharts'
 * chart types, so this draws the classic elbow-connector dendrogram
 * directly in SVG from the raw linkage matrix -- SciPy's own output
 * contract (see types/visualizationData.ts), one row per merge:
 * `[idx1, idx2, distance, sample_count]`. Clusterer teal
 * (--type-cluster), per frontend.md's model-type colour coding.
 */
export function DendrogramChart({ linkageMatrix }: DendrogramChartProps) {
  const nLeaves = linkageMatrix.length + 1;
  const maxDistance = Math.max(...linkageMatrix.map((row) => row[2]), 0) || 1;
  const color = typeColorVar("clusterer");

  // Leaf x-order: recurse through the merge tree so sibling subtrees never
  // cross, the standard dendrogram leaf-ordering rule. Memoized since a
  // node deep in the tree is revisited once per ancestor merge otherwise.
  const orderCache = new Map<number, number[]>();
  function leafOrder(nodeId: number): number[] {
    if (nodeId < nLeaves) return [nodeId];
    const cached = orderCache.get(nodeId);
    if (cached) return cached;
    const row = linkageMatrix[nodeId - nLeaves];
    const order = [...leafOrder(row[0]), ...leafOrder(row[1])];
    orderCache.set(nodeId, order);
    return order;
  }
  const rootId = nLeaves + linkageMatrix.length - 1;
  const order = leafOrder(rootId);
  const leafX = new Map(order.map((leafId, slot) => [leafId, slot]));

  // x of every node (leaf or merge), bottom-up; height of every node,
  // leaves at 0, merges at their linkage distance.
  const x = new Map<number, number>(leafX);
  const height = new Map<number, number>(order.map((leafId) => [leafId, 0]));
  linkageMatrix.forEach((row, i) => {
    const nodeId = nLeaves + i;
    const [left, right, distance] = row;
    x.set(nodeId, (x.get(left)! + x.get(right)!) / 2);
    height.set(nodeId, distance);
  });

  const plotWidth = (nLeaves - 1) * LEAF_SPACING;
  const width = plotWidth + MARGIN.left + MARGIN.right;
  const svgHeight = PLOT_HEIGHT + MARGIN.top + MARGIN.bottom;
  const px = (nodeId: number) => MARGIN.left + x.get(nodeId)! * LEAF_SPACING;
  const py = (dist: number) => MARGIN.top + PLOT_HEIGHT - (dist / maxDistance) * PLOT_HEIGHT;

  const yTicks = [0, 0.25, 0.5, 0.75, 1].map((f) => f * maxDistance);
  const showLeafTicks = nLeaves <= MAX_LABELLED_LEAVES;

  return (
    <div>
      <div className="overflow-x-auto rounded-panel border border-rule bg-surface p-2">
        <svg width={width} height={svgHeight} role="img" aria-label={`Dendrogram of ${nLeaves} samples`}>
          {yTicks.map((tick) => (
            <g key={tick}>
              <line
                x1={MARGIN.left}
                x2={width - MARGIN.right}
                y1={py(tick)}
                y2={py(tick)}
                stroke="var(--color-rule)"
              />
              <text
                x={MARGIN.left - 6}
                y={py(tick)}
                textAnchor="end"
                dominantBaseline="middle"
                fontFamily="var(--font-mono)"
                fontSize={9}
                fill="var(--color-muted)"
              >
                {tick.toFixed(1)}
              </text>
            </g>
          ))}
          {linkageMatrix.map((row, i) => {
            const nodeId = nLeaves + i;
            const [left, right] = row;
            const mergeY = py(height.get(nodeId)!);
            return (
              <g key={nodeId}>
                <line x1={px(left)} x2={px(left)} y1={py(height.get(left)!)} y2={mergeY} stroke={color} />
                <line x1={px(right)} x2={px(right)} y1={py(height.get(right)!)} y2={mergeY} stroke={color} />
                <line x1={px(left)} x2={px(right)} y1={mergeY} y2={mergeY} stroke={color} />
              </g>
            );
          })}
          {showLeafTicks &&
            order.map((leafId) => (
              <text
                key={leafId}
                x={px(leafId)}
                y={MARGIN.top + PLOT_HEIGHT + 12}
                textAnchor="middle"
                fontFamily="var(--font-mono)"
                fontSize={8}
                fill="var(--color-muted)"
              >
                {leafId}
              </text>
            ))}
        </svg>
      </div>
      <p className="mt-1 font-mono text-xs text-muted">
        {nLeaves} samples, {linkageMatrix.length} merges, max distance {maxDistance.toFixed(3)}
        {!showLeafTicks && " (leaf labels omitted above 40 samples)"}.
      </p>
    </div>
  );
}
