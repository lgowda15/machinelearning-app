import type { TreeNode, TreeStructurePayload } from "../../types/visualizationData";

interface TreeStructureChartProps {
  data: TreeStructurePayload;
}

/** Screen 5's model-specific visual for group_01's four tree algorithms
 * (CLAUDE.md "Known gap", DATA_FLOW_GUIDE.md §5.3). Rendered last in the
 * results panel, alongside -- not instead of -- the standard metrics block
 * and feature importance chart.
 *
 * A decision tree isn't one of Recharts' chart types, so this lays the
 * tree out with nested flexbox rather than a charting library: each node
 * is a card, its children sit in a column to its right behind a rule,
 * labelled with the branch condition that leads to them. That works
 * unchanged across all four algorithms' split types (binary, oblique,
 * multiway) since `children` and `edges[].label` are already
 * ready-to-display regardless of split type. Classifier blue (--signal),
 * per frontend.md's model-type colour coding -- trees are always
 * classifiers here (group_01 ships none of the other three model types).
 */
export function TreeStructureChart({ data }: TreeStructureChartProps) {
  const nodesById = new Map(data.nodes.map((node) => [node.id, node]));
  const edgeLabelByTarget = new Map(data.edges.map((edge) => [edge.target, edge.label]));
  const root = nodesById.get(data.root_id);

  if (!root) {
    return <p className="text-sm text-muted">Tree structure has no root node.</p>;
  }

  return (
    <div>
      <div className="overflow-x-auto rounded-panel border border-rule bg-surface p-4">
        <NodeSubtree node={root} nodesById={nodesById} edgeLabelByTarget={edgeLabelByTarget} />
      </div>
      <p className="mt-1 font-mono text-xs text-muted">
        {data.algorithm}: {data.n_nodes} nodes, {data.n_leaves} leaves, depth {data.max_depth_reached},{" "}
        {data.impurity_measure} impurity.
      </p>
    </div>
  );
}

function NodeSubtree({
  node,
  nodesById,
  edgeLabelByTarget,
}: {
  node: TreeNode;
  nodesById: Map<number, TreeNode>;
  edgeLabelByTarget: Map<number, string>;
}) {
  const children = node.children
    .map((id) => nodesById.get(id))
    .filter((child): child is TreeNode => child != null);

  return (
    <div className="flex items-start gap-4">
      <NodeCard node={node} />
      {children.length > 0 && (
        <div className="flex flex-col gap-3 border-l border-rule pl-4">
          {children.map((child) => (
            <div key={child.id} className="flex flex-col items-start gap-1">
              <span className="rounded-panel bg-surface-alt px-1.5 py-0.5 font-mono text-[10px] text-muted">
                {edgeLabelByTarget.get(child.id) ?? ""}
              </span>
              <NodeSubtree node={child} nodesById={nodesById} edgeLabelByTarget={edgeLabelByTarget} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function NodeCard({ node }: { node: TreeNode }) {
  const topProbability = Math.max(...node.class_probabilities);

  return (
    <div
      className={
        "w-44 shrink-0 rounded-panel border p-2 text-xs " +
        (node.is_leaf ? "border-signal bg-surface" : "border-rule bg-surface-alt")
      }
    >
      {node.is_leaf ? (
        <p className="font-medium text-signal">class {String(node.predicted_class)}</p>
      ) : (
        <p className="break-words font-mono text-ink">{node.split?.condition}</p>
      )}
      <p className="mt-1 font-mono text-muted">
        n={node.n_samples} · {node.impurity_measure}={node.impurity.toFixed(3)}
      </p>
      {node.is_leaf && (
        <p className="font-mono text-muted">p={topProbability.toFixed(2)}</p>
      )}
      {!node.is_leaf && node.split?.chi_square != null && (
        <p className="font-mono text-muted">
          χ²={node.split.chi_square.toFixed(2)}
          {node.split.p_value != null && ` p=${node.split.p_value.toFixed(3)}`}
        </p>
      )}
    </div>
  );
}
