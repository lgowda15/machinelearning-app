import type { TreeNode, TreeStructurePayload } from "../../types/visualizationData";

const MAX_DEPTH = 3;
const MAX_RULES = 8;

/**
 * Builds the "first 3 levels of splits" text summary for a tree_structure
 * payload (docs/plans/pdf-report.md's "Tree chart" design decision -- no
 * SVG conversion). Each rule is one root-to-frontier path, arrow-joined,
 * e.g. "feature_3 <= 0.42 -> feature_1 <= -0.8 -> class 1": a leaf reached
 * within the first 3 levels ends the rule in its predicted class; a path
 * still inside an internal node at depth 3 ends in that node's own
 * condition instead, since the walk stops there rather than assuming a
 * binary tree with a single leftmost path (CHAID/ID3 splits can have more
 * than two children).
 */
export function buildTreeSummaryRules(tree: TreeStructurePayload): string[] {
  const nodesById = new Map(tree.nodes.map((node) => [node.id, node]));
  const root = nodesById.get(tree.root_id);
  if (!root) return [];

  const rules: string[] = [];

  function describe(node: TreeNode): string {
    return node.is_leaf ? `class ${node.predicted_class}` : (node.split?.condition ?? `node ${node.id}`);
  }

  function childrenOf(node: TreeNode): TreeNode[] {
    return node.children
      .map((id) => nodesById.get(id))
      .filter((child): child is TreeNode => child !== undefined);
  }

  function walk(node: TreeNode, path: string[], depth: number): void {
    if (rules.length >= MAX_RULES) return;

    const nextPath = [...path, describe(node)];

    if (node.is_leaf || depth === MAX_DEPTH) {
      rules.push(nextPath.join(" → "));
      return;
    }

    for (const child of childrenOf(node)) {
      walk(child, nextPath, depth + 1);
    }
  }

  walk(root, [], 1);
  return rules.slice(0, MAX_RULES);
}
