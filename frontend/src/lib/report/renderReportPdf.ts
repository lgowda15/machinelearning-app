import { jsPDF } from "jspdf";
import autoTable from "jspdf-autotable";
import { svg2pdf } from "svg2pdf.js";
import { createElement } from "react";
import { ClusterScatterChart } from "../../components/charts/ClusterScatterChart";
import { DendrogramChart } from "../../components/charts/DendrogramChart";
import { FeatureImportanceChart } from "../../components/charts/FeatureImportanceChart";
import { PredictedVsActualChart } from "../../components/charts/PredictedVsActualChart";
import { ShapValuesChart } from "../../components/charts/ShapValuesChart";
import { VariancePlotChart } from "../../components/charts/VariancePlotChart";
import { DistributionChart } from "../../components/DistributionChart";
import { captureChartSvg } from "./captureChartSvg";
import { formatMetricValue } from "../format";
import type { ChartSpec, ReportModel } from "./types";

const PAGE_MARGIN = 40;
const CHART_WIDTH = 460;
const CHART_HEIGHT = 220;
const DENDROGRAM_CAPTURE_WIDTH = 900;
const DENDROGRAM_CAPTURE_HEIGHT = 320;

/**
 * Assembles the PDF report from an already-built `ReportModel`. Dynamically
 * imported on click (docs/plans/pdf-report.md) so jspdf/svg2pdf.js/
 * jspdf-autotable/react-dom never ship in the main bundle.
 */
export async function renderReportPdf(model: ReportModel): Promise<Blob> {
  const doc = new jsPDF({ unit: "pt", format: "a4" });
  const pageWidth = doc.internal.pageSize.getWidth();
  const pageHeight = doc.internal.pageSize.getHeight();
  let cursorY = PAGE_MARGIN;

  function ensureSpace(height: number): void {
    if (cursorY + height > pageHeight - PAGE_MARGIN) {
      doc.addPage();
      cursorY = PAGE_MARGIN;
    }
  }

  function heading(text: string): void {
    ensureSpace(28);
    doc.setFontSize(14);
    doc.text(text, PAGE_MARGIN, cursorY);
    cursorY += 20;
    doc.setFontSize(10);
  }

  function paragraph(text: string): void {
    const lines: string[] = doc.splitTextToSize(text, pageWidth - PAGE_MARGIN * 2);
    ensureSpace(lines.length * 12 + 8);
    doc.text(lines, PAGE_MARGIN, cursorY);
    cursorY += lines.length * 12 + 8;
  }

  function table(headers: string[], rows: (string | number)[][]): void {
    autoTable(doc, {
      startY: cursorY,
      margin: { left: PAGE_MARGIN, right: PAGE_MARGIN },
      head: [headers],
      body: rows,
      styles: { fontSize: 8, cellPadding: 4 },
      headStyles: { fillColor: [230, 230, 230], textColor: [20, 20, 20] },
    });
    cursorY = (doc as unknown as { lastAutoTable: { finalY: number } }).lastAutoTable.finalY + 16;
  }

  async function renderChart(spec: ChartSpec): Promise<void> {
    if (spec.kind === "confusion-matrix") {
      paragraph(spec.title);
      table(["", ...spec.labels], spec.confusionMatrix.map((row, i) => [spec.labels[i] ?? String(i), ...row]));
      return;
    }

    if (spec.kind === "tree-summary") {
      paragraph(spec.title);
      paragraph(`Depth ${spec.depth} · ${spec.nodeCount} nodes · ${spec.leafCount} leaves`);
      for (const rule of spec.rules) {
        paragraph(`  ${rule}`);
      }
      return;
    }

    const captureWidth = spec.kind === "dendrogram" ? DENDROGRAM_CAPTURE_WIDTH : CHART_WIDTH;
    const captureHeight = spec.kind === "dendrogram" ? DENDROGRAM_CAPTURE_HEIGHT : CHART_HEIGHT;
    const captured = captureChartSvg(chartElement(spec, captureWidth, captureHeight), captureWidth, captureHeight);

    const renderWidth = pageWidth - PAGE_MARGIN * 2;
    const renderHeight = (renderWidth / captured.width) * captured.height;

    ensureSpace(renderHeight + 24);
    doc.setFontSize(10);
    doc.text(spec.title, PAGE_MARGIN, cursorY);
    cursorY += 14;

    await svg2pdf(captured.element, doc, {
      x: PAGE_MARGIN,
      y: cursorY,
      width: renderWidth,
      height: renderHeight,
    });
    cursorY += renderHeight + 16;
  }

  // -- Summary --------------------------------------------------------------
  doc.setFontSize(18);
  doc.text("Model report", PAGE_MARGIN, cursorY);
  cursorY += 26;
  doc.setFontSize(10);

  table(
    ["Field", "Value"],
    [
      ["Dataset", model.summary.datasetName],
      ["Rows × columns", `${model.summary.nRows} × ${model.summary.nCols}`],
      ["Target", model.summary.target ?? "—"],
      ["Data type", model.summary.dataType],
      ["Test split", `${Math.round(model.summary.testSize * 100)}%`],
      ["Models", model.summary.modelDisplayNames.join(", ")],
      ["Generated", model.summary.generatedAt],
    ],
  );

  // -- Preprocessing ----------------------------------------------------------
  heading("Preprocessing");
  paragraph(
    "Every dataset is imputed (median for numeric columns, mode for categorical), " +
      "encoded, and scaled to float64 before any model sees it. Every fit -- imputer, " +
      "encoder, scaler -- happens on the training split only; the test split is only " +
      "ever transformed, never fit on.",
  );

  // -- Dataset ----------------------------------------------------------------
  heading("Dataset");

  if (model.dataset.imbalance) {
    paragraph(
      model.dataset.imbalance.isImbalanced
        ? (model.dataset.imbalance.message ?? "This dataset's target classes are imbalanced.")
        : "This dataset's target classes are not significantly imbalanced.",
    );
  }

  table(
    ["Column", "Type", "Missing %", "Unique values"],
    model.dataset.columns.map((column) => [
      column.is_target ? `${column.name} (target)` : column.name,
      column.dtype,
      `${(column.missing_pct * 100).toFixed(1)}%`,
      column.unique_count,
    ]),
  );

  if (model.dataset.truncatedColumnsCount > 0) {
    paragraph(`${model.dataset.truncatedColumnsCount} more column(s) omitted.`);
  }

  for (const chart of model.dataset.distributionCharts) {
    await renderChart(chart);
  }

  // -- Models -------------------------------------------------------------
  for (const reportModel of model.models) {
    heading(`${reportModel.displayName} (${reportModel.modelType})`);

    if (Object.keys(reportModel.hyperparameters).length > 0) {
      table(
        ["Hyperparameter", "Value"],
        Object.entries(reportModel.hyperparameters).map(([name, value]) => [name, formatMetricValue(value)]),
      );
    }

    const scalarMetrics = Object.entries(reportModel.metrics).filter(([, value]) => typeof value === "number");
    if (scalarMetrics.length > 0) {
      table(
        ["Metric", "Value"],
        scalarMetrics.map(([name, value]) => [name, formatMetricValue(value)]),
      );
    }

    for (const chart of reportModel.charts) {
      await renderChart(chart);
    }
  }

  // -- Comparison ---------------------------------------------------------
  if (model.comparison) {
    heading("Comparison");
    table(model.comparison.metricsTable.headers, model.comparison.metricsTable.rows);
  }

  // -- Prediction -----------------------------------------------------------
  if (model.prediction) {
    heading("Prediction");
    paragraph(`Model: ${model.prediction.modelKey} · Input: ${model.prediction.inputSummary}`);
    paragraph(`${model.prediction.totalRows} row(s) predicted.`);
    table(model.prediction.outputSample.headers, model.prediction.outputSample.rows);
    paragraph("Predicted value distribution");
    table(model.prediction.predictedValueDistribution.headers, model.prediction.predictedValueDistribution.rows);
  }

  return doc.output("blob");
}

/* -------------------------------------------------------------------------- */
/* Chart element dispatch                                                    */
/* -------------------------------------------------------------------------- */

function chartElement(spec: ChartSpec, width: number, height: number) {
  const common = { width, height, isAnimationActive: false };

  switch (spec.kind) {
    case "distribution":
      return createElement(DistributionChart, { column: spec.column, targetAccentVar: spec.targetAccentVar, ...common });
    case "feature-importance":
      return createElement(FeatureImportanceChart, { featureImportance: spec.featureImportance, ...common });
    case "cluster-scatter":
      return createElement(ClusterScatterChart, { points: spec.points, labels: spec.labels, ...common });
    case "predicted-vs-actual":
      return createElement(PredictedVsActualChart, { yTrue: spec.yTrue, yPred: spec.yPred, ...common });
    case "variance-plot":
      return createElement(VariancePlotChart, { explainedVarianceRatio: spec.explainedVarianceRatio, ...common });
    case "shap-values":
      return createElement(ShapValuesChart, {
        shapValues: spec.shapValues,
        featureNames: spec.featureNames,
        classLabels: spec.classLabels,
        ...common,
      });
    case "dendrogram":
      return createElement(DendrogramChart, { linkageMatrix: spec.linkageMatrix });
    default:
      throw new Error(`renderChart: unreachable chart kind "${(spec as ChartSpec).kind}"`);
  }
}
