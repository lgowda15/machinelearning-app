import { useState } from "react";
import type { CycleState } from "../lib/report/types";

interface DownloadReportButtonProps {
  cycle: CycleState;
}

/** `report_<dataset-name>_<YYYYMMDD-HHmm>.pdf` (docs/plans/pdf-report.md's
 * "Filename" design decision). Strips path separators/extension from the
 * dataset name and replaces anything else unsafe for a filesystem. */
function reportFilename(datasetName: string, now: Date): string {
  const stem = datasetName
    .replace(/^.*[/\\]/, "")
    .replace(/\.[^.]+$/, "")
    .replace(/[^a-zA-Z0-9]+/g, "_");

  const pad = (n: number) => String(n).padStart(2, "0");
  const stamp = `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}-${pad(now.getHours())}${pad(now.getMinutes())}`;

  return `report_${stem}_${stamp}.pdf`;
}

/**
 * One button in StepShell's header, enabled once a training run exists
 * (docs/plans/pdf-report.md's "Download button placement" design decision)
 * -- it reads whatever is currently in the lifted cycle state regardless of
 * which screen is mounted. jspdf/svg2pdf.js/jspdf-autotable/react-dom are
 * dynamically imported here so they never ship in the main bundle.
 */
export function DownloadReportButton({ cycle }: DownloadReportButtonProps) {
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const enabled = cycle.trainingResults !== null && !generating;

  const handleClick = async () => {
    setGenerating(true);
    setError(null);

    try {
      const [{ buildReportModel }, { renderReportPdf }] = await Promise.all([
        import("../lib/report/buildReportModel"),
        import("../lib/report/renderReportPdf"),
      ]);

      const model = buildReportModel(cycle);
      const blob = await renderReportPdf(model);

      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = reportFilename(cycle.profile?.source ?? "report", new Date());
      link.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setGenerating(false);
    }
  };

  return (
    <div className="flex flex-col items-end gap-1">
      <button
        type="button"
        onClick={handleClick}
        disabled={!enabled}
        className={`
          rounded-panel
          border
          px-3
          py-1.5
          font-mono
          text-xs
          uppercase
          tracking-wide
          transition-colors
          duration-150
          ${
            enabled
              ? "cursor-pointer border-white/40 text-white hover:border-signal hover:text-signal"
              : "cursor-not-allowed border-white/15 text-white/35"
          }
        `}
      >
        {generating ? "Generating…" : "Download report"}
      </button>

      {error && (
        <p className="max-w-56 text-right text-[10px] text-signal">
          Report failed: {error.message}
        </p>
      )}
    </div>
  );
}
