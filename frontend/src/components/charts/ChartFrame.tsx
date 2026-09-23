import { cloneElement, type ReactElement } from "react";
import { ResponsiveContainer } from "recharts";

interface ChartFrameProps {
  className?: string;
  /** Both set together (docs/plans/pdf-report.md's `captureChartSvg`)
   * bypasses ResponsiveContainer entirely: it measures its parent via
   * ResizeObserver, whose callback fires asynchronously and never resolves
   * in time for a synchronous off-screen capture, even when the parent
   * already has explicit pixel dimensions. On-screen (the normal case,
   * both undefined) keeps the existing ResponsiveContainer behavior. */
  width?: number;
  height?: number;
  children: ReactElement<{ width?: number; height?: number }>;
}

/** Shared wrapper so each Recharts-based chart doesn't reimplement the
 * on-screen/off-screen branch itself -- see chartTheme.ts for this file's
 * companion shared styling constants. */
export function ChartFrame({ className, width, height, children }: ChartFrameProps) {
  if (width !== undefined && height !== undefined) {
    return (
      <div className={className} style={{ width, height }}>
        {cloneElement(children, { width, height })}
      </div>
    );
  }

  return (
    <div className={className}>
      <ResponsiveContainer width="100%" height="100%">
        {children}
      </ResponsiveContainer>
    </div>
  );
}
