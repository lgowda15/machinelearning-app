import type { ReactNode } from "react";

interface ScreenPanelProps {
  children: ReactNode;
  maxWidthClassName?: string;
}

/** The data-dense screens' shared width (EDA, Model selection, Results,
 * Compare -- frontend.md's Layout section). One constant so the four
 * screens can't drift from each other. */
export const WORKSPACE_WIDTH = "max-w-[1200px]";

/** Every screen's outer card. Default width is the "focused, single-decision
 * screen" size (frontend.md's Layout section, max 720px) -- Start, Training
 * and Predict use it as-is; the data-dense screens (EDA, Model selection,
 * Results, Compare) pass the wider workspace width explicitly, and Upload
 * passes its own narrower 640px per its screen contract. */
export function ScreenPanel({ children, maxWidthClassName = "max-w-[720px]" }: ScreenPanelProps) {
  return (
    <section className={`mx-auto ${maxWidthClassName} rounded-panel border border-rule bg-surface p-6 text-ink`}>
      {children}
    </section>
  );
}
