import type { ReactNode } from "react";

interface ScreenPanelProps {
  children: ReactNode;
  maxWidthClassName?: string;
}

/**
 * Shared width for data-dense screens such as:
 * EDA, Model Selection, Results and Compare.
 */
export const WORKSPACE_WIDTH = "max-w-[1400px]";

/**
 * Shared outer card used by all screens.
 *
 * Default:
 * - Focused screens: 900px
 *
 * Wide:
 * - EDA
 * - Model Selection
 * - Results
 * - Compare
 *
 * Individual screens can still override the width when necessary.
 */
export function ScreenPanel({
  children,
  maxWidthClassName = "max-w-[900px]",
}: ScreenPanelProps) {
  return (
    <section
      className={`
        mx-auto
        w-full
        ${maxWidthClassName}
        rounded-panel
        border
        border-rule
        bg-surface
        px-8
        py-7
        text-ink
        shadow-sm
      `}
    >
      {children}
    </section>
  );
}