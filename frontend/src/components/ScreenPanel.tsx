import type { ReactNode } from "react";

interface ScreenPanelProps {
  children: ReactNode;
}

/**
 * Shared outer panel for every step screen. Fills 100% of the shell's main
 * row (frontend.md's Layout section) -- no centred, width-capped card.
 *
 * `overflow-y-auto` here is a whole-panel scroll fallback, not a deliberate
 * design choice: it's what keeps a screen usable before its own layout
 * gives it a more precise single scroll region (CLAUDE.md's layout rule).
 * Screens built after that migration manage their own scroll region instead
 * and should render inside a non-scrolling wrapper here, not this one.
 */
export function ScreenPanel({ children }: ScreenPanelProps) {
  return (
    <section className="flex h-full w-full flex-col overflow-y-auto rounded-panel border border-rule bg-surface px-8 py-6 text-ink shadow-sm">
      {children}
    </section>
  );
}