interface ScreenHeaderProps {
  title: string;
  description: string;
}

/**
 * Single compact header line for a step screen -- StepShell's stepper
 * already names the current step, so this doesn't repeat a "Step N ·
 * Category" eyebrow, just the screen's own title and a one-line
 * description, collapsed onto the same line.
 */
export function ScreenHeader({ title, description }: ScreenHeaderProps) {
  return (
    <div className="mb-4 flex shrink-0 flex-wrap items-baseline gap-x-3 gap-y-1 border-b border-rule pb-3">
      <h1 className="text-base font-semibold text-ink">{title}</h1>
      <p className="text-xs text-muted">{description}</p>
    </div>
  );
}
