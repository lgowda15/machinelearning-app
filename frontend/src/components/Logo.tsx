interface LogoProps {
  // "mark" -- the small corner logo every screen's header carries. "full"
  // -- the full logo plus the "Celebrating 50 Years" banner, Start screen
  // only (frontend.md). There is one source asset (public/assets/logo.png,
  // compass mark stacked over the wordmark and banner); "mark" crops to
  // just the compass by covering a square box anchored to the image's top,
  // rather than shipping a second cropped asset to keep in sync.
  variant: "mark" | "full";
  className?: string;
}

export function Logo({ variant, className = "" }: LogoProps) {
  if (variant === "mark") {
    return (
      <div className={`aspect-square shrink-0 overflow-hidden rounded-panel ${className}`}>
        <img
          src="/assets/logo.png"
          alt="PES University"
          className="h-full w-full object-cover object-top"
        />
      </div>
    );
  }

  return (
    <img
      src="/assets/logo.png"
      alt="PES University — Celebrating 50 Years"
      className={className}
    />
  );
}
