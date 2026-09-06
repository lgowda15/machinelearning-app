import { Logo } from "../Logo";

interface StartScreenProps {
  onBegin: () => void;
}

export function StartScreen({ onBegin }: StartScreenProps) {
  return (
    <section className="grid h-screen w-full grid-cols-1 overflow-hidden lg:grid-cols-2">
      {/* LEFT — Branding */}
      <div className="relative flex flex-col justify-between overflow-hidden bg-brand-navy px-10 py-10 text-white sm:px-14 lg:px-16">
        {/* Decorative circles */}
        <div className="pointer-events-none absolute -right-32 -top-32 h-80 w-80 rounded-full border border-white/10" />
        <div className="pointer-events-none absolute -bottom-40 -left-32 h-96 w-96 rounded-full border border-white/10" />
        <div className="pointer-events-none absolute right-20 top-32 h-24 w-24 rounded-full bg-brand-gold/10" />

        <div className="relative z-10">
          {/* PES branding */}
          <div className="flex items-center gap-3">
            <Logo variant="mark" className="h-11 w-11" />

            <div>
              <p className="text-sm font-semibold tracking-wide text-white">
                PES UNIVERSITY
              </p>

              <p className="font-mono text-[9px] uppercase tracking-[0.2em] text-white/45">
                Machine Learning
              </p>
            </div>
          </div>

          {/* Main heading */}
          <div className="mt-[18vh] max-w-[650px]">
            <p className="font-mono text-xs uppercase tracking-[0.22em] text-brand-gold">
              ML Integration Platform
            </p>

            <h1 className="mt-5 text-5xl font-semibold leading-[1.05] tracking-tight sm:text-6xl xl:text-7xl">
              Turn your data
              <br />
              into <span className="text-brand-gold">insights.</span>
            </h1>

            <p className="mt-8 max-w-[560px] text-base leading-7 text-white/65 sm:text-lg">
              Upload your dataset, explore its structure, train multiple
              machine-learning models, and compare their results — all in one
              place.
            </p>
          </div>
        </div>

        {/* Bottom label */}
        <div className="relative z-10 flex items-center gap-3">
          <div className="h-px w-12 bg-brand-gold" />

          <span className="font-mono text-[10px] uppercase tracking-[0.2em] text-white/40">
            Explore · Train · Compare
          </span>
        </div>
      </div>

      {/* RIGHT — Welcome */}
      <div className="flex items-center overflow-y-auto bg-surface px-10 py-12 sm:px-14 lg:px-16 xl:px-24">
        <div className="w-full max-w-[560px]">
          {/* Original PES logo */}
          <div className="flex items-center gap-4">
            <Logo variant="mark" className="h-16 w-16" />

            <div>
              <p className="text-2xl font-semibold tracking-tight text-ink">
                PES
              </p>

              <p className="text-xs font-medium uppercase tracking-[0.18em] text-muted">
                University
              </p>
            </div>
          </div>

          {/* Welcome text */}
          <div className="mt-14">
            <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-muted">
              Welcome
            </p>

            <h2 className="mt-3 text-4xl font-semibold tracking-tight text-ink sm:text-5xl">
              Build your ML workflow.
            </h2>

            <p className="mt-6 max-w-[520px] text-base leading-7 text-muted">
              Start by uploading a dataset or choosing one of the available
              sample datasets. The platform will guide you through
              exploration, model selection, training, results, prediction,
              and comparison.
            </p>
          </div>

          {/* Workflow */}
          <div className="mt-10 grid grid-cols-3 gap-3">
            {[
              ["01", "Upload"],
              ["02", "Train"],
              ["03", "Compare"],
            ].map(([number, label]) => (
              <div
                key={number}
                className="rounded-panel border border-rule bg-ground px-4 py-4"
              >
                <p className="font-mono text-[10px] text-signal">
                  {number}
                </p>

                <p className="mt-2 text-sm font-medium text-ink">
                  {label}
                </p>
              </div>
            ))}
          </div>

          {/* Begin */}
          <button
            type="button"
            onClick={onBegin}
            className="mt-8 flex w-full items-center justify-center gap-3 rounded-panel bg-signal px-6 py-4 text-sm font-medium text-white transition-all duration-150 hover:-translate-y-0.5 hover:opacity-90 hover:shadow-md active:translate-y-0"
          >
            <span>Begin</span>
            <span aria-hidden="true">→</span>
          </button>

          <p className="mt-5 text-center font-mono text-[9px] uppercase tracking-[0.15em] text-muted">
            No model preference · Data-driven selection
          </p>
        </div>
      </div>
    </section>
  );
}