interface StartScreenProps {
  onBegin: () => void;
}

export function StartScreen({ onBegin }: StartScreenProps) {
  return (
    <section className="grid h-screen w-full grid-cols-1 overflow-hidden lg:grid-cols-2">
      {/* ============================================================= */}
      {/* LEFT — BRANDING                                               */}
      {/* ============================================================= */}

      <div className="relative flex h-screen flex-col justify-between overflow-hidden bg-brand-navy px-8 py-8 text-white sm:px-12 lg:px-14">
        {/* Decorative circles */}
        <div className="pointer-events-none absolute -right-32 -top-32 h-80 w-80 rounded-full border border-white/10" />

        <div className="pointer-events-none absolute -bottom-40 -left-32 h-96 w-96 rounded-full border border-white/10" />

        <div className="pointer-events-none absolute right-20 top-32 h-24 w-24 rounded-full bg-brand-gold/10" />

        <div className="relative z-10">
          {/* Main heading */}
          <div className="mt-[20vh] max-w-[650px]">
            <p className="font-mono text-[10px] uppercase tracking-[0.22em] text-brand-gold">
              ML Integration Platform
            </p>

            <h1 className="mt-4 text-4xl font-semibold leading-[1.02] tracking-tight sm:text-5xl xl:text-6xl">
              Turn your data
              <br />
              into <span className="text-brand-gold">insights.</span>
            </h1>

            <p className="mt-6 max-w-[560px] text-sm leading-6 text-white/65 sm:text-base">
              Upload your dataset, explore its structure, train multiple
              machine-learning models, and compare their results — all in one
              place.
            </p>
          </div>
        </div>

        {/* Bottom label */}
        <div className="relative z-10 flex items-center gap-3">
          <div className="h-px w-10 bg-brand-gold" />

          <span className="font-mono text-[9px] uppercase tracking-[0.2em] text-white/40">
            Explore · Train · Compare
          </span>
        </div>
      </div>

      {/* ============================================================= */}
      {/* RIGHT — WELCOME                                                */}
      {/* ============================================================= */}

      <div className="flex h-screen items-center overflow-hidden bg-surface px-8 py-8 sm:px-12 lg:px-14 xl:px-20">
        <div className="w-full max-w-[560px]">
          {/* PES University logo */}
          <div className="flex items-center">
            <img
              src="/pes-logo.png"
              alt="PES University"
              className="h-16 w-auto object-contain"
            />
          </div>

          {/* Welcome text */}
          <div className="mt-7">
            <p className="font-mono text-[9px] uppercase tracking-[0.2em] text-muted">
              Welcome
            </p>

            <h2 className="mt-2 text-3xl font-semibold leading-tight tracking-tight text-ink sm:text-4xl">
              Build your ML workflow.
            </h2>

            <p className="mt-4 max-w-[520px] text-sm leading-6 text-muted sm:text-base">
              Start by uploading a dataset or choosing one of the available
              sample datasets. The platform will guide you through
              exploration, model selection, training, results, prediction,
              and comparison.
            </p>
          </div>

          {/* Workflow */}
          <div className="mt-7 grid grid-cols-3 gap-2">
            {[
              ["01", "Upload"],
              ["02", "Train"],
              ["03", "Compare"],
            ].map(([number, label]) => (
              <div
                key={number}
                className="rounded-panel border border-rule bg-ground px-3 py-3"
              >
                <p className="font-mono text-[9px] text-signal">
                  {number}
                </p>

                <p className="mt-1 text-sm font-medium text-ink">
                  {label}
                </p>
              </div>
            ))}
          </div>

          {/* Begin */}
          <button
            type="button"
            onClick={onBegin}
            className="
              mt-5
              flex
              w-full
              items-center
              justify-center
              gap-3
              rounded-panel
              bg-signal
              px-6
              py-3
              text-sm
              font-medium
              text-white
              transition-all
              duration-150
              hover:-translate-y-0.5
              hover:opacity-90
              hover:shadow-md
              active:translate-y-0
            "
          >
            <span>Begin</span>
            <span aria-hidden="true">→</span>
          </button>

          <p className="mt-3 text-center font-mono text-[8px] uppercase tracking-[0.15em] text-muted">
            No model preference · Data-driven selection
          </p>
        </div>
      </div>
    </section>
  );
}