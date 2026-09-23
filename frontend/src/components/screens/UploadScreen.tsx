import { useState } from "react";
import { Dropzone } from "../Dropzone";
import { ScreenHeader } from "../ScreenHeader";
import { ScreenPanel } from "../ScreenPanel";
import { SplitSlider } from "../SplitSlider";
import { useDataset, type DataProfileResponse } from "../../hooks/useDataset";

const NO_TARGET = "__none__";

interface UploadScreenProps {
  profile: DataProfileResponse | null;
  onProfile: (profile: DataProfileResponse) => void;
  testSize: number;
  onTestSizeChange: (testSize: number) => void;
}

/**
 * Screen 1 -- two columns, no scroll (CLAUDE.md's layout rule): the
 * dropzone/loaded-dataset config on the left, sample datasets as a compact
 * list on the right. Neither column grows past the shell's one screen
 * height, so nothing here needs its own scroll region.
 */
export function UploadScreen({
  profile,
  onProfile,
  testSize,
  onTestSizeChange,
}: UploadScreenProps) {
  const {
    samples,
    samplesError,
    loading,
    error,
    upload,
    loadSample,
  } = useDataset();

  // Kept so changing the target column can re-ingest the same file.
  const [file, setFile] = useState<File | null>(null);

  const handleFile = async (nextFile: File) => {
    setFile(nextFile);

    const result = await upload(nextFile, null, true);

    if (result) {
      onProfile(result);
    }
  };

  const handleTargetChange = async (value: string) => {
    if (!file) return;

    const result =
      value === NO_TARGET
        ? await upload(file, null, false)
        : await upload(file, value, true);

    if (result) {
      onProfile(result);
    }
  };

  const handleSample = async (sampleId: string) => {
    setFile(null);

    const result = await loadSample(sampleId);

    if (result) {
      onProfile(result);
    }
  };

  return (
    <ScreenPanel>
      <ScreenHeader
        title="Upload"
        description="Upload a CSV dataset or start with one of the sample datasets below."
      />

      <div className="grid min-h-0 flex-1 grid-cols-2 gap-6">
        {/* Left: dropzone + loaded-dataset config */}
        <div className="flex min-h-0 flex-col gap-3">
          <div className={profile ? "shrink-0" : "flex min-h-0 flex-1 flex-col"}>
            <Dropzone
              onFile={handleFile}
              loading={loading}
              fileName={file?.name ?? profile?.source}
              fill={!profile}
            />
          </div>

          {error && (
            <div className="shrink-0 rounded-panel border border-rule bg-ground px-4 py-3">
              <p className="text-sm font-medium text-ink">Upload failed</p>
              <p className="mt-1 text-xs leading-relaxed text-muted">{error.message}</p>
            </div>
          )}

          {profile && (
            <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto rounded-panel border border-rule bg-ground p-4">
              {/* Compact stat chips */}
              <div className="flex flex-wrap items-center gap-2">
                <StatChip label="Rows" value={profile.n_rows.toLocaleString()} />
                <StatChip label="Columns" value={profile.n_columns.toString()} />
                <StatChip label="Type" value={profile.data_type} />
                <span className="ml-auto rounded-full bg-surface px-3 py-1.5 font-mono text-[10px] uppercase tracking-wide text-signal">
                  ✓ Ready
                </span>
              </div>

              {/* Target */}
              {file && (
                <div>
                  <label className="text-xs font-medium text-ink" htmlFor="target-column">
                    Target column
                  </label>

                  <select
                    id="target-column"
                    className="
                      mt-2
                      w-full
                      cursor-pointer
                      rounded-panel
                      border
                      border-rule
                      bg-surface
                      px-3
                      py-2
                      text-sm
                      text-ink
                      transition-colors
                      hover:border-ink/40
                      focus:border-signal
                    "
                    value={profile.target_column ?? NO_TARGET}
                    disabled={loading}
                    onChange={(e) => handleTargetChange(e.target.value)}
                  >
                    {profile.columns.map((column) => (
                      <option key={column.name} value={column.name}>
                        {column.name}
                      </option>
                    ))}

                    <option value={NO_TARGET}>No target (clustering)</option>
                  </select>
                </div>
              )}

              {/* Split */}
              <div className="rounded-panel border border-rule bg-surface p-3">
                <SplitSlider
                  testSize={testSize}
                  onChange={onTestSizeChange}
                  disabled={loading}
                />
              </div>
            </div>
          )}
        </div>

        {/* Right: sample datasets, compact list */}
        <div className="flex min-h-0 flex-col gap-3">
          <div>
            <h2 className="text-sm font-semibold text-ink">Sample datasets</h2>
            <p className="mt-1 text-xs text-muted">Try the platform without uploading your own data.</p>
          </div>

          {samplesError && (
            <div className="shrink-0 rounded-panel border border-rule bg-ground px-4 py-3">
              <p className="text-xs text-muted">{samplesError.message}</p>
            </div>
          )}

          <div className="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto">
            {(samples ?? []).map((sample) => (
              <button
                key={sample.id}
                type="button"
                onClick={() => handleSample(sample.id)}
                disabled={loading}
                className="
                  group
                  w-full
                  shrink-0
                  cursor-pointer
                  rounded-panel
                  border
                  border-rule
                  bg-surface
                  px-4
                  py-3
                  text-left
                  transition-all
                  duration-150
                  hover:border-ink/40
                  hover:bg-ground
                  active:scale-[0.995]
                  disabled:cursor-not-allowed
                  disabled:opacity-50
                "
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-sm font-semibold text-ink">{sample.name}</span>
                      <span className="font-mono text-[10px] uppercase tracking-wide text-muted">
                        {sample.data_type}
                      </span>
                    </div>

                    <p className="mt-1 text-xs leading-relaxed text-muted">{sample.description}</p>
                  </div>

                  <span className="shrink-0 font-mono text-xs text-muted transition-colors group-hover:text-signal">
                    {sample.n_rows} × {sample.n_columns}
                  </span>
                </div>
              </button>
            ))}
          </div>
        </div>
      </div>
    </ScreenPanel>
  );
}

/* -------------------------------------------------------------------------- */
/* Small dataset statistic chip                                               */
/* -------------------------------------------------------------------------- */

function StatChip({ label, value }: { label: string; value: string }) {
  return (
    <span className="rounded-full bg-surface px-3 py-1.5 font-mono text-[10px] uppercase tracking-wide text-ink">
      <span className="text-muted">{label}</span> {value}
    </span>
  );
}
