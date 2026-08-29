import { useState } from "react";
import { Dropzone } from "../Dropzone";
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

/** Screen 1 (frontend.md): dropzone, target-column select (defaults to the
 * last column -- DATA_FLOW_GUIDE.md SS2), split slider, sample datasets as a
 * secondary path. */
export function UploadScreen({ profile, onProfile, testSize, onTestSizeChange }: UploadScreenProps) {
  const { samples, samplesError, loading, error, upload, loadSample } = useDataset();
  // Kept so changing the target column can re-ingest the same file (the
  // backend only knows the target at ingest time; samples have no file).
  const [file, setFile] = useState<File | null>(null);

  const handleFile = async (nextFile: File) => {
    setFile(nextFile);
    const result = await upload(nextFile, null, true);
    if (result) onProfile(result);
  };

  const handleTargetChange = async (value: string) => {
    if (!file) return;
    const result = value === NO_TARGET ? await upload(file, null, false) : await upload(file, value, true);
    if (result) onProfile(result);
  };

  const handleSample = async (sampleId: string) => {
    setFile(null);
    const result = await loadSample(sampleId);
    if (result) onProfile(result);
  };

  return (
    <ScreenPanel maxWidthClassName="max-w-[640px]">
      <div className="flex flex-col gap-6">
        <div>
          <h1 className="text-lg font-medium text-ink">Upload</h1>
          <p className="mt-1 text-sm text-muted">
            Upload a CSV to profile it, choose a target column, and set the train/test split.
          </p>
        </div>

        <Dropzone onFile={handleFile} loading={loading} fileName={file?.name ?? profile?.source} />

        {error && <p className="text-sm text-ink">{error.message}</p>}

        {profile && file && (
          <div className="rounded-panel border border-rule p-4">
            <label className="text-sm text-muted" htmlFor="target-column">
              Target column
            </label>
            <select
              id="target-column"
              className="mt-1 w-full rounded-panel border border-rule bg-surface px-3 py-2 text-sm text-ink"
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

        {profile && (
          <div className="rounded-panel border border-rule p-4">
            <SplitSlider testSize={testSize} onChange={onTestSizeChange} />
          </div>
        )}

        {profile && (
          <p className="font-mono text-xs text-muted">
            {profile.n_rows} rows × {profile.n_columns} columns · detected as {profile.data_type}
          </p>
        )}

        <div>
          <h2 className="text-sm font-medium uppercase tracking-wide text-muted">Sample datasets</h2>
          {samplesError && <p className="mt-1 text-sm text-ink">{samplesError.message}</p>}
          <ul className="mt-2 flex flex-col gap-2">
            {(samples ?? []).map((sample) => (
              <li key={sample.id}>
                <button
                  type="button"
                  onClick={() => handleSample(sample.id)}
                  disabled={loading}
                  className="w-full rounded-panel border border-rule px-4 py-3 text-left text-sm hover:border-ink disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <span className="font-medium text-ink">{sample.name}</span>
                  <span className="ml-2 font-mono text-xs text-muted">
                    {sample.n_rows}×{sample.n_columns} · {sample.data_type}
                  </span>
                  <p className="mt-1 text-xs text-muted">{sample.description}</p>
                </button>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </ScreenPanel>
  );
}
