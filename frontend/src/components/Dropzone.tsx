import { useState, type DragEvent } from "react";

interface DropzoneProps {
  onFile: (file: File) => void;
  loading: boolean;
  fileName?: string;
}

/**
 * Shared CSV upload dropzone.
 *
 * States:
 * - Idle: ready for a file
 * - Dragging: highlighted drop target
 * - Loading: upload in progress
 * - Selected: file has been accepted
 */
export function Dropzone({
  onFile,
  loading,
  fileName,
}: DropzoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const inputId = "csv-upload";

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();

    if (loading) return;

    setIsDragging(false);

    const droppedFile = e.dataTransfer.files[0];

    if (droppedFile) {
      onFile(droppedFile);
    }
  };

  const hasFile = !!fileName;

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();

        if (!loading) {
          setIsDragging(true);
        }
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      className={`
        relative
        flex
        min-h-44
        flex-col
        items-center
        justify-center
        gap-3
        rounded-panel
        border-2
        border-dashed
        px-6
        py-8
        text-center
        transition-all
        duration-150
        ${
          loading
            ? "cursor-not-allowed border-rule bg-ground opacity-70"
            : isDragging
              ? "border-signal bg-ground shadow-sm"
              : hasFile
                ? "border-signal/60 bg-ground"
                : "border-rule bg-surface hover:border-ink/40 hover:bg-ground"
        }
      `}
    >
      {/* Upload icon / status */}
      <div
        className={`
          flex
          h-12
          w-12
          items-center
          justify-center
          rounded-full
          border
          font-mono
          text-sm
          transition-colors
          ${
            loading
              ? "border-rule text-muted"
              : isDragging || hasFile
                ? "border-signal text-signal"
                : "border-rule text-muted"
          }
        `}
      >
        {loading ? "…" : hasFile ? "✓" : "↑"}
      </div>

      {/* Main action */}
      <div>
        <label
          htmlFor={inputId}
          className={`
            text-sm
            font-medium
            ${
              loading
                ? "cursor-not-allowed text-muted"
                : "cursor-pointer text-ink hover:text-signal"
            }
          `}
        >
          {loading
            ? "Uploading…"
            : isDragging
              ? "Drop your CSV here"
              : hasFile
                ? fileName
                : "Drop a CSV file or click to browse"}
        </label>

        {!loading && hasFile && (
          <p className="mt-1 text-xs text-signal">
            File selected
          </p>
        )}
      </div>

      {/* File requirements */}
      <span className="font-mono text-[10px] uppercase tracking-wide text-muted">
        CSV · 50+ rows · ≤100 columns
      </span>

      <input
        id={inputId}
        type="file"
        accept=".csv"
        className="sr-only"
        disabled={loading}
        onChange={(e) => {
          const selected = e.target.files?.[0];

          if (selected) {
            onFile(selected);
          }

          // Allows selecting the same file again.
          e.target.value = "";
        }}
      />
    </div>
  );
}