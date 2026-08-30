import { useState, type DragEvent } from "react";

interface DropzoneProps {
  onFile: (file: File) => void;
  loading: boolean;
  fileName?: string;
}

/** Screen 1's dropzone (frontend.md layout contract). Click-to-browse and
 * drag-and-drop both land on the same file input for one code path. */
export function Dropzone({ onFile, loading, fileName }: DropzoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const inputId = "csv-upload";

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) onFile(droppedFile);
  };

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      className={
        "flex flex-col items-center justify-center gap-1 rounded-panel border border-dashed p-8 text-center " +
        (isDragging ? "border-signal" : "border-rule")
      }
    >
      <label htmlFor={inputId} className="cursor-pointer text-sm text-ink">
        {loading ? "Uploading…" : fileName ? `Selected: ${fileName}` : "Drop a CSV file, or click to browse"}
      </label>
      <span className="text-xs text-muted">CSV, at least 50 rows, at most 100 columns</span>
      <input
        id={inputId}
        type="file"
        accept=".csv"
        className="sr-only"
        disabled={loading}
        onChange={(e) => {
          const selected = e.target.files?.[0];
          if (selected) onFile(selected);
        }}
      />
    </div>
  );
}
