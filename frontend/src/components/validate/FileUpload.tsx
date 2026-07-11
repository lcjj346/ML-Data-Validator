import { useCallback, useState, type DragEvent } from 'react';

interface Props {
  onFile: (file: File) => void;
  disabled?: boolean;
  uploadError?: string | null; // server rejected the file - show failure, not success
}

function formatFileSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function FileUpload({ onFile, disabled, uploadError }: Props) {
  const [dragging, setDragging] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);

  const handleDrop = useCallback(
    (e: DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file && (file.name.endsWith('.csv') || file.name.endsWith('.xlsx'))) {
        setUploadedFile(file);
        onFile(file);
      }
    },
    [onFile],
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) {
        setUploadedFile(file);
        onFile(file);
      }
      e.target.value = '';
    },
    [onFile],
  );

  if (uploadedFile) {
    return (
      <div className={`glass-card p-5 transition-all duration-200 ${disabled ? 'opacity-50 pointer-events-none' : ''}`}>
        <div className="flex items-center gap-4">
          <div className="flex-shrink-0 w-11 h-11 bg-cyan-500/10 border border-cyan-500/30 rounded-lg flex items-center justify-center">
            <svg className="w-5 h-5 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
            </svg>
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-slate-200 text-sm font-medium truncate">{uploadedFile.name}</p>
            <p className="text-slate-500 text-xs mt-0.5">
              {formatFileSize(uploadedFile.size)}
              {uploadError && <span className="text-red-400 ml-2">- upload rejected</span>}
            </p>
          </div>
          {uploadError ? (
            <div className="flex-shrink-0 w-7 h-7 bg-red-500/10 border border-red-500/30 rounded-full flex items-center justify-center">
              <svg className="w-3.5 h-3.5 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </div>
          ) : (
            <div className="flex-shrink-0 w-7 h-7 bg-green-500/10 border border-green-500/30 rounded-full flex items-center justify-center">
              <svg className="w-3.5 h-3.5 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />
              </svg>
            </div>
          )}
        </div>
        <div className="mt-3 pt-3 border-t border-white/5">
          <input
            type="file"
            accept=".csv,.xlsx"
            onChange={handleChange}
            className="hidden"
            id="file-upload-replace"
            disabled={disabled}
          />
          <label
            htmlFor="file-upload-replace"
            className="text-xs text-slate-500 hover:text-slate-300 cursor-pointer transition-colors duration-150 underline underline-offset-2"
          >
            Replace file
          </label>
        </div>
      </div>
    );
  }

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      className={`glass-card p-8 text-center transition-all duration-200 cursor-pointer border-dashed ${
        dragging
          ? 'border-cyan-500 bg-cyan-500/10 scale-[1.01]'
          : 'hover:border-white/20 hover:bg-white/5'
      } ${disabled ? 'opacity-50 pointer-events-none' : ''}`}
    >
      <svg className="w-10 h-10 mx-auto mb-3 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
      </svg>
      <p className="text-slate-300 mb-1">Drag & drop a CSV or Excel file here, or click to browse</p>
      <p className="text-slate-500 text-xs mb-4 flex items-center justify-center gap-2">
        <span className="inline-flex items-center px-2 py-0.5 bg-slate-800/80 rounded-full text-slate-400">.csv / .xlsx</span>
        <span className="text-slate-600">·</span>
        <span>Max 10 MB</span>
      </p>
      <input
        type="file"
        accept=".csv,.xlsx"
        onChange={handleChange}
        className="hidden"
        id="file-upload"
        disabled={disabled}
      />
      <label
        htmlFor="file-upload"
        className="inline-block px-5 py-2 bg-gradient-to-r from-cyan-500 to-sky-600 text-white rounded-lg text-sm font-medium hover:from-cyan-400 hover:to-sky-500 cursor-pointer transition-all duration-200 shadow-lg shadow-cyan-500/20 hover:shadow-cyan-500/30"
      >
        Choose File
      </label>
    </div>
  );
}
