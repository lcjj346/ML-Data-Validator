import { useCallback, useState, type DragEvent } from 'react';

interface Props {
  onFile: (file: File) => void;
  disabled?: boolean;
}

export default function FileUpload({ onFile, disabled }: Props) {
  const [dragging, setDragging] = useState(false);

  const handleDrop = useCallback(
    (e: DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file && file.name.endsWith('.csv')) onFile(file);
    },
    [onFile],
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) onFile(file);
      e.target.value = '';
    },
    [onFile],
  );

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      className={`glass-card p-8 text-center transition-all duration-200 cursor-pointer border-dashed ${
        dragging
          ? 'border-indigo-500 bg-indigo-500/10 scale-[1.01]'
          : 'hover:border-white/20 hover:bg-white/5'
      } ${disabled ? 'opacity-50 pointer-events-none' : ''}`}
    >
      <svg className="w-10 h-10 mx-auto mb-3 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
      </svg>
      <p className="text-gray-300 mb-1">Drag & drop a CSV file here, or click to browse</p>
      <p className="text-gray-500 text-xs mb-4">
        <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-gray-800/80 rounded-full text-xs text-gray-400">.csv</span>
      </p>
      <input
        type="file"
        accept=".csv"
        onChange={handleChange}
        className="hidden"
        id="file-upload"
        disabled={disabled}
      />
      <label
        htmlFor="file-upload"
        className="inline-block px-5 py-2 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-lg text-sm font-medium hover:from-indigo-500 hover:to-purple-500 cursor-pointer transition-all duration-200 shadow-lg shadow-indigo-500/20 hover:shadow-indigo-500/30"
      >
        Choose File
      </label>
    </div>
  );
}
