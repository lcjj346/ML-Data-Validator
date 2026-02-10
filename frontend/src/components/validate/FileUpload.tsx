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
    },
    [onFile],
  );

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors cursor-pointer ${
        dragging ? 'border-indigo-500 bg-indigo-500/10' : 'border-gray-600 hover:border-gray-500'
      } ${disabled ? 'opacity-50 pointer-events-none' : ''}`}
    >
      <p className="text-gray-300 mb-2">Drag & drop a CSV file here, or click to browse</p>
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
        className="inline-block px-4 py-2 bg-indigo-600 text-white rounded-lg text-sm font-medium hover:bg-indigo-500 cursor-pointer transition-colors"
      >
        Choose File
      </label>
    </div>
  );
}
