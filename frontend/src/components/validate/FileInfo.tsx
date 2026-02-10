interface Props {
  filename: string;
  rows: number;
  columns: number;
}

export default function FileInfo({ filename, rows, columns }: Props) {
  return (
    <div className="grid grid-cols-3 gap-4 my-4">
      {[
        { label: 'Rows', value: rows },
        { label: 'Columns', value: columns },
        { label: 'File', value: filename },
      ].map((m) => (
        <div key={m.label} className="bg-gray-800 rounded-lg p-4 text-center">
          <div className="text-gray-400 text-xs uppercase tracking-wide mb-1">{m.label}</div>
          <div className="text-2xl font-bold">{m.value}</div>
        </div>
      ))}
    </div>
  );
}
