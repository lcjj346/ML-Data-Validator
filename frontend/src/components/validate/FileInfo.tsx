interface Props {
  filename: string;
  rows: number;
  columns: number;
}

const icons = {
  Rows: (
    <svg className="w-5 h-5 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5" />
    </svg>
  ),
  Columns: (
    <svg className="w-5 h-5 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M9 4.5v15m6-15v15m-10.875 0h15.75c.621 0 1.125-.504 1.125-1.125V5.625c0-.621-.504-1.125-1.125-1.125H4.125C3.504 4.5 3 5.004 3 5.625v12.75c0 .621.504 1.125 1.125 1.125z" />
    </svg>
  ),
  File: (
    <svg className="w-5 h-5 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
    </svg>
  ),
};

export default function FileInfo({ filename, rows, columns }: Props) {
  return (
    <div className="grid grid-cols-3 gap-4 my-4 animate-fadeIn">
      {[
        { label: 'Rows', value: rows },
        { label: 'Columns', value: columns },
        { label: 'File', value: filename },
      ].map((m) => (
        <div key={m.label} className="glass-card p-4 text-center hover:bg-white/[0.07] transition-colors">
          <div className="flex justify-center mb-2">{icons[m.label as keyof typeof icons]}</div>
          <div className="text-gray-400 text-xs uppercase tracking-wide mb-1">{m.label}</div>
          <div className="text-xl font-bold truncate">{m.value}</div>
        </div>
      ))}
    </div>
  );
}
