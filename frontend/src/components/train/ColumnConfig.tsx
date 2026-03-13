interface Props {
  allColumns: string[];
  excluded: string[];
  onChange: (excluded: string[]) => void;
}

export default function ColumnConfig({ allColumns, excluded, onChange }: Props) {
  const toggle = (col: string) => {
    if (excluded.includes(col)) {
      onChange(excluded.filter((c) => c !== col));
    } else {
      onChange([...excluded, col]);
    }
  };

  const included = allColumns.filter((c) => !excluded.includes(c));

  return (
    <div>
      <p className="text-xs text-slate-400 mb-3">Click to exclude columns from training:</p>
      <div className="flex flex-wrap gap-2 mb-3">
        {allColumns.map((col) => {
          const isExcluded = excluded.includes(col);
          return (
            <button
              key={col}
              onClick={() => toggle(col)}
              className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all duration-200 ${
                isExcluded
                  ? 'bg-slate-800/60 text-slate-500 line-through border border-white/5'
                  : 'bg-cyan-500/15 text-cyan-300 border border-cyan-500/30 hover:bg-cyan-500/25 hover:scale-105'
              }`}
            >
              {col}
            </button>
          );
        })}
      </div>
      <p className="text-sm text-cyan-400">
        Will train on {included.length} columns: {included.join(', ')}
      </p>
    </div>
  );
}
