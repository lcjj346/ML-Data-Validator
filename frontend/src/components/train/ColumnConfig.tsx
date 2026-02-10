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
      <p className="text-xs text-gray-400 mb-2">Click to exclude columns from training:</p>
      <div className="flex flex-wrap gap-2 mb-3">
        {allColumns.map((col) => {
          const isExcluded = excluded.includes(col);
          return (
            <button
              key={col}
              onClick={() => toggle(col)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                isExcluded
                  ? 'bg-gray-700 text-gray-400 line-through'
                  : 'bg-indigo-600/20 text-indigo-300 border border-indigo-600/50'
              }`}
            >
              {col}
            </button>
          );
        })}
      </div>
      <p className="text-sm text-indigo-400">
        Will train on {included.length} columns: {included.join(', ')}
      </p>
    </div>
  );
}
