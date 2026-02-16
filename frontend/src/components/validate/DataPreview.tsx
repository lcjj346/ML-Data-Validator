import Collapsible from '../Collapsible';

interface Props {
  preview: Record<string, unknown>[];
  columns: string[];
}

export default function DataPreview({ preview, columns }: Props) {
  if (!preview.length) return null;

  return (
    <Collapsible title="Preview uploaded data" defaultOpen={false}>
      <div className="overflow-x-auto rounded-lg">
        <table className="w-full text-sm text-left">
          <thead className="text-xs uppercase text-gray-400 bg-gray-800/60">
            <tr>
              {columns.map((col) => (
                <th key={col} className="px-3 py-2.5 font-medium">{col}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {preview.map((row, i) => (
              <tr key={i} className="border-b border-white/5 hover:bg-white/5 transition-colors">
                {columns.map((col) => (
                  <td key={col} className="px-3 py-2 text-gray-300">{String(row[col] ?? '')}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Collapsible>
  );
}
