import { useState, useMemo } from 'react';
import type { CorrectionItem } from '../../types';
import Collapsible from '../Collapsible';

interface Props {
  corrections: CorrectionItem[];
  modifiedCells: string[];
  onApplySingle: (correction: CorrectionItem) => void;
  onApplyAll: () => void;
  applying?: boolean;
}

export default function CorrectionsPanel({ corrections, modifiedCells, onApplySingle, onApplyAll, applying }: Props) {
  const modifiedSet = useMemo(() => new Set(modifiedCells), [modifiedCells]);

  // Filter out already-applied corrections
  const pending = useMemo(
    () => corrections.filter((c) => !modifiedSet.has(`${c.row_index}_${c.column}`)),
    [corrections, modifiedSet],
  );

  const applicable = pending.filter((c) => c.has_correction);
  const noSuggestion = pending.filter((c) => !c.has_correction);

  const [filterCol, setFilterCol] = useState('all');
  const uniqueCols = useMemo(
    () => [...new Set(pending.map((c) => c.column))].sort(),
    [pending],
  );

  const filtered = useMemo(() => {
    const list = filterCol === 'all' ? pending : pending.filter((c) => c.column === filterCol);
    return list.sort((a, b) => a.row - b.row);
  }, [pending, filterCol]);

  if (!pending.length) {
    return (
      <div className="bg-green-900/30 border border-green-700 rounded-lg p-4 text-green-300 text-sm">
        All corrections have been applied! Your data is clean.
      </div>
    );
  }

  return (
    <Collapsible
      title={`Suggested Corrections (${applicable.length} fixable, ${noSuggestion.length} need review)`}
      defaultOpen={true}
    >
      {/* Controls */}
      <div className="flex items-end gap-4 mb-4">
        <div className="flex-1">
          <label className="block text-xs text-gray-400 mb-1">Filter by column</label>
          <select
            value={filterCol}
            onChange={(e) => setFilterCol(e.target.value)}
            className="w-full bg-gray-800 border border-gray-600 rounded-lg px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none"
          >
            <option value="all">All Columns</option>
            {uniqueCols.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </div>
        {applicable.length > 0 && (
          <button
            onClick={onApplyAll}
            disabled={applying}
            className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white text-sm font-medium rounded-lg transition-colors whitespace-nowrap"
          >
            {applying ? 'Applying...' : `Apply All ${applicable.length} Corrections`}
          </button>
        )}
      </div>

      {/* Header */}
      <div className="grid grid-cols-[50px_1fr_1.5fr_1.5fr_2fr_70px] gap-2 text-xs font-semibold text-gray-400 uppercase tracking-wide border-b border-gray-700 pb-2 mb-2">
        <span>Row</span>
        <span>Column</span>
        <span>Original</span>
        <span>Suggestion</span>
        <span>Reason</span>
        <span>Action</span>
      </div>

      {/* Rows */}
      <div className="max-h-[400px] overflow-y-auto">
        {filtered.map((c, i) => (
          <div
            key={`${c.row_index}_${c.column}_${i}`}
            className={`grid grid-cols-[50px_1fr_1.5fr_1.5fr_2fr_70px] gap-2 items-center py-2 text-sm border-b border-gray-800/50 ${
              c.has_correction ? 'border-l-2 border-l-green-500 pl-2' : 'border-l-2 border-l-red-500 pl-2'
            }`}
          >
            <span className="font-bold">{c.row}</span>
            <code className="text-indigo-400 text-xs">{c.column}</code>
            <span className="text-red-400 font-medium truncate">{c.original}</span>
            <span>
              {c.has_correction ? (
                <span className="text-green-400">
                  -&gt; <strong>{c.suggested}</strong>
                </span>
              ) : (
                <span className="text-gray-500">No suggestion</span>
              )}
            </span>
            <span className="text-gray-400 text-xs truncate">{c.reason}</span>
            <span>
              {c.has_correction && (
                <button
                  onClick={() => onApplySingle(c)}
                  className="px-2.5 py-1 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium rounded transition-colors"
                >
                  Fix
                </button>
              )}
            </span>
          </div>
        ))}
      </div>
    </Collapsible>
  );
}
