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

// Which pipeline stage flagged the cell - colors match how deterministic the check is
const STAGE_STYLES: Record<string, { label: string; cls: string }> = {
  empty: { label: 'empty', cls: 'bg-slate-500/20 text-slate-300 border-slate-400/30' },
  rule: { label: 'rule', cls: 'bg-purple-500/20 text-purple-300 border-purple-400/30' },
  range: { label: 'range', cls: 'bg-blue-500/20 text-blue-300 border-blue-400/30' },
  typo: { label: 'typo', cls: 'bg-amber-500/20 text-amber-300 border-amber-400/30' },
  'unknown-value': { label: 'unknown', cls: 'bg-orange-500/20 text-orange-300 border-orange-400/30' },
  ml: { label: 'ML', cls: 'bg-cyan-500/20 text-cyan-300 border-cyan-400/30' },
};

function StageBadge({ stage }: { stage?: string }) {
  const s = STAGE_STYLES[stage ?? 'ml'] ?? STAGE_STYLES.ml;
  return (
    <span className={`inline-block px-1.5 py-0.5 rounded border text-[10px] font-semibold uppercase tracking-wide ${s.cls}`}>
      {s.label}
    </span>
  );
}

export default function CorrectionsPanel({ corrections, modifiedCells, onApplySingle, onApplyAll, applying }: Props) {
  const modifiedSet = useMemo(() => new Set(modifiedCells), [modifiedCells]);

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
      <div className="glass-card border-green-500/30 p-4 text-green-300 text-sm flex items-center gap-2">
        <svg className="w-5 h-5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
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
          <label className="block text-xs text-slate-400 mb-1">Filter by column</label>
          <select
            value={filterCol}
            onChange={(e) => setFilterCol(e.target.value)}
            className="w-full bg-slate-800/80 border border-white/10 rounded-xl px-3 py-2 text-sm focus:border-cyan-500 focus:outline-none focus:ring-1 focus:ring-cyan-500/50 transition-all duration-200"
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
            className="px-5 py-2 bg-gradient-to-r from-cyan-500 to-sky-600 hover:from-cyan-400 hover:to-sky-500 disabled:opacity-50 text-white text-sm font-medium rounded-xl transition-all duration-200 whitespace-nowrap shadow-lg shadow-cyan-500/20"
          >
            {applying ? 'Applying...' : `Apply All ${applicable.length} Corrections`}
          </button>
        )}
      </div>

      {/* Table: header sticky inside scroll container so columns always align */}
      <div className="max-h-[400px] overflow-y-auto">
        {/* Header */}
        <div className="grid grid-cols-[40px_0.4fr_0.5fr_0.5fr_0.6fr_70px_80px_70px] gap-2 text-xs font-semibold text-white uppercase tracking-wide border-b border-white/10 pb-2 mb-2 pl-2 border-l-2 border-l-transparent sticky top-0 bg-slate-900/95 z-10">
          <span>Row</span>
          <span>Column</span>
          <span>Original</span>
          <span>Suggestion</span>
          <span>Reason</span>
          <span>Stage</span>
          <span>Confidence</span>
          <span>Action</span>
        </div>

        {filtered.map((c, i) => (
          <div
            key={`${c.row_index}_${c.column}_${i}`}
            className={`grid grid-cols-[40px_0.4fr_0.5fr_0.5fr_0.6fr_70px_80px_70px] gap-2 items-center py-2.5 text-sm border-b border-white/5 hover:bg-white/5 transition-colors pl-2 border-l-2 ${
              c.has_correction ? 'border-l-green-500/60' : 'border-l-red-500/60'
            }`}
          >
            <span className="font-bold text-slate-300">{c.row}</span>
            <code className="text-cyan-400 text-xs">{c.column}</code>
            <span className="text-red-400 font-medium truncate">{c.original}</span>
            <span>
              {c.has_correction ? (
                <span className="text-green-400 flex items-center gap-1">
                  <svg className="w-3 h-3 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M13 7l5 5m0 0l-5 5m5-5H6" />
                  </svg>
                  <strong>{c.suggested}</strong>
                </span>
              ) : (
                <span className="text-slate-500 italic">No suggestion</span>
              )}
            </span>
            <span className="text-slate-400 text-xs truncate cursor-help" title={c.reason}>{c.reason}</span>
            <span><StageBadge stage={c.stage} /></span>
            <span className="text-xs text-slate-400">
              {c.confidence != null ? `${(c.confidence * 100).toFixed(0)}%` : '—'}
            </span>
            <span>
              {c.has_correction && (
                <button
                  onClick={() => onApplySingle(c)}
                  className="px-3 py-1 bg-cyan-500/80 hover:bg-cyan-400 text-white text-xs font-medium rounded-lg transition-all duration-200 hover:scale-105"
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
