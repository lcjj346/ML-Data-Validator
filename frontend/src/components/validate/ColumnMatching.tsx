import type { MatchColumnsResponse } from '../../types';
import Collapsible from '../Collapsible';

interface Props {
  match: MatchColumnsResponse;
}

export default function ColumnMatching({ match }: Props) {
  return (
    <Collapsible title="Column Matching" defaultOpen={true}>
      <p className="text-sm text-slate-300 mb-3">
        <span className="font-medium">Model trained on:</span>{' '}
        {match.trained_columns.join(', ')}
      </p>

      {Object.keys(match.matched).length > 0 && (
        <div className="mb-3">
          <p className="text-sm font-medium text-slate-300 mb-2">Auto-detected matches:</p>
          <div className="flex flex-wrap gap-2">
            {Object.entries(match.matched).map(([input, trained]) => (
              <span
                key={input}
                className="inline-flex items-center gap-1.5 bg-slate-800/80 px-3 py-1.5 rounded-lg text-xs border border-white/5"
              >
                <code className="text-cyan-400 font-medium">{input}</code>
                <svg className="w-3 h-3 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M13 7l5 5m0 0l-5 5m5-5H6" />
                </svg>
                <code className="text-green-400 font-medium">{trained}</code>
              </span>
            ))}
          </div>
        </div>
      )}

      {match.unmatched.length > 0 && (
        <p className="text-sm text-slate-400">
          <span className="font-medium">Unmatched (will be skipped):</span>{' '}
          {match.unmatched.join(', ')}
        </p>
      )}
    </Collapsible>
  );
}
