import type { MatchColumnsResponse } from '../../types';
import Collapsible from '../Collapsible';

interface Props {
  match: MatchColumnsResponse;
}

export default function ColumnMatching({ match }: Props) {
  return (
    <Collapsible title="Column Matching" defaultOpen={true}>
      <p className="text-sm text-gray-300 mb-2">
        <span className="font-medium">Model trained on:</span>{' '}
        {match.trained_columns.join(', ')}
      </p>

      {Object.keys(match.matched).length > 0 && (
        <div className="mb-2">
          <p className="text-sm font-medium text-gray-300 mb-1">Auto-detected matches:</p>
          <div className="flex flex-wrap gap-2">
            {Object.entries(match.matched).map(([input, trained]) => (
              <span
                key={input}
                className="inline-flex items-center gap-1 bg-gray-800 px-2.5 py-1 rounded text-xs"
              >
                <code className="text-indigo-400">{input}</code>
                <span className="text-gray-500">-&gt;</span>
                <code className="text-green-400">{trained}</code>
              </span>
            ))}
          </div>
        </div>
      )}

      {match.unmatched.length > 0 && (
        <p className="text-sm text-gray-400">
          <span className="font-medium">Unmatched (will be skipped):</span>{' '}
          {match.unmatched.join(', ')}
        </p>
      )}
    </Collapsible>
  );
}
