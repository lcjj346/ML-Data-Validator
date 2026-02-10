import type { TrainingMetrics as TMetrics } from '../../types';
import Collapsible from '../Collapsible';

interface Props {
  metrics: TMetrics;
}

function Metric({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="bg-gray-800 rounded-lg p-3 text-center">
      <div className="text-gray-400 text-xs uppercase tracking-wide mb-1">{label}</div>
      <div className="text-lg font-bold">{value}</div>
    </div>
  );
}

function ConfusionMatrix({ cm }: { cm: number[][] }) {
  return (
    <div className="mt-2">
      <p className="text-xs font-semibold text-gray-400 mb-1">Confusion Matrix</p>
      <table className="text-xs border border-gray-700">
        <thead>
          <tr className="text-gray-400">
            <th className="px-2 py-1"></th>
            <th className="px-2 py-1">Pred Invalid</th>
            <th className="px-2 py-1">Pred Valid</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="px-2 py-1 text-gray-400 font-medium">Act Invalid</td>
            <td className="px-2 py-1 text-center bg-blue-900/30">{cm[0]?.[0] ?? 0}</td>
            <td className="px-2 py-1 text-center">{cm[0]?.[1] ?? 0}</td>
          </tr>
          <tr>
            <td className="px-2 py-1 text-gray-400 font-medium">Act Valid</td>
            <td className="px-2 py-1 text-center">{cm[1]?.[0] ?? 0}</td>
            <td className="px-2 py-1 text-center bg-blue-900/30">{cm[1]?.[1] ?? 0}</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

export default function TrainingMetrics({ metrics }: Props) {
  return (
    <div>
      <h3 className="text-lg font-semibold mb-3">Training Metrics</h3>
      {Object.entries(metrics).map(([colName, m]) => (
        <Collapsible key={colName} title={`Column: ${colName}`} defaultOpen={true}>
          <div className="grid grid-cols-3 gap-3 mb-3">
            <Metric label="Unique Valid" value={m.unique_valid ?? 'N/A'} />
            <Metric label="Total Samples" value={m.total_samples ?? 'N/A'} />
            <Metric label="Test Size" value={m.used_split ? (m.test_size ?? 'N/A') : 'N/A (small)'} />
          </div>

          {m.used_split ? (
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-xs font-semibold text-gray-400 mb-2">Train Set</p>
                <div className="grid grid-cols-2 gap-2">
                  <Metric label="Accuracy" value={m.train_accuracy != null ? `${(m.train_accuracy * 100).toFixed(1)}%` : 'N/A'} />
                  <Metric label="F1 Score" value={m.train_f1 != null ? `${(m.train_f1 * 100).toFixed(1)}%` : 'N/A'} />
                </div>
              </div>
              <div>
                <p className="text-xs font-semibold text-gray-400 mb-2">Test Set</p>
                <div className="grid grid-cols-2 gap-2">
                  <Metric label="Accuracy" value={m.test_accuracy != null ? `${(m.test_accuracy * 100).toFixed(1)}%` : 'N/A'} />
                  <Metric label="F1 Score" value={m.test_f1 != null ? `${(m.test_f1 * 100).toFixed(1)}%` : 'N/A'} />
                </div>
                {m.test_confusion_matrix && <ConfusionMatrix cm={m.test_confusion_matrix} />}
              </div>
            </div>
          ) : (
            <div>
              <div className="bg-yellow-900/30 border border-yellow-700 rounded-lg p-2 text-yellow-300 text-xs mb-2">
                Dataset too small for train/test split. Metrics may be overfit.
              </div>
              <div className="grid grid-cols-2 gap-2">
                <Metric label="Accuracy" value={m.train_accuracy != null ? `${(m.train_accuracy * 100).toFixed(1)}%` : 'N/A'} />
                <Metric label="F1 Score" value={m.train_f1 != null ? `${(m.train_f1 * 100).toFixed(1)}%` : 'N/A'} />
              </div>
            </div>
          )}
        </Collapsible>
      ))}
    </div>
  );
}
