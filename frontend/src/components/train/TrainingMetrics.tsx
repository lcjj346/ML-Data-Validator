import type { TrainingMetrics as TMetrics } from '../../types';
import Collapsible from '../Collapsible';

interface Props {
  metrics: TMetrics;
}

function ConfusionMatrix({ cm }: { cm: number[][] }) {
  return (
    <div className="mt-2">
      <p className="text-xs font-semibold text-slate-400 mb-1">Confusion Matrix</p>
      <table className="text-xs border border-white/10 rounded-lg overflow-hidden">
        <thead>
          <tr className="text-slate-400">
            <th className="px-2 py-1"></th>
            <th className="px-2 py-1">Pred Invalid</th>
            <th className="px-2 py-1">Pred Valid</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="px-2 py-1 text-slate-400 font-medium">Act Invalid</td>
            <td className="px-2 py-1 text-center bg-cyan-500/15">{cm[0]?.[0] ?? 0}</td>
            <td className="px-2 py-1 text-center">{cm[0]?.[1] ?? 0}</td>
          </tr>
          <tr>
            <td className="px-2 py-1 text-slate-400 font-medium">Act Valid</td>
            <td className="px-2 py-1 text-center">{cm[1]?.[0] ?? 0}</td>
            <td className="px-2 py-1 text-center bg-cyan-500/15">{cm[1]?.[1] ?? 0}</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

function accColor(value: number | undefined): string {
  if (value == null) return 'text-slate-500';
  const pct = value * 100;
  if (pct >= 90) return 'text-green-400';
  if (pct >= 75) return 'text-yellow-400';
  return 'text-red-400';
}

function fmtPct(value: number | undefined): string {
  if (value == null) return '—';
  return `${(value * 100).toFixed(1)}%`;
}

export default function TrainingMetrics({ metrics }: Props) {
  const entries = Object.entries(metrics);
  const columnsWithMatrix = entries.filter(([, m]) => m.test_confusion_matrix);

  return (
    <div>
      <h3 className="text-lg font-bold mb-4">Training Results</h3>

      <div className="glass-card overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-white text-xs uppercase border-b border-white/10">
              <th className="text-left px-4 py-3 font-semibold tracking-wide">Column</th>
              <th className="text-right px-4 py-3 font-semibold tracking-wide">Samples</th>
              <th className="text-right px-4 py-3 font-semibold tracking-wide">Train Acc</th>
              <th className="text-right px-4 py-3 font-semibold tracking-wide">Test Acc</th>
              <th className="text-right px-4 py-3 font-semibold tracking-wide">Test F1</th>
              <th className="text-right px-4 py-3 font-semibold tracking-wide">Best C</th>
              <th className="text-right px-4 py-3 font-semibold tracking-wide">CV F1</th>
              <th className="text-center px-4 py-3 font-semibold tracking-wide">Split</th>
            </tr>
          </thead>
          <tbody>
            {entries.map(([colName, m]) => (
              <tr
                key={colName}
                className="border-b border-white/5 hover:bg-white/5 transition-colors"
              >
                <td className="px-4 py-2.5">
                  <span className="font-mono text-cyan-300">{colName}</span>
                </td>
                <td className="px-4 py-2.5 text-right text-slate-300">
                  {m.total_samples ?? '—'}
                </td>
                <td className={`px-4 py-2.5 text-right font-medium ${accColor(m.train_accuracy)}`}>
                  {fmtPct(m.train_accuracy)}
                </td>
                <td className={`px-4 py-2.5 text-right font-medium ${m.used_split ? accColor(m.test_accuracy) : 'text-slate-500'}`}>
                  {m.used_split ? fmtPct(m.test_accuracy) : '—'}
                </td>
                <td className={`px-4 py-2.5 text-right font-medium ${m.used_split ? accColor(m.test_f1) : 'text-slate-500'}`}>
                  {m.used_split ? fmtPct(m.test_f1) : '—'}
                </td>
                <td className="px-4 py-2.5 text-right text-slate-300">
                  {m.best_C ?? '—'}
                </td>
                <td className={`px-4 py-2.5 text-right font-medium ${m.used_split ? accColor(m.cv_f1_score) : 'text-slate-500'}`}>
                  {m.used_split ? fmtPct(m.cv_f1_score) : '—'}
                </td>
                <td className="px-4 py-2.5 text-center">
                  {m.used_split ? (
                    <span className="inline-block px-2 py-0.5 rounded-full text-xs bg-green-500/20 text-green-400 border border-green-500/30">
                      yes
                    </span>
                  ) : (
                    <span className="inline-block px-2 py-0.5 rounded-full text-xs bg-yellow-500/20 text-yellow-400 border border-yellow-500/30">
                      small dataset
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {columnsWithMatrix.length > 0 && (
        <div className="mt-4 space-y-2">
          {columnsWithMatrix.map(([colName, m]) => (
            <Collapsible
              key={colName}
              title={`Confusion Matrix: ${colName}`}
              defaultOpen={false}
            >
              <ConfusionMatrix cm={m.test_confusion_matrix!} />
            </Collapsible>
          ))}
        </div>
      )}
    </div>
  );
}
