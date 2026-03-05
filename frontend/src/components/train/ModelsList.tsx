import { useEffect, useState } from 'react';
import type { ModelInfo } from '../../types';
import { listModels, deleteModel } from '../../api/client';
import Collapsible from '../Collapsible';

interface Props {
  refreshKey?: number;
}

export default function ModelsList({ refreshKey }: Props) {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    listModels()
      .then(setModels)
      .finally(() => setLoading(false));
  }, [refreshKey]);

  const handleDelete = async (name: string) => {
    if (!window.confirm(`Delete model "${name}"? This cannot be undone.`)) return;
    setDeleting(name);
    setError(null);
    try {
      await deleteModel(name);
      setModels((prev) => prev.filter((m) => m.name !== name));
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Delete failed');
    } finally {
      setDeleting(null);
    }
  };

  return (
    <Collapsible title="Trained Models" defaultOpen={true}>
      {error && (
        <div className="glass-card border-red-500/30 p-3 text-red-300 text-sm mb-3">{error}</div>
      )}
      {loading ? (
        <p className="text-gray-400 text-sm">Loading...</p>
      ) : models.length === 0 ? (
        <p className="text-gray-400 text-sm">No trained models yet. Upload training data above to get started.</p>
      ) : (
        <div className="overflow-x-auto rounded-lg">
          <table className="w-full text-sm text-left">
            <thead className="text-xs uppercase text-gray-400 bg-gray-800/60">
              <tr>
                <th className="px-3 py-2.5 font-medium">Name</th>
                <th className="px-3 py-2.5 font-medium">Size</th>
                <th className="px-3 py-2.5 font-medium">Columns</th>
                <th className="px-3 py-2.5 font-medium"></th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <tr key={m.name} className="border-b border-white/5 hover:bg-white/5 transition-colors">
                  <td className="px-3 py-2.5 font-medium">
                    {m.name}
                    {m.name === 'base_model' && (
                      <span className="ml-2 text-xs px-1.5 py-0.5 rounded bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
                        default
                      </span>
                    )}
                  </td>
                  <td className="px-3 py-2.5 text-gray-400">{m.size}</td>
                  <td className="px-3 py-2.5 text-gray-400">{m.columns}</td>
                  <td className="px-3 py-2.5 text-right">
                    {m.name !== 'base_model' && (
                      <button
                        onClick={() => handleDelete(m.name)}
                        disabled={deleting === m.name}
                        className="px-2.5 py-1 text-xs text-red-400 border border-red-500/30 rounded-lg hover:bg-red-500/10 hover:text-red-300 transition-all duration-200 disabled:opacity-50"
                      >
                        {deleting === m.name ? 'Deleting...' : 'Delete'}
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Collapsible>
  );
}
