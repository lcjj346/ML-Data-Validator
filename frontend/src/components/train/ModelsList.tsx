import { useEffect, useState } from 'react';
import type { ModelInfo } from '../../types';
import { listModels } from '../../api/client';
import Collapsible from '../Collapsible';

interface Props {
  refreshKey?: number; // increment to force refresh
}

export default function ModelsList({ refreshKey }: Props) {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    listModels()
      .then(setModels)
      .finally(() => setLoading(false));
  }, [refreshKey]);

  return (
    <Collapsible title="Trained Models" defaultOpen={true}>
      {loading ? (
        <p className="text-gray-400 text-sm">Loading...</p>
      ) : models.length === 0 ? (
        <p className="text-gray-400 text-sm">No trained models yet. Upload training data above to get started.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left">
            <thead className="text-xs uppercase text-gray-400 bg-gray-800">
              <tr>
                <th className="px-3 py-2">Name</th>
                <th className="px-3 py-2">Size</th>
                <th className="px-3 py-2">Columns</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <tr key={m.name} className="border-b border-gray-800 hover:bg-gray-800/50">
                  <td className="px-3 py-2 font-medium">{m.name}</td>
                  <td className="px-3 py-2 text-gray-400">{m.size}</td>
                  <td className="px-3 py-2 text-gray-400">{m.columns}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Collapsible>
  );
}
