import { useEffect, useState } from 'react';
import type { ModelInfo } from '../../types';
import { listModels } from '../../api/client';

interface Props {
  selected: string;
  onSelect: (name: string) => void;
}

export default function ModelSelector({ selected, onSelect }: Props) {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    listModels()
      .then((m) => {
        setModels(m);
        if (m.length && !selected) onSelect(m[0].name);
      })
      .finally(() => setLoading(false));
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  if (loading) return <p className="text-gray-400 text-sm">Loading models...</p>;
  if (!models.length) {
    return (
      <div className="bg-yellow-900/30 border border-yellow-700 rounded-lg p-3 text-yellow-300 text-sm">
        No trained models found. Go to "Train Models" tab first.
      </div>
    );
  }

  return (
    <select
      value={selected}
      onChange={(e) => onSelect(e.target.value)}
      className="w-full bg-gray-800 border border-gray-600 rounded-lg px-3 py-2.5 text-sm focus:border-indigo-500 focus:outline-none"
    >
      {models.map((m) => (
        <option key={m.name} value={m.name}>
          {m.name}
        </option>
      ))}
    </select>
  );
}
