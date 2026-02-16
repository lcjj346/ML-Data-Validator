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
    setLoading(true);
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
      <div className="glass-card border-yellow-500/30 p-4 text-yellow-300 text-sm">
        No trained models found. Go to "Train Models" tab first.
      </div>
    );
  }

  return (
    <div className="relative">
      <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
      </svg>
      <select
        value={selected}
        onChange={(e) => onSelect(e.target.value)}
        className="w-full bg-gray-800/80 border border-white/10 rounded-xl pl-9 pr-3 py-2.5 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500/50 appearance-none cursor-pointer transition-all duration-200 hover:border-white/20"
      >
        {models.map((m) => (
          <option key={m.name} value={m.name}>
            {m.name}
          </option>
        ))}
      </select>
      <svg className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
      </svg>
    </div>
  );
}
