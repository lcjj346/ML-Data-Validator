import { useEffect, useState } from 'react';
import type { ModelInfo } from '../../types';
import { listModels } from '../../api/client';

interface Props {
  mode: 'new' | 'existing';
  onModeChange: (mode: 'new' | 'existing') => void;
  modelName: string;
  onModelNameChange: (name: string) => void;
  useReferenceLists: boolean;
  onReferenceListsChange: (v: boolean) => void;
}

export default function TrainingMode({
  mode,
  onModeChange,
  modelName,
  onModelNameChange,
  useReferenceLists,
  onReferenceListsChange,
}: Props) {
  const [existingModels, setExistingModels] = useState<ModelInfo[]>([]);

  useEffect(() => {
    listModels().then(setExistingModels);
  }, []);

  return (
    <div className="space-y-4">
      {/* Mode selector */}
      <div className="flex gap-3">
        {(['new', 'existing'] as const).map((m) => (
          <button
            key={m}
            onClick={() => onModeChange(m)}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all duration-200 ${
              mode === m
                ? 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/40'
                : 'bg-gray-800/60 text-gray-400 border border-white/5 hover:border-white/15'
            }`}
          >
            <span className={`w-3 h-3 rounded-full border-2 flex items-center justify-center ${
              mode === m ? 'border-indigo-400' : 'border-gray-500'
            }`}>
              {mode === m && <span className="w-1.5 h-1.5 rounded-full bg-indigo-400" />}
            </span>
            {m === 'new' ? 'Create new model' : 'Add data to existing model'}
          </button>
        ))}
      </div>

      {/* Model name input / selector */}
      {mode === 'new' ? (
        <input
          type="text"
          value={modelName}
          onChange={(e) => onModelNameChange(e.target.value)}
          placeholder="e.g., customer_data, sales_records"
          className="w-full bg-gray-800/80 border border-white/10 rounded-xl px-4 py-2.5 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500/50 transition-all duration-200 placeholder-gray-500"
        />
      ) : existingModels.length === 0 ? (
        <div className="glass-card border-yellow-500/30 p-4 text-yellow-300 text-sm">
          No existing models found. Please create a new model first.
        </div>
      ) : (
        <select
          value={modelName}
          onChange={(e) => onModelNameChange(e.target.value)}
          className="w-full bg-gray-800/80 border border-white/10 rounded-xl px-3 py-2.5 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500/50 transition-all duration-200"
        >
          <option value="">Select model...</option>
          {existingModels.map((m) => (
            <option key={m.name} value={m.name}>{m.name}</option>
          ))}
        </select>
      )}

      {/* Reference lists toggle */}
      <label className="flex items-center gap-3 cursor-pointer group">
        <div className={`relative w-9 h-5 rounded-full transition-colors duration-200 ${
          useReferenceLists ? 'bg-indigo-600' : 'bg-gray-600'
        }`}
          onClick={(e) => { e.preventDefault(); onReferenceListsChange(!useReferenceLists); }}
        >
          <div className={`absolute top-0.5 w-4 h-4 bg-white rounded-full shadow-sm transition-transform duration-200 ${
            useReferenceLists ? 'translate-x-4' : 'translate-x-0.5'
          }`} />
        </div>
        <span className="text-sm text-gray-300 group-hover:text-gray-200 transition-colors">
          Load reference lists for standard fields (countries, etc.)
        </span>
      </label>
    </div>
  );
}
