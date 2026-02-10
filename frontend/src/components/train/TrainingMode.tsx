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
      <div className="flex gap-4">
        {(['new', 'existing'] as const).map((m) => (
          <label key={m} className="flex items-center gap-2 cursor-pointer">
            <input
              type="radio"
              name="training-mode"
              checked={mode === m}
              onChange={() => onModeChange(m)}
              className="text-indigo-500"
            />
            <span className="text-sm">
              {m === 'new' ? 'Create new model' : 'Add data to existing model'}
            </span>
          </label>
        ))}
      </div>

      {/* Model name input / selector */}
      {mode === 'new' ? (
        <input
          type="text"
          value={modelName}
          onChange={(e) => onModelNameChange(e.target.value)}
          placeholder="e.g., customer_data, sales_records"
          className="w-full bg-gray-800 border border-gray-600 rounded-lg px-3 py-2.5 text-sm focus:border-indigo-500 focus:outline-none"
        />
      ) : existingModels.length === 0 ? (
        <div className="bg-yellow-900/30 border border-yellow-700 rounded-lg p-3 text-yellow-300 text-sm">
          No existing models found. Please create a new model first.
        </div>
      ) : (
        <select
          value={modelName}
          onChange={(e) => onModelNameChange(e.target.value)}
          className="w-full bg-gray-800 border border-gray-600 rounded-lg px-3 py-2.5 text-sm focus:border-indigo-500 focus:outline-none"
        >
          <option value="">Select model...</option>
          {existingModels.map((m) => (
            <option key={m.name} value={m.name}>{m.name}</option>
          ))}
        </select>
      )}

      {/* Reference lists toggle */}
      <label className="flex items-center gap-2 cursor-pointer">
        <input
          type="checkbox"
          checked={useReferenceLists}
          onChange={(e) => onReferenceListsChange(e.target.checked)}
          className="rounded text-indigo-500"
        />
        <span className="text-sm text-gray-300">
          Load reference lists for standard fields (countries, etc.)
        </span>
      </label>
    </div>
  );
}
