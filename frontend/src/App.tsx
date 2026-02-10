import { useState } from 'react';
import ValidateTab from './components/validate/ValidateTab';
import TrainTab from './components/train/TrainTab';

type Tab = 'validate' | 'train';

export default function App() {
  const [tab, setTab] = useState<Tab>('validate');

  return (
    <div className="max-w-7xl mx-auto px-4 py-6">
      <h1 className="text-3xl font-bold mb-1">ML Data Validator</h1>
      <p className="text-gray-400 text-sm mb-6">
        Train on your data, validate any column type, suggest corrections.
      </p>

      {/* Tab bar */}
      <div className="flex gap-1 border-b border-gray-700 mb-6">
        <button
          onClick={() => setTab('validate')}
          className={`px-5 py-2.5 text-sm font-medium rounded-t-lg transition-colors ${
            tab === 'validate'
              ? 'bg-gray-800 text-white border-b-2 border-indigo-500'
              : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/50'
          }`}
        >
          Validate Data
        </button>
        <button
          onClick={() => setTab('train')}
          className={`px-5 py-2.5 text-sm font-medium rounded-t-lg transition-colors ${
            tab === 'train'
              ? 'bg-gray-800 text-white border-b-2 border-indigo-500'
              : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/50'
          }`}
        >
          Train Models
        </button>
      </div>

      {tab === 'validate' ? <ValidateTab /> : <TrainTab />}
    </div>
  );
}
