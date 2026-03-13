import { useState } from 'react';
import ValidateTab from './components/validate/ValidateTab';
import TrainTab from './components/train/TrainTab';

type Tab = 'validate' | 'train';

export default function App() {
  const [tab, setTab] = useState<Tab>('validate');

  return (
    <div className="max-w-[95vw] mx-auto px-6 py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-extrabold tracking-tight mb-1 bg-gradient-to-r from-cyan-200 to-slate-400 bg-clip-text text-transparent">
          ML Data Validator
        </h1>
        <p className="text-slate-500 text-sm">
          Train on your data, validate any column type, suggest corrections.
        </p>
      </div>

      {/* Pill tab bar */}
      <div className="inline-flex bg-slate-800/60 backdrop-blur-sm rounded-xl p-1 mb-8 border border-white/5">
        <button
          onClick={() => setTab('validate')}
          className={`px-5 py-2 text-sm font-medium rounded-lg transition-all duration-200 ${
            tab === 'validate'
              ? 'bg-gradient-to-r from-cyan-500 to-sky-600 text-white shadow-lg shadow-cyan-500/20'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          Validate Data
        </button>
        <button
          onClick={() => setTab('train')}
          className={`px-5 py-2 text-sm font-medium rounded-lg transition-all duration-200 ${
            tab === 'train'
              ? 'bg-gradient-to-r from-cyan-500 to-sky-600 text-white shadow-lg shadow-cyan-500/20'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          Train Models
        </button>
      </div>

      {tab === 'validate' ? <ValidateTab /> : <TrainTab />}
    </div>
  );
}
