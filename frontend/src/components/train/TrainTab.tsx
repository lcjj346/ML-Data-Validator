import { useCallback, useEffect, useRef, useState } from 'react';
import { useTraining } from '../../hooks/useTraining';
import { useToast } from '../../hooks/useToast';
import Toast from '../Toast';
import ProgressBar from '../ProgressBar';
import FileInfo from '../validate/FileInfo';
import DataPreview from '../validate/DataPreview';
import TrainingUpload from './TrainingUpload';
import ColumnConfig from './ColumnConfig';
import TrainingMode from './TrainingMode';
import TrainingMetricsView from './TrainingMetrics';
import ModelsList from './ModelsList';

function StepBadge({ n }: { n: number }) {
  return (
    <span className="inline-flex items-center justify-center w-7 h-7 bg-gradient-to-br from-indigo-500 to-purple-600 text-white rounded-full text-xs font-bold mr-2 shadow-lg shadow-indigo-500/20">
      {n}
    </span>
  );
}

export default function TrainTab() {
  const t = useTraining();
  const { toasts, addToast, removeToast } = useToast();

  const [excludeCols, setExcludeCols] = useState<string[]>([]);
  const [mode, setMode] = useState<'new' | 'existing'>('new');
  const [modelName, setModelName] = useState('my_model');
  const [useRefLists, setUseRefLists] = useState(true);
  const [modelsRefresh, setModelsRefresh] = useState(0);

  const handleFile = useCallback(
    async (file: File) => {
      t.reset();
      setExcludeCols([]);
      await t.upload(file);
    },
    [t],
  );

  const handleTrain = useCallback(() => {
    if (!t.uploadInfo) return;
    if (!modelName.trim()) {
      addToast('Please enter a model name', 'error');
      return;
    }
    const included = t.uploadInfo.column_names.filter((c) => !excludeCols.includes(c));
    if (included.length === 0) {
      addToast('No columns selected for training', 'error');
      return;
    }

    t.train(t.uploadInfo.session_id, modelName.trim(), mode, excludeCols, useRefLists);
  }, [t, modelName, mode, excludeCols, useRefLists, addToast]);

  const prevIsTraining = useRef(false);
  useEffect(() => {
    if (prevIsTraining.current && !t.isTraining && t.metrics) {
      setModelsRefresh((n) => n + 1);
      addToast(
        mode === 'new' ? `Model "${modelName}" trained and saved!` : `Model "${modelName}" updated!`,
        'success',
      );
    }
    prevIsTraining.current = t.isTraining;
  }, [t.isTraining, t.metrics, mode, modelName, addToast]);

  return (
    <div className="animate-fadeIn">
      <Toast toasts={toasts} onRemove={removeToast} />

      <h2 className="text-xl font-bold mb-6">Train Your ML Models</h2>

      {/* Step 1: Upload */}
      <div className="mb-8">
        <div className="flex items-center mb-3">
          <StepBadge n={1} />
          <span className="text-sm font-semibold text-gray-200">Upload your data CSV</span>
        </div>
        <TrainingUpload onFile={handleFile} disabled={t.isTraining} />
      </div>

      {t.uploadInfo && (
        <>
          <FileInfo filename={t.uploadInfo.filename} rows={t.uploadInfo.rows} columns={t.uploadInfo.columns} />
          <DataPreview preview={t.uploadInfo.preview} columns={t.uploadInfo.column_names} />

          <div className="border-t border-white/5 my-8" />

          {/* Step 2: Column config */}
          <div className="mb-8">
            <div className="flex items-center mb-3">
              <StepBadge n={2} />
              <span className="text-sm font-semibold text-gray-200">Configure columns</span>
            </div>
            <ColumnConfig
              allColumns={t.uploadInfo.column_names}
              excluded={excludeCols}
              onChange={setExcludeCols}
            />
          </div>

          <div className="border-t border-white/5 my-8" />

          {/* Step 3: Training mode */}
          <div className="mb-8">
            <div className="flex items-center mb-3">
              <StepBadge n={3} />
              <span className="text-sm font-semibold text-gray-200">Choose training mode</span>
            </div>
            <TrainingMode
              mode={mode}
              onModeChange={setMode}
              modelName={modelName}
              onModelNameChange={setModelName}
              useReferenceLists={useRefLists}
              onReferenceListsChange={setUseRefLists}
            />
          </div>

          {/* Train button */}
          <button
            onClick={handleTrain}
            disabled={t.isTraining || !modelName.trim()}
            className="w-full py-3 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold rounded-xl transition-all duration-200 shadow-lg shadow-indigo-500/20 hover:shadow-indigo-500/30 hover:scale-[1.01] active:scale-[0.99] mb-4"
          >
            {t.isTraining
              ? 'Training...'
              : mode === 'new'
                ? 'Train Model'
                : 'Add Data & Retrain'}
          </button>

          {t.isTraining && <ProgressBar progress={t.progress} message={t.progressMessage} />}

          {t.error && (
            <div className="glass-card border-red-500/30 p-4 text-red-300 text-sm mb-4">
              {t.error}
            </div>
          )}

          {t.metrics && (
            <>
              <div className="border-t border-white/5 my-8" />
              <TrainingMetricsView metrics={t.metrics} />
            </>
          )}
        </>
      )}

      <div className="border-t border-white/5 my-8" />
      <ModelsList refreshKey={modelsRefresh} />
    </div>
  );
}
