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
    <span className="inline-block bg-gradient-to-br from-indigo-500 to-purple-600 text-white px-3 py-0.5 rounded-full text-xs font-semibold mr-2">
      Step {n}
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

  // Refresh models list when training completes
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
    <div>
      <Toast toasts={toasts} onRemove={removeToast} />

      <h2 className="text-xl font-semibold mb-4">Train Your ML Models</h2>

      {/* Step 1: Upload */}
      <div className="mb-6">
        <p className="text-sm font-medium mb-2">
          <StepBadge n={1} /> <strong>Upload your data CSV</strong>
        </p>
        <TrainingUpload onFile={handleFile} disabled={t.isTraining} />
      </div>

      {t.uploadInfo && (
        <>
          <FileInfo filename={t.uploadInfo.filename} rows={t.uploadInfo.rows} columns={t.uploadInfo.columns} />
          <DataPreview preview={t.uploadInfo.preview} columns={t.uploadInfo.column_names} />

          <hr className="border-gray-700 my-6" />

          {/* Step 2: Column config */}
          <div className="mb-6">
            <p className="text-sm font-medium mb-2">
              <StepBadge n={2} /> <strong>Configure columns</strong>
            </p>
            <ColumnConfig
              allColumns={t.uploadInfo.column_names}
              excluded={excludeCols}
              onChange={setExcludeCols}
            />
          </div>

          <hr className="border-gray-700 my-6" />

          {/* Step 3: Training mode */}
          <div className="mb-6">
            <p className="text-sm font-medium mb-2">
              <StepBadge n={3} /> <strong>Choose training mode</strong>
            </p>
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
            className="w-full py-3 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium rounded-lg transition-colors mb-4"
          >
            {t.isTraining
              ? 'Training...'
              : mode === 'new'
                ? 'Train Model'
                : 'Add Data & Retrain'}
          </button>

          {t.isTraining && <ProgressBar progress={t.progress} message={t.progressMessage} />}

          {t.error && (
            <div className="bg-red-900/30 border border-red-700 rounded-lg p-3 text-red-300 text-sm mb-4">
              {t.error}
            </div>
          )}

          {t.metrics && (
            <>
              <hr className="border-gray-700 my-6" />
              <TrainingMetricsView metrics={t.metrics} />
            </>
          )}
        </>
      )}

      <hr className="border-gray-700 my-6" />
      <ModelsList refreshKey={modelsRefresh} />
    </div>
  );
}
