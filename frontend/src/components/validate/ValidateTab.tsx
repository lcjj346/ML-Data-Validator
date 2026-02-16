import { useCallback, useState } from 'react';
import { useValidation } from '../../hooks/useValidation';
import { useToast } from '../../hooks/useToast';
import Toast from '../Toast';
import ProgressBar from '../ProgressBar';
import FileUpload from './FileUpload';
import FileInfo from './FileInfo';
import DataPreview from './DataPreview';
import ModelSelector from './ModelSelector';
import ColumnMatching from './ColumnMatching';
import QualityMetrics from './QualityMetrics';
import ValidationGrid from './ValidationGrid';
import CorrectionsPanel from './CorrectionsPanel';
import ExportSection from './ExportSection';

function StepBadge({ n }: { n: number }) {
  return (
    <span className="inline-flex items-center justify-center w-7 h-7 bg-gradient-to-br from-indigo-500 to-purple-600 text-white rounded-full text-xs font-bold mr-2 shadow-lg shadow-indigo-500/20">
      {n}
    </span>
  );
}

export default function ValidateTab() {
  const v = useValidation();
  const { toasts, addToast, removeToast } = useToast();
  const [applyingAll, setApplyingAll] = useState(false);
  const [uploadKey, setUploadKey] = useState(0);

  const handleFile = useCallback(
    async (file: File) => {
      v.reset();
      const info = await v.upload(file);
      if (info) {
        setUploadKey((k) => k + 1);
      }
    },
    [v],
  );

  const handleModelSelect = useCallback(
    async (name: string) => {
      v.setSelectedModel(name);
      if (v.fileInfo) {
        await v.matchColumns(v.fileInfo.session_id, name);
      }
    },
    [v],
  );

  const handleValidate = useCallback(async () => {
    if (!v.fileInfo || !v.selectedModel) return;
    await v.validate(v.fileInfo.session_id, v.selectedModel);
  }, [v]);

  const handleCellEdit = useCallback(
    (row: number, column: string, value: unknown) => {
      if (!v.fileInfo) return;
      v.editCell(v.fileInfo.session_id, row, column, value);
    },
    [v],
  );

  const handleApplySingle = useCallback(
    async (correction: Parameters<typeof v.applySingleCorrection>[1]) => {
      if (!v.fileInfo) return;
      const res = await v.applySingleCorrection(v.fileInfo.session_id, correction);
      if (res) {
        addToast(`Applied: ${correction.original} -> ${correction.suggested}`, 'success');
      }
    },
    [v, addToast],
  );

  const handleApplyAll = useCallback(async () => {
    if (!v.fileInfo) return;
    setApplyingAll(true);
    const res = await v.applyAll(v.fileInfo.session_id);
    setApplyingAll(false);
    if (res) {
      addToast(`Applied ${res.applied} corrections`, 'success');
    }
  }, [v, addToast]);

  return (
    <div className="animate-fadeIn">
      <Toast toasts={toasts} onRemove={removeToast} />

      {/* Step 1: Upload */}
      <div className="mb-8">
        <div className="flex items-center mb-3">
          <StepBadge n={1} />
          <span className="text-sm font-semibold text-gray-200">Upload your CSV file</span>
        </div>
        <FileUpload onFile={handleFile} disabled={v.isValidating} />
      </div>

      {v.fileInfo && (
        <>
          <FileInfo filename={v.fileInfo.filename} rows={v.fileInfo.rows} columns={v.fileInfo.columns} />
          <DataPreview preview={v.fileInfo.preview} columns={v.fileInfo.column_names} />

          <div className="border-t border-white/5 my-8" />

          {/* Step 2: Select Model */}
          <div className="mb-8">
            <div className="flex items-center mb-3">
              <StepBadge n={2} />
              <span className="text-sm font-semibold text-gray-200">Select trained model</span>
            </div>
            <ModelSelector key={uploadKey} selected={v.selectedModel} onSelect={handleModelSelect} />
          </div>

          {v.matchResult && <ColumnMatching match={v.matchResult} />}

          <div className="border-t border-white/5 my-8" />

          {/* Step 3: Validate */}
          <div className="mb-8">
            <div className="flex items-center mb-3">
              <StepBadge n={3} />
              <span className="text-sm font-semibold text-gray-200">Run validation</span>
            </div>
            <button
              onClick={handleValidate}
              disabled={!v.selectedModel || v.isValidating}
              className="w-full py-3 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold rounded-xl transition-all duration-200 shadow-lg shadow-indigo-500/20 hover:shadow-indigo-500/30 hover:scale-[1.01] active:scale-[0.99]"
            >
              {v.isValidating ? 'Validating...' : 'Validate'}
            </button>
          </div>

          {v.isValidating && <ProgressBar progress={v.progress} message={v.progressMessage} />}

          {v.error && (
            <div className="glass-card border-red-500/30 p-4 text-red-300 text-sm mb-4">
              {v.error}
            </div>
          )}

          {/* Results */}
          {v.results && (
            <>
              <div className="border-t border-white/5 my-8" />
              <QualityMetrics
                validCells={v.results.valid_cells}
                invalidCells={v.results.invalid_cells}
                quality={v.results.quality}
              />
              <ValidationGrid
                data={v.results.data}
                columnNames={v.fileInfo.column_names}
                matchedColumns={Object.keys(v.matchResult?.matched ?? {})}
                cellValidity={v.results.cell_validity}
                modifiedCells={v.results.modified_cells}
                onCellEdit={handleCellEdit}
              />
              {v.results.corrections.length > 0 && (
                <CorrectionsPanel
                  corrections={v.results.corrections}
                  modifiedCells={v.results.modified_cells}
                  onApplySingle={handleApplySingle}
                  onApplyAll={handleApplyAll}
                  applying={applyingAll}
                />
              )}
              <ExportSection sessionId={v.fileInfo.session_id} totalRows={v.results.data.length} />
            </>
          )}
        </>
      )}
    </div>
  );
}
