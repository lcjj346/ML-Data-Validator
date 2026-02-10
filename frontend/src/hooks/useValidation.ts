import { useState, useCallback, useRef } from 'react';
import type {
  FileInfo,
  MatchColumnsResponse,
  ValidationResults,
  CorrectionItem,
  SSEEvent,
} from '../types';
import * as api from '../api/client';

export function useValidation() {
  const [fileInfo, setFileInfo] = useState<FileInfo | null>(null);
  const [matchResult, setMatchResult] = useState<MatchColumnsResponse | null>(null);
  const [selectedModel, setSelectedModel] = useState('');
  const [isValidating, setIsValidating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState('');
  const [results, setResults] = useState<ValidationResults | null>(null);
  const [error, setError] = useState<string | null>(null);

  const cancelRef = useRef<(() => void) | null>(null);

  const upload = useCallback(async (file: File) => {
    setError(null);
    setResults(null);
    setMatchResult(null);
    try {
      const info = await api.uploadFile(file);
      setFileInfo(info);
      return info;
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Upload failed';
      setError(msg);
      return null;
    }
  }, []);

  const matchColumns = useCallback(async (sessionId: string, modelName: string) => {
    setError(null);
    try {
      const result = await api.matchColumns(sessionId, modelName);
      setMatchResult(result);
      return result;
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Column matching failed';
      setError(msg);
      return null;
    }
  }, []);

  const validate = useCallback(async (sessionId: string, modelName: string) => {
    setIsValidating(true);
    setProgress(0);
    setProgressMessage('Starting...');
    setError(null);

    const cancel = api.runValidation(sessionId, modelName, async (event: SSEEvent) => {
      if (event.type === 'progress') {
        setProgress(event.progress);
        setProgressMessage(event.message);
      } else if (event.type === 'done') {
        // Fetch full results
        try {
          const res = await api.getResults(sessionId);
          setResults(res);
        } catch (e: unknown) {
          setError(e instanceof Error ? e.message : 'Failed to fetch results');
        }
        setIsValidating(false);
      } else if (event.type === 'error') {
        setError(event.message);
        setIsValidating(false);
      }
    });

    cancelRef.current = cancel;
  }, []);

  const editCell = useCallback(async (sessionId: string, row: number, column: string, value: unknown) => {
    try {
      const res = await api.editCell(sessionId, row, column, value);
      // Update local state
      if (results) {
        const newModified = [...results.modified_cells, res.key];
        setResults({ ...results, modified_cells: newModified });
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Edit failed');
    }
  }, [results]);

  const applySingleCorrection = useCallback(async (sessionId: string, correction: CorrectionItem) => {
    try {
      const res = await api.applyCorrection(sessionId, correction.row_index, correction.column, correction.suggested);
      // Refresh results after correction
      const updated = await api.getResults(sessionId);
      setResults(updated);
      return res;
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Correction failed');
      return null;
    }
  }, []);

  const applyAll = useCallback(async (sessionId: string) => {
    try {
      const res = await api.applyAllCorrections(sessionId);
      const updated = await api.getResults(sessionId);
      setResults(updated);
      return res;
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Apply all failed');
      return null;
    }
  }, []);

  const reset = useCallback(() => {
    if (cancelRef.current) cancelRef.current();
    setFileInfo(null);
    setMatchResult(null);
    setSelectedModel('');
    setIsValidating(false);
    setProgress(0);
    setProgressMessage('');
    setResults(null);
    setError(null);
  }, []);

  return {
    fileInfo,
    matchResult,
    selectedModel,
    setSelectedModel,
    isValidating,
    progress,
    progressMessage,
    results,
    error,
    upload,
    matchColumns,
    validate,
    editCell,
    applySingleCorrection,
    applyAll,
    reset,
  };
}
