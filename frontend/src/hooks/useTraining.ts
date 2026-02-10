import { useState, useCallback, useRef } from 'react';
import type { TrainUploadInfo, TrainingMetrics, SSEEvent } from '../types';
import * as api from '../api/client';

export function useTraining() {
  const [uploadInfo, setUploadInfo] = useState<TrainUploadInfo | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState('');
  const [metrics, setMetrics] = useState<TrainingMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);

  const cancelRef = useRef<(() => void) | null>(null);

  const upload = useCallback(async (file: File) => {
    setError(null);
    setMetrics(null);
    try {
      const info = await api.uploadTrainingFile(file);
      setUploadInfo(info);
      return info;
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Upload failed');
      return null;
    }
  }, []);

  const train = useCallback(
    (
      sessionId: string,
      modelName: string,
      trainingMode: string,
      excludeColumns: string[],
      useReferenceLists: boolean,
    ) => {
      setIsTraining(true);
      setProgress(0);
      setProgressMessage('Starting...');
      setError(null);
      setMetrics(null);

      const cancel = api.runTraining(
        sessionId,
        modelName,
        trainingMode,
        excludeColumns,
        useReferenceLists,
        (event: SSEEvent) => {
          if (event.type === 'progress') {
            setProgress(event.progress);
            setProgressMessage(event.message);
          } else if (event.type === 'done') {
            if (event.metrics) {
              setMetrics(event.metrics);
            }
            setIsTraining(false);
          } else if (event.type === 'error') {
            setError(event.message);
            setIsTraining(false);
          }
        },
      );

      cancelRef.current = cancel;
    },
    [],
  );

  const reset = useCallback(() => {
    if (cancelRef.current) cancelRef.current();
    setUploadInfo(null);
    setIsTraining(false);
    setProgress(0);
    setProgressMessage('');
    setMetrics(null);
    setError(null);
  }, []);

  return {
    uploadInfo,
    isTraining,
    progress,
    progressMessage,
    metrics,
    error,
    upload,
    train,
    reset,
  };
}
