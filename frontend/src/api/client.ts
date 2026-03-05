import type {
  FileInfo,
  MatchColumnsResponse,
  ValidationResults,
  TrainUploadInfo,
  ModelInfo,
  SSEEvent,
} from '../types';

const BASE = '/api';

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || `Request failed: ${res.status}`);
  }
  return res.json();
}

// ── Validation ──────────────────────────────────────────────

export async function uploadFile(file: File): Promise<FileInfo> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${BASE}/validate/upload`, { method: 'POST', body: form });
  return handleResponse<FileInfo>(res);
}

export async function matchColumns(sessionId: string, modelName: string): Promise<MatchColumnsResponse> {
  const res = await fetch(`${BASE}/validate/match-columns`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, model_name: modelName }),
  });
  return handleResponse<MatchColumnsResponse>(res);
}

export function runValidation(
  sessionId: string,
  modelName: string,
  onEvent: (event: SSEEvent) => void,
): () => void {
  const url = `${BASE}/validate/${sessionId}/run?model_name=${encodeURIComponent(modelName)}`;
  const source = new EventSource(url);

  source.onmessage = (e) => {
    try {
      const data: SSEEvent = JSON.parse(e.data);
      onEvent(data);
      if (data.type === 'done' || data.type === 'error') {
        source.close();
      }
    } catch {
      // ignore parse errors
    }
  };

  source.onerror = () => {
    source.close();
    onEvent({ type: 'error', message: 'Connection lost' });
  };

  return () => source.close();
}

export async function getResults(sessionId: string): Promise<ValidationResults> {
  const res = await fetch(`${BASE}/validate/${sessionId}/results`);
  return handleResponse<ValidationResults>(res);
}

export async function editCell(sessionId: string, row: number, column: string, value: unknown) {
  const res = await fetch(`${BASE}/validate/${sessionId}/edit-cell`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ row, column, value }),
  });
  return handleResponse<{ ok: boolean; key: string }>(res);
}

export async function applyCorrection(sessionId: string, row: number, column: string, suggested: string) {
  const res = await fetch(`${BASE}/validate/${sessionId}/apply-correction`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ row, column, suggested }),
  });
  return handleResponse<{ ok: boolean; key: string }>(res);
}

export async function applyAllCorrections(sessionId: string) {
  const res = await fetch(`${BASE}/validate/${sessionId}/apply-all`, { method: 'POST' });
  return handleResponse<{ ok: boolean; applied: number }>(res);
}

export function getExportUrl(sessionId: string): string {
  return `${BASE}/validate/${sessionId}/export`;
}

export function getSummaryReportUrl(sessionId: string): string {
  return `${BASE}/validate/${sessionId}/export-report`;
}

// ── Training ────────────────────────────────────────────────

export async function uploadTrainingFile(file: File): Promise<TrainUploadInfo> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${BASE}/train/upload`, { method: 'POST', body: form });
  return handleResponse<TrainUploadInfo>(res);
}

export function runTraining(
  sessionId: string,
  modelName: string,
  trainingMode: string,
  excludeColumns: string[],
  useReferenceLists: boolean,
  onEvent: (event: SSEEvent) => void,
): () => void {
  const params = new URLSearchParams({
    model_name: modelName,
    training_mode: trainingMode,
    exclude_columns: excludeColumns.join(','),
    use_reference_lists: String(useReferenceLists),
  });
  const url = `${BASE}/train/${sessionId}/run?${params}`;
  const source = new EventSource(url);

  source.onmessage = (e) => {
    try {
      const data: SSEEvent = JSON.parse(e.data);
      onEvent(data);
      if (data.type === 'done' || data.type === 'error') {
        source.close();
      }
    } catch {
      // ignore parse errors
    }
  };

  source.onerror = () => {
    source.close();
    onEvent({ type: 'error', message: 'Connection lost' });
  };

  return () => source.close();
}

// ── Models ──────────────────────────────────────────────────

export async function listModels(): Promise<ModelInfo[]> {
  const res = await fetch(`${BASE}/models`);
  return handleResponse<ModelInfo[]>(res);
}

export async function deleteModel(name: string): Promise<{ ok: boolean }> {
  const res = await fetch(`${BASE}/models/${encodeURIComponent(name)}`, { method: 'DELETE' });
  return handleResponse<{ ok: boolean }>(res);
}
