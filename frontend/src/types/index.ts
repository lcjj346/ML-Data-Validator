// ── Validation ──────────────────────────────────────────────

export interface FileInfo {
  session_id: string;
  filename: string;
  rows: number;
  columns: number;
  column_names: string[];
  preview: Record<string, unknown>[];
}

export interface MatchColumnsResponse {
  matched: Record<string, string>;
  unmatched: string[];
  trained_columns: string[];
}

export interface CorrectionItem {
  row_index: number;
  row: number;
  column: string;
  original: string;
  suggested: string;
  has_correction: boolean;
  reason: string;
}

export interface ValidationResults {
  data: Record<string, unknown>[];
  cell_validity: Record<string, boolean>;
  modified_cells: string[];
  corrections: CorrectionItem[];
  quality: number;
  valid_cells: number;
  invalid_cells: number;
  total_cells: number;
}

// ── Training ────────────────────────────────────────────────

export interface TrainUploadInfo {
  session_id: string;
  filename: string;
  rows: number;
  columns: number;
  column_names: string[];
  preview: Record<string, unknown>[];
}

export interface ColumnMetrics {
  unique_valid?: number;
  total_samples?: number;
  used_split: boolean;
  test_size?: number;
  train_accuracy?: number;
  train_f1?: number;
  test_accuracy?: number;
  test_f1?: number;
  test_confusion_matrix?: number[][];
  best_C?: number;
  cv_f1_score?: number;
}

export interface TrainingMetrics {
  [columnName: string]: ColumnMetrics;
}

// ── Models ──────────────────────────────────────────────────

export interface ModelInfo {
  name: string;
  size: string;
  columns: string;
}

// ── SSE Events ──────────────────────────────────────────────

export interface SSEProgressEvent {
  type: 'progress';
  progress: number;
  message: string;
}

export interface SSEDoneEvent {
  type: 'done';
  metrics?: TrainingMetrics;
}

export interface SSEErrorEvent {
  type: 'error';
  message: string;
}

export type SSEEvent = SSEProgressEvent | SSEDoneEvent | SSEErrorEvent;
