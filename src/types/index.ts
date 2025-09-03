export interface DataRow {
  [key: string]: string | number;
}

export interface ValidationResult {
  rowIndex: number;
  issues: FieldIssue[];
  isValid: boolean;
}

export interface FieldIssue {
  field: string;
  isValid: boolean;
  confidence: number;
  suggestedValue?: string | number;
  errorType: 'format' | 'range' | 'missing' | 'invalid';
  message: string;
}

export interface MLModel {
  predict: (value: any) => Promise<{
    isValid: boolean;
    confidence: number;
    suggestedValue?: string | number;
  }>;
}