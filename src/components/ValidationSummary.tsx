import React from 'react';
import { AlertTriangle, CheckCircle, Download, Wand2, Loader2 } from 'lucide-react';
import type { ValidationResult } from '../types';

interface ValidationSummaryProps {
  results: ValidationResult[];
  onCorrect: () => void;
  onExport: () => void;
  isProcessing: boolean;
}

const ValidationSummary: React.FC<ValidationSummaryProps> = ({ 
  results, 
  onCorrect, 
  onExport, 
  isProcessing 
}) => {
  const totalRows = results.length;
  const validRows = results.filter(r => r.isValid).length;
  const invalidRows = totalRows - validRows;
  
  const allIssues = results.flatMap(r => r.issues);
  const issuesByField = allIssues.reduce((acc, issue) => {
    if (!issue.isValid) {
      acc[issue.field] = (acc[issue.field] || 0) + 1;
    }
    return acc;
  }, {} as Record<string, number>);

  const hasCorrections = allIssues.some(issue => !issue.isValid && issue.suggestedValue);

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold text-gray-900">Validation Summary</h2>
        <div className="flex gap-3">
          {hasCorrections && (
            <button 
              onClick={onCorrect}
              disabled={isProcessing}
              className="btn-primary inline-flex items-center gap-2"
            >
              {isProcessing ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Wand2 className="w-4 h-4" />
              )}
              Apply ML Corrections
            </button>
          )}
          <button 
            onClick={onExport}
            className="btn-secondary inline-flex items-center gap-2"
          >
            <Download className="w-4 h-4" />
            Export Data
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
        <div className="bg-gray-50 rounded-lg p-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-primary-100 rounded-lg flex items-center justify-center">
              <CheckCircle className="w-5 h-5 text-primary-600" />
            </div>
            <div>
              <p className="text-2xl font-bold text-gray-900">{totalRows}</p>
              <p className="text-sm text-gray-600">Total Rows</p>
            </div>
          </div>
        </div>

        <div className="bg-success-50 rounded-lg p-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-success-100 rounded-lg flex items-center justify-center">
              <CheckCircle className="w-5 h-5 text-success-600" />
            </div>
            <div>
              <p className="text-2xl font-bold text-success-700">{validRows}</p>
              <p className="text-sm text-success-600">Valid Rows</p>
            </div>
          </div>
        </div>

        <div className="bg-error-50 rounded-lg p-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-error-100 rounded-lg flex items-center justify-center">
              <AlertTriangle className="w-5 h-5 text-error-600" />
            </div>
            <div>
              <p className="text-2xl font-bold text-error-700">{invalidRows}</p>
              <p className="text-sm text-error-600">Invalid Rows</p>
            </div>
          </div>
        </div>
      </div>

      {Object.keys(issuesByField).length > 0 && (
        <div>
          <h3 className="text-lg font-medium text-gray-900 mb-3">Issues by Field</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {Object.entries(issuesByField).map(([field, count]) => (
              <div key={field} className="bg-error-50 border border-error-200 rounded-lg p-3">
                <div className="flex items-center justify-between">
                  <span className="font-medium text-error-700">{field}</span>
                  <span className="text-sm text-error-600">{count} issues</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default ValidationSummary;