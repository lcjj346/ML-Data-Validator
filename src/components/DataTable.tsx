import React, { useState } from 'react';
import { AlertCircle, CheckCircle, Edit3, Save, X } from 'lucide-react';
import type { DataRow, ValidationResult } from '../types';

interface DataTableProps {
  data: DataRow[];
  validationResults: ValidationResult[];
  onCellEdit: (rowIndex: number, field: string, value: string) => void;
}

const DataTable: React.FC<DataTableProps> = ({ data, validationResults, onCellEdit }) => {
  const [editingCell, setEditingCell] = useState<{ row: number; field: string } | null>(null);
  const [editValue, setEditValue] = useState('');

  if (data.length === 0) return null;

  const columns = Object.keys(data[0]);

  const startEdit = (rowIndex: number, field: string, currentValue: any) => {
    setEditingCell({ row: rowIndex, field });
    setEditValue(String(currentValue || ''));
  };

  const saveEdit = () => {
    if (editingCell) {
      onCellEdit(editingCell.row, editingCell.field, editValue);
      setEditingCell(null);
    }
  };

  const cancelEdit = () => {
    setEditingCell(null);
    setEditValue('');
  };

  const getCellIssue = (rowIndex: number, field: string) => {
    const result = validationResults[rowIndex];
    return result?.issues.find(issue => issue.field === field);
  };

  const getCellStyle = (rowIndex: number, field: string) => {
    const issue = getCellIssue(rowIndex, field);
    if (!issue) return 'bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-600';
    
    if (!issue.isValid) {
      return 'bg-error-50 dark:bg-error-900/20 border-error-200 dark:border-error-700';
    }
    return 'bg-success-50 dark:bg-success-900/20 border-success-200 dark:border-success-700';
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
      <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Data Validation Results</h3>
        <p className="text-sm text-gray-600 dark:text-gray-300 mt-1">
          Click on any cell to edit. Invalid cells are highlighted in red.
        </p>
      </div>
      
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-gray-50 dark:bg-gray-700">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                Row
              </th>
              {columns.map(column => (
                <th key={column} className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  {column}
                </th>
              ))}
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                Status
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
            {data.map((row, rowIndex) => (
              <tr key={rowIndex} className="hover:bg-gray-50 dark:hover:bg-gray-700">
                <td className="px-4 py-3 text-sm text-gray-600 dark:text-gray-300 font-medium">
                  {rowIndex + 1}
                </td>
                {columns.map(column => {
                  const isEditing = editingCell?.row === rowIndex && editingCell?.field === column;
                  const issue = getCellIssue(rowIndex, column);
                  
                  return (
                    <td key={column} className="px-4 py-3">
                      {isEditing ? (
                        <div className="flex items-center gap-2">
                          <input
                            type="text"
                            value={editValue}
                            onChange={(e) => setEditValue(e.target.value)}
                            className="input-field text-sm w-full"
                            autoFocus
                            onKeyDown={(e) => {
                              if (e.key === 'Enter') saveEdit();
                              if (e.key === 'Escape') cancelEdit();
                            }}
                          />
                          <button onClick={saveEdit} className="text-success-600 hover:text-success-700">
                            <Save className="w-4 h-4" />
                          </button>
                          <button onClick={cancelEdit} className="text-gray-400 dark:text-gray-500 hover:text-gray-600 dark:hover:text-gray-300">
                            <X className="w-4 h-4" />
                          </button>
                        </div>
                      ) : (
                        <div 
                          className={`relative group cursor-pointer p-2 rounded border transition-colors ${getCellStyle(rowIndex, column)}`}
                          onClick={() => startEdit(rowIndex, column, row[column])}
                        >
                          <div className="flex items-center justify-between">
                            <span className="text-sm text-gray-900 dark:text-gray-100">{String(row[column] || '')}</span>
                            <Edit3 className="w-3 h-3 text-gray-400 dark:text-gray-500 opacity-0 group-hover:opacity-100 transition-opacity" />
                          </div>
                          {issue && !issue.isValid && issue.suggestedValue && (
                            <div className="absolute top-full left-0 mt-1 bg-gray-900 dark:bg-gray-700 text-white text-xs px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity z-10 whitespace-nowrap">
                              Suggested: {String(issue.suggestedValue)}
                            </div>
                          )}
                        </div>
                      )}
                    </td>
                  );
                })}
                <td className="px-4 py-3">
                  {validationResults[rowIndex]?.isValid ? (
                    <CheckCircle className="w-5 h-5 text-success-500" />
                  ) : (
                    <AlertCircle className="w-5 h-5 text-error-500" />
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default DataTable;