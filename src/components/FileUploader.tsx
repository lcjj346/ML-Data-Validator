import React, { useCallback } from 'react';
import { Upload, FileText, Loader2 } from 'lucide-react';
import Papa from 'papaparse';
import * as XLSX from 'xlsx';
import type { DataRow } from '../types';

interface FileUploaderProps {
  onFileUpload: (data: DataRow[], filename: string) => void;
  isProcessing: boolean;
}

const FileUploader: React.FC<FileUploaderProps> = ({ onFileUpload, isProcessing }) => {
  const handleFileUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      const data = e.target?.result;
      if (!data) return;

      try {
        let parsedData: DataRow[] = [];

        if (file.name.endsWith('.csv')) {
          const result = Papa.parse(data as string, {
            header: true,
            skipEmptyLines: true,
            transformHeader: (header) => header.trim(),
          });
          parsedData = result.data as DataRow[];
        } else if (file.name.endsWith('.xlsx') || file.name.endsWith('.xls')) {
          const workbook = XLSX.read(data, { type: 'binary' });
          const sheetName = workbook.SheetNames[0];
          const worksheet = workbook.Sheets[sheetName];
          parsedData = XLSX.utils.sheet_to_json(worksheet) as DataRow[];
        }

        onFileUpload(parsedData, file.name);
      } catch (error) {
        console.error('Error parsing file:', error);
        alert('Error parsing file. Please check the format and try again.');
      }
    };

    if (file.name.endsWith('.csv')) {
      reader.readAsText(file);
    } else {
      reader.readAsBinaryString(file);
    }
  }, [onFileUpload]);

  return (
    <div className="max-w-2xl mx-auto">
      <div className="border-2 border-dashed border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 rounded-lg p-12 text-center hover:border-primary-400 transition-colors duration-200">
        <div className="flex flex-col items-center gap-4">
          {isProcessing ? (
            <>
              <Loader2 className="w-12 h-12 text-primary-600 animate-spin" />
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Processing your data...</h3>
              <p className="text-gray-600 dark:text-gray-300">Running ML validation models</p>
            </>
          ) : (
            <>
              <Upload className="w-12 h-12 text-gray-400" />
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Upload Survey Data</h3>
              <p className="text-gray-600 dark:text-gray-300 mb-4">
                Drop your CSV or Excel file here, or click to browse
              </p>
              <label className="btn-primary cursor-pointer inline-flex items-center gap-2">
                <FileText className="w-4 h-4" />
                Choose File
                <input
                  type="file"
                  accept=".csv,.xlsx,.xls"
                  onChange={handleFileUpload}
                  className="hidden"
                />
              </label>
              <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">
                Supports CSV, XLSX, and XLS files
              </p>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default FileUploader;