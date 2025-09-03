import React, { useState } from 'react';
import { Upload, Download, AlertCircle, CheckCircle, FileText } from 'lucide-react';
import FileUploader from './components/FileUploader';
import DataTable from './components/DataTable';
import ValidationSummary from './components/ValidationSummary';
import { validateData } from './ml/validator';
import { correctData } from './ml/corrector';
import type { DataRow, ValidationResult } from './types';

function App() {
  const [data, setData] = useState<DataRow[]>([]);
  const [validationResults, setValidationResults] = useState<ValidationResult[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [fileName, setFileName] = useState<string>('');

  const handleFileUpload = async (uploadedData: DataRow[], filename: string) => {
    setIsProcessing(true);
    setFileName(filename);
    
    try {
      // Validate the data using ML models
      const results = await validateData(uploadedData);
      setData(uploadedData);
      setValidationResults(results);
    } catch (error) {
      console.error('Validation error:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDataCorrection = async () => {
    setIsProcessing(true);
    
    try {
      const correctedData = await correctData(data, validationResults);
      const newResults = await validateData(correctedData);
      setData(correctedData);
      setValidationResults(newResults);
    } catch (error) {
      console.error('Correction error:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleCellEdit = async (rowIndex: number, field: string, value: string) => {
    const updatedData = [...data];
    updatedData[rowIndex] = { ...updatedData[rowIndex], [field]: value };
    
    // Re-validate the specific row
    const rowResults = await validateData([updatedData[rowIndex]]);
    const updatedResults = [...validationResults];
    updatedResults[rowIndex] = rowResults[0];
    
    setData(updatedData);
    setValidationResults(updatedResults);
  };

  const exportData = () => {
    const csvContent = [
      Object.keys(data[0] || {}).join(','),
      ...data.map(row => Object.values(row).join(','))
    ].join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `corrected_${fileName}`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 py-8">
        <header className="mb-8">
          <div className="flex items-center gap-3 mb-4">
            <FileText className="w-8 h-8 text-primary-600" />
            <h1 className="text-3xl font-bold text-gray-900">ML Data Validator</h1>
          </div>
          <p className="text-gray-600 max-w-2xl">
            Upload your survey data and let our ML models automatically detect and suggest corrections 
            for invalid entries. Clean your data before feeding it into analytics pipelines.
          </p>
        </header>

        {data.length === 0 ? (
          <FileUploader onFileUpload={handleFileUpload} isProcessing={isProcessing} />
        ) : (
          <div className="space-y-6">
            <ValidationSummary 
              results={validationResults} 
              onCorrect={handleDataCorrection}
              onExport={exportData}
              isProcessing={isProcessing}
            />
            
            <DataTable 
              data={data}
              validationResults={validationResults}
              onCellEdit={handleCellEdit}
            />
          </div>
        )}
      </div>
    </div>
  );
}

export default App;