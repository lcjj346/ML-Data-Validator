import React, { useState } from 'react';
import { Brain, Upload, Play, CheckCircle, AlertCircle, Loader2, Download } from 'lucide-react';
import Papa from 'papaparse';
import * as XLSX from 'xlsx';
import { trainModels } from '../ml/trainer';
import type { DataRow } from '../types';

interface ModelTrainerProps {
  onTrainingComplete: () => void;
}

const ModelTrainer: React.FC<ModelTrainerProps> = ({ onTrainingComplete }) => {
  const [trainingData, setTrainingData] = useState<DataRow[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState<string>('');
  const [trainingResults, setTrainingResults] = useState<any>(null);

  const handleTrainingFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
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

        setTrainingData(parsedData);
      } catch (error) {
        console.error('Error parsing training file:', error);
        alert('Error parsing file. Please check the format and try again.');
      }
    };

    if (file.name.endsWith('.csv')) {
      reader.readAsText(file);
    } else {
      reader.readAsBinaryString(file);
    }
  };

  const startTraining = async () => {
    if (trainingData.length === 0) {
      alert('Please upload training data first');
      return;
    }

    setIsTraining(true);
    setTrainingProgress('Initializing training...');

    try {
      const results = await trainModels(trainingData, (progress) => {
        setTrainingProgress(progress);
      });
      
      setTrainingResults(results);
      setTrainingProgress('Training completed successfully!');
      
      setTimeout(() => {
        onTrainingComplete();
      }, 2000);
    } catch (error) {
      console.error('Training error:', error);
      setTrainingProgress('Training failed. Please check your data format.');
    } finally {
      setIsTraining(false);
    }
  };

  const downloadSampleData = () => {
    const sampleData = [
      { PhoneNumber: '+1234567890', BloodSugar: '95', PhoneNumber_Valid: '1', BloodSugar_Valid: '1' },
      { PhoneNumber: 'abc123', BloodSugar: '999', PhoneNumber_Valid: '0', BloodSugar_Valid: '0' },
      { PhoneNumber: '+65 91234567', BloodSugar: '120', PhoneNumber_Valid: '1', BloodSugar_Valid: '1' },
      { PhoneNumber: '123', BloodSugar: '-10', PhoneNumber_Valid: '0', BloodSugar_Valid: '0' },
    ];

    const csvContent = [
      Object.keys(sampleData[0]).join(','),
      ...sampleData.map(row => Object.values(row).join(','))
    ].join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'sample_training_data.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
      <div className="flex items-center gap-3 mb-6">
        <Brain className="w-6 h-6 text-primary-600" />
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white">ML Model Training</h2>
      </div>

      <div className="space-y-6">
        <div>
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-3">Training Data Requirements</h3>
          <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4 mb-4">
            <p className="text-sm text-blue-800 dark:text-blue-200 mb-2">
              Your training data should include columns with "_Valid" suffix indicating whether each value is correct:
            </p>
            <ul className="text-sm text-blue-700 dark:text-blue-300 space-y-1">
              <li>• <code className="bg-blue-100 dark:bg-blue-800 px-1 rounded">PhoneNumber</code> + <code className="bg-blue-100 dark:bg-blue-800 px-1 rounded">PhoneNumber_Valid</code> (1 for valid, 0 for invalid)</li>
              <li>• <code className="bg-blue-100 dark:bg-blue-800 px-1 rounded">BloodSugar</code> + <code className="bg-blue-100 dark:bg-blue-800 px-1 rounded">BloodSugar_Valid</code></li>
              <li>• <code className="bg-blue-100 dark:bg-blue-800 px-1 rounded">Email</code> + <code className="bg-blue-100 dark:bg-blue-800 px-1 rounded">Email_Valid</code></li>
            </ul>
          </div>
          
          <div className="flex gap-3">
            <label className="btn-secondary cursor-pointer inline-flex items-center gap-2">
              <Upload className="w-4 h-4" />
              Upload Training Data
              <input
                type="file"
                accept=".csv,.xlsx,.xls"
                onChange={handleTrainingFileUpload}
                className="hidden"
              />
            </label>
            <button
              onClick={downloadSampleData}
              className="btn-secondary inline-flex items-center gap-2"
            >
              <Download className="w-4 h-4" />
              Download Sample Format
            </button>
          </div>
        </div>

        {trainingData.length > 0 && (
          <div>
            <div className="flex items-center justify-between mb-4">
              <div>
                <h4 className="font-medium text-gray-900 dark:text-white">Training Data Loaded</h4>
                <p className="text-sm text-gray-600 dark:text-gray-300">{trainingData.length} rows ready for training</p>
              </div>
              <button
                onClick={startTraining}
                disabled={isTraining}
                className="btn-primary inline-flex items-center gap-2"
              >
                {isTraining ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Play className="w-4 h-4" />
                )}
                {isTraining ? 'Training...' : 'Start Training'}
              </button>
            </div>

            {isTraining && (
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <div className="flex items-center gap-3">
                  <Loader2 className="w-5 h-5 text-primary-600 animate-spin" />
                  <span className="text-sm text-gray-700 dark:text-gray-300">{trainingProgress}</span>
                </div>
              </div>
            )}

            {trainingResults && (
              <div className="bg-success-50 dark:bg-success-900/20 border border-success-200 dark:border-success-700 rounded-lg p-4">
                <div className="flex items-center gap-3 mb-3">
                  <CheckCircle className="w-5 h-5 text-success-600" />
                  <h4 className="font-medium text-success-800 dark:text-success-400">Training Complete!</h4>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {Object.entries(trainingResults.modelAccuracies).map(([model, accuracy]) => (
                    <div key={model} className="text-center">
                      <p className="text-lg font-bold text-success-700 dark:text-success-400">
                        {(accuracy as number * 100).toFixed(1)}%
                      </p>
                      <p className="text-xs text-success-600 dark:text-success-500">{model}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default ModelTrainer;