import type { DataRow, ValidationResult } from '../types';

export async function correctData(data: DataRow[], validationResults: ValidationResult[]): Promise<DataRow[]> {
  const correctedData = [...data];

  for (let rowIndex = 0; rowIndex < correctedData.length; rowIndex++) {
    const result = validationResults[rowIndex];
    if (!result) continue;

    for (const issue of result.issues) {
      if (!issue.isValid && issue.suggestedValue !== undefined) {
        correctedData[rowIndex] = {
          ...correctedData[rowIndex],
          [issue.field]: issue.suggestedValue,
        };
      }
    }
  }

  return correctedData;
}