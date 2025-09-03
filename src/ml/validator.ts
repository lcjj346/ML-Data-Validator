import type { DataRow, ValidationResult, FieldIssue } from '../types';
import { PhoneValidator } from './models/PhoneValidator';
import { BloodSugarValidator } from './models/BloodSugarValidator';
import { EmailValidator } from './models/EmailValidator';
import { AgeValidator } from './models/AgeValidator';

const validators = {
  phone: new PhoneValidator(),
  bloodSugar: new BloodSugarValidator(),
  email: new EmailValidator(),
  age: new AgeValidator(),
};

export async function validateData(data: DataRow[]): Promise<ValidationResult[]> {
  const results: ValidationResult[] = [];

  for (let rowIndex = 0; rowIndex < data.length; rowIndex++) {
    const row = data[rowIndex];
    const issues: FieldIssue[] = [];

    for (const [field, value] of Object.entries(row)) {
      const issue = await validateField(field, value);
      if (issue) {
        issues.push(issue);
      }
    }

    results.push({
      rowIndex,
      issues,
      isValid: issues.every(issue => issue.isValid),
    });
  }

  return results;
}

async function validateField(field: string, value: any): Promise<FieldIssue | null> {
  const fieldLower = field.toLowerCase();
  
  // Determine which validator to use based on field name
  let validator;
  if (fieldLower.includes('phone')) {
    validator = validators.phone;
  } else if (fieldLower.includes('blood') || fieldLower.includes('sugar') || fieldLower.includes('glucose')) {
    validator = validators.bloodSugar;
  } else if (fieldLower.includes('email')) {
    validator = validators.email;
  } else if (fieldLower.includes('age')) {
    validator = validators.age;
  } else {
    // Generic validation for unknown fields
    return validateGenericField(field, value);
  }

  const result = await validator.predict(value);
  
  return {
    field,
    isValid: result.isValid,
    confidence: result.confidence,
    suggestedValue: result.suggestedValue,
    errorType: result.isValid ? 'format' : getErrorType(field, value),
    message: result.isValid ? 'Valid' : getErrorMessage(field, value, result.suggestedValue),
  };
}

function validateGenericField(field: string, value: any): FieldIssue {
  const isEmpty = value === null || value === undefined || String(value).trim() === '';
  
  return {
    field,
    isValid: !isEmpty,
    confidence: isEmpty ? 0.9 : 1.0,
    errorType: isEmpty ? 'missing' : 'format',
    message: isEmpty ? 'Field is empty' : 'Valid',
  };
}

function getErrorType(field: string, value: any): 'format' | 'range' | 'missing' | 'invalid' {
  if (value === null || value === undefined || String(value).trim() === '') {
    return 'missing';
  }
  
  const fieldLower = field.toLowerCase();
  if (fieldLower.includes('phone')) {
    return 'format';
  } else if (fieldLower.includes('blood') || fieldLower.includes('age')) {
    return 'range';
  } else if (fieldLower.includes('email')) {
    return 'format';
  }
  
  return 'invalid';
}

function getErrorMessage(field: string, value: any, suggestedValue?: any): string {
  const fieldLower = field.toLowerCase();
  
  if (fieldLower.includes('phone')) {
    return 'Invalid phone number format';
  } else if (fieldLower.includes('blood')) {
    return 'Blood sugar value out of normal range (50-500 mg/dL)';
  } else if (fieldLower.includes('email')) {
    return 'Invalid email format';
  } else if (fieldLower.includes('age')) {
    return 'Invalid age value';
  }
  
  return 'Invalid value detected';
}