
import type { ChangeEvent } from 'react';
import { Input } from '@/components/ui/input';
import Papa from 'papaparse';

interface FileUploadProps {
  onClassifySignals: (signals: number[][]) => void;
}


import { useRef, useState, type DragEvent } from 'react';

export function FileUpload({ onClassifySignals }: FileUploadProps) {
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const processFile = (file: File) => {
    Papa.parse<string[]>(file, {
      complete: (result: Papa.ParseResult<string[]>) => {
        const rows: string[][] = result.data;
        console.log('Parsed CSV rows:', rows);
        if (rows.length === 0) {
          alert('File contains no data!');
          return;
        }
        const signals: number[][] = rows
          .map((row: string[]) => {
            if (row.length < 188) return null;
            const signal: number[] = row.slice(0, -1).map((val: string) => {
              const num: number = Number(val);
              return isNaN(num) ? 0 : num;
            });
            return signal.length === 187 ? signal : null;
          })
          .filter((signal: number[] | null): signal is number[] => signal !== null);
        console.log('Processed signals:', signals);
        if (signals.length === 0) {
          alert('No valid signals in file!');
          return;
        }
        onClassifySignals(signals);
      },
      header: false,
      skipEmptyLines: true,
      error: (error: Error) => {
        console.error('Error parsing CSV:', error);
        alert('Error reading CSV file: ' + error.message);
      },
    });
  };

  const handleFileUpload = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) {
      alert('Please select a file to upload!');
      return;
    }
    processFile(file);
  };

  const handleDragOver = (e: DragEvent<HTMLLabelElement | HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLLabelElement | HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
  };

  const handleDrop = (e: DragEvent<HTMLLabelElement | HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      if (file.name.endsWith('.csv')) {
        processFile(file);
      } else {
        alert('Only .csv files are supported!');
      }
    }
  };

  return (
    <div className="mt-8 flex flex-col items-center">
      <h2 className="text-xl font-bold mb-4 text-blue-700 dark:text-blue-300 tracking-tight">Upload Test File</h2>
      <label
        htmlFor="file-upload"
        className={`flex flex-col items-center justify-center gap-2 px-6 py-8 rounded-lg border-2 border-dashed transition-all cursor-pointer w-full max-w-md bg-blue-50 dark:bg-blue-900 text-blue-700 dark:text-blue-200 font-semibold shadow-sm border-blue-200 dark:border-blue-700 hover:bg-blue-100 dark:hover:bg-blue-800 focus-within:ring-2 focus-within:ring-blue-400 ${dragActive ? 'border-blue-500 bg-blue-100 dark:bg-blue-800' : ''}`}
        tabIndex={0}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M7 10l5 5m0 0l5-5m-5 5V4" />
        </svg>
        <span className="text-base">Drag & drop your <span className="font-medium">.csv</span> file here, or <span className="underline">click to select</span></span>
        <Input
          id="file-upload"
          type="file"
          accept=".csv"
          onChange={handleFileUpload}
          className="hidden"
          ref={inputRef}
        />
      </label>
      <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">Only <span className="font-medium">.csv</span> files are supported.</p>
    </div>
  );
}