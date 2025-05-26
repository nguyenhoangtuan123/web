import { useLayoutEffect, useRef, useState } from 'react';
import { Chart, registerables } from 'chart.js';

// Register Chart.js components
Chart.register(...registerables);

// Define props interface
interface ShapValuesChartProps {
  signalToExplain: number[] | null;
}

interface ShapResponse {
  shap_values: number[][]; // Shape: (5, 187) - 5 classes, 187 time points
  scaled_instance: number[]; // Shape: (187,) - Scaled ECG signal
  class_idx: number; // Index of the predicted class
  error?: string;
}

const ShapValuesChart: React.FC<ShapValuesChartProps> = ({ signalToExplain }) => {
  const chartRef = useRef<HTMLCanvasElement>(null);
  const chartInstanceRef = useRef<Chart | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [shapData, setShapData] = useState<ShapResponse | null>(null);

  // Fetch SHAP values
  useLayoutEffect(() => {
    if (!signalToExplain || signalToExplain.length === 0) {
      console.log('No signal to explain');
      return;
    }

    setIsLoading(true);
    setError(null);

    fetch('http://192.168.1.1:5000/get_shape_xgboost', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ signal: signalToExplain }),
    })
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        if (data.error) {
          setError(data.error);
          return;
        }
        setShapData(data);
      })
      .catch(error => {
        console.error('Error fetching SHAP values:', error);
        setError(error instanceof Error ? error.message : 'Unknown error');
      })
      .finally(() => {
        setIsLoading(false);
      });

    // Cleanup
    return () => {
      if (chartInstanceRef.current) {
        chartInstanceRef.current.destroy();
        chartInstanceRef.current = null;
      }
    };
  }, [signalToExplain]);

  // Render chart
  useLayoutEffect(() => {
    if (!shapData || !chartRef.current) {
      console.error('No shapData or canvas ref available');
      return;
    }

    const ctx = chartRef.current.getContext('2d');
    if (!ctx) {
      console.error('Failed to get 2D context for canvas');
      setError('Failed to initialize chart context');
      return;
    }

    // Validate data
    const { scaled_instance: [scaledEcgSignal], shap_values: shapValues } = shapData;
    if (
      !Array.isArray(scaledEcgSignal) ||
      !scaledEcgSignal.every(val => typeof val === 'number' && !isNaN(val)) ||
      !shapValues.every(arr => Array.isArray(arr) && arr.every(val => typeof val === 'number' && !isNaN(val))) ||
      !shapValues.every(arr => arr.length === scaledEcgSignal.length)
    ) {
      console.error('Invalid data:', { scaledEcgSignal, shapValues });
      console.error("scaledEcgSignal:", scaledEcgSignal);
      setError('Invalid SHAP data format');
      return;
    }

    // Destroy existing chart
    if (chartInstanceRef.current) {
      chartInstanceRef.current.destroy();
      chartInstanceRef.current = null;
    }

    console.log('SHAP values shape:', shapValues.length, 'x', shapValues[0]?.length);

    // Class names for MIT-BIH dataset
    const classNames = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unknown'];

    // Find class with highest SHAP values
    // const shapSum = shapValues.map(classValues =>
    //   classValues.reduce((sum, val) => sum + Math.abs(val), 0)
    // );
    const predictedClassIdx = shapData.class_idx;

    // Colors for each class
    const colors = [
      'rgba(255, 99, 132, 1)', // Red
      'rgba(75, 192, 192, 1)', // Cyan
      'rgba(255, 159, 64, 1)', // Orange
      'rgba(153, 102, 255, 1)', // Purple
      'rgba(165, 42, 42, 1)', // Brown
    ];

    // Prepare datasets
    const datasets = [
      {
        label: 'Scaled ECG Signal',
        data: scaledEcgSignal,
        borderColor: 'rgba(54, 162, 235, 0.5)', // Blue
        borderWidth: 1,
        fill: false,
        pointRadius: 0,
      },
      ...classNames.map((className, idx) => ({
        label: `SHAP Values (${className})`,
        data: shapValues[idx],
        borderColor: colors[idx],
        borderWidth: idx === predictedClassIdx ? 3 : 1,
        pointRadius: 0,
        fill: false,
        borderDash: idx === predictedClassIdx ? undefined : [5, 5],
      })),
    ];

    console.log('Chart datasets:', datasets);

    // Create chart
    chartInstanceRef.current = new Chart(ctx, {
      type: 'line',
      data: {
        labels: Array.from({ length: scaledEcgSignal.length }, (_, i) => i),
        datasets,
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            title: { display: true, text: 'Time Point' },
            ticks: {
              maxTicksLimit: 10,
              callback: function (value) {
                return value.toString();
              },
            },
          },
          y: {
            title: { display: true, text: 'Value' },
          },
        },
        plugins: {
          title: {
            display: true,
            text: `SHAP Values - Predicted Class: ${classNames[predictedClassIdx]}`,
          },
          tooltip: {
            mode: 'index',
            intersect: false,
          },
          legend: {
            display: true,
            position: 'top',
          },
        },
      },
    });

    console.log('Chart instance created:', chartInstanceRef.current);

    // Cleanup
    return () => {
      if (chartInstanceRef.current) {
        chartInstanceRef.current.destroy();
        chartInstanceRef.current = null;
      }
    };
  }, [shapData]);

  if (isLoading) {
    return <div className="text-yellow-500 mb-4">Loading SHAP values...</div>;
  }

  if (error) {
    return <div className="text-red-500 mb-4">Error loading SHAP values: {error}</div>;
  }

  if (!shapData) {
    return null;
  }

  return (
    <div className="mb-4" style={{ position: 'relative', width: '100%', height: '400px' }}>
      <h2 className="text-lg font-semibold">Model Explanation (SHAP Values)</h2>
      <div className="text-sm text-gray-500 mb-2">
        This chart shows how each time point in the signal contributes to the prediction for each class.
        The highlighted line represents the predicted class.
      </div>
      <canvas id="shapValuesChart" ref={chartRef} style={{ width: '100%', height: '100%' }}></canvas>
    </div>
  );
};

export default ShapValuesChart;