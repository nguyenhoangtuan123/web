import { useEffect, useRef } from 'react';
import Chart from 'chart.js/auto';

// Define props interface
interface SignalChartProps {
  signalData: number[];
}

const SignalChart: React.FC<SignalChartProps> = ({ signalData }) => {
  const chartRef = useRef<HTMLCanvasElement>(null);
  const chartInstanceRef = useRef<Chart | null>(null); // Store chart instance locally

  useEffect(() => {
    // console.log('SignalChart useEffect triggered', signalData );
    if (signalData === undefined || signalData.length === 0) {
      return
    }

    const ctx = chartRef.current?.getContext('2d');

    if (!ctx) return;

    // Destroy existing chart if it exists
    if (chartInstanceRef.current) {
      chartInstanceRef.current.destroy();
    }

    // Create new chart
    chartInstanceRef.current = new Chart(ctx, {
      type: 'line',
      data: {
        labels: Array.from({ length: signalData.length }, (_, i) => i),
        datasets: [
          {
            label: 'Heartbeat Signal',
            data: signalData,
            borderColor: 'white',
            borderWidth: 1,
            fill: false,
          },
        ],
      },
      options: {
        responsive: true,
        scales: {
          x: { title: { display: true, text: 'Time' } },
          y: { title: { display: true, text: 'Amplitude' } },
        },
      },
    });

    // Cleanup on unmount or when dependencies change
    return () => {
      if (chartInstanceRef.current) {
        chartInstanceRef.current.destroy();
        chartInstanceRef.current = null;
      }
    };
  }, [signalData]); // Only depend on selectedSignal and samples

  return (
    (signalData !== null && signalData !== undefined) && (
      <div className="mb-4">
        <h2 className="text-lg font-semibold">Signal Visualization</h2>
        <canvas id="heartbeatChart" ref={chartRef}></canvas>
      </div>
    )
  );
};

export default SignalChart;