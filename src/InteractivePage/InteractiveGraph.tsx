// import { Switch } from "@/components/ui/switch";
import React, { useEffect, useRef, useState, forwardRef } from "react";

export interface InteractiveGraphRef {
  setPointsFromInput: (inputPoints: Point[]) => void;
}

interface IProps {
  points_input: Point[];
  setPoints_input: React.Dispatch<React.SetStateAction<Point[]>>;
  regPoints_output: Point[];
  setRegPoints_output: React.Dispatch<React.SetStateAction<Point[]>>;
  ref: React.RefObject<HTMLCanvasElement>;
}

const InteractiveGraph = forwardRef<InteractiveGraphRef, IProps>(
    ({points_input, setPoints_input, regPoints_output, setRegPoints_output}, ref) => {
  const WIDTH = 2000;
  const HEIGHT = 200;
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [points, setPoints] = useState<Point[]>(points_input.map((point) => ({ x: (point.x-0.5) * WIDTH, y: (point.y-0.5) * HEIGHT }))); // Convert to canvas coordinates
  const [regPoints, setRegPoints] = useState<Point[]>(regPoints_output); // Regression points
  const [draggingPoint, setDraggingPoint] = useState<number | null>(null);
  const [hoveredPoint, setHoveredPoint] = useState<number | null>(null);
  const [offset, setOffset] = useState<Point>({ x: 0, y: 0 });
  const [isViewData, setIsViewData] = useState(false);

  // Function to evaluate the monotonic cubic spline at a given x_spline
  const evaluateMonotonicSpline = (x_spline: number, sortedPoints: Point[]) => {
    const n = sortedPoints.length;
    if (n < 2) return 0; // No curve possible

    // Convert points to spline coordinates
    const x = sortedPoints.map(p => p.x);
    const y = sortedPoints.map(p => p.y);

    // Find the segment containing x_spline
    let i = 0;
    if (x_spline <= x[0]) i = 0;
    else if (x_spline >= x[n - 1]) i = n - 2;
    else {
      for (i = 0; i < n - 1; i++) {
        if (x_spline >= x[i] && x_spline <= x[i + 1]) break;
      }
    }

    // Handle 2 points (linear)
    if (n === 2) {
      const t = x[1] !== x[0] ? (x_spline - x[0]) / (x[1] - x[0]) : 0;
      return y[0] + t * (y[1] - y[0]);
    }

    // Handle 3 points (quadratic Bézier)
    if (n === 3) {
      const t = x[2] !== x[0] ? (x_spline - x[0]) / (x[2] - x[0]) : 0;
      const t0 = (1 - t) * (1 - t);
      const t1 = 2 * t * (1 - t);
      const t2 = t * t;
      return t0 * y[0] + t1 * y[1] + t2 * y[2];
    }

    // Catmull-Rom to Cubic Bézier for 4+ points
    // Monotonic cubic spline (4+ points)
    const dx = new Array(n - 1);
    const dy = new Array(n - 1);
    const m = new Array(n); // Slopes

    // Compute deltas
    for (let j = 0; j < n - 1; j++) {
      dx[j] = x[j + 1] - x[j];
      dy[j] = y[j + 1] - y[j];
    }

    // Initialize slopes
    m[0] = dx[0] !== 0 ? dy[0] / dx[0] : 0;
    for (let j = 1; j < n - 1; j++) {
      m[j] = dx[j - 1] !== 0 && dx[j] !== 0 ? (dy[j - 1] / dx[j - 1] + dy[j] / dx[j]) / 2 : 0;
    }
    m[n - 1] = dx[n - 2] !== 0 ? dy[n - 2] / dx[n - 2] : 0;

    // Enforce monotonicity (Fritsch-Carlson)
    for (let j = 0; j < n - 1; j++) {
      if (dx[j] === 0) {
        m[j] = m[j + 1] = 0;
        continue;
      }
      const alpha = dx[j] !== 0 ? m[j] / (dy[j] / dx[j]) : 0;
      const beta = dx[j] !== 0 ? m[j + 1] / (dy[j] / dx[j]) : 0;
      if (alpha < 0 || beta < 0) {
        m[j] = m[j + 1] = 0;
      } else if (alpha > 3 || beta > 3) {
        const tau = 3 / Math.sqrt(alpha * alpha + beta * beta);
        m[j] *= tau;
        m[j + 1] *= tau;
      }
    }

    // Compute Bézier control points for segment i
    const h = dx[i];
    if (h === 0) return y[i]; // Handle overlapping points
    const p0 = sortedPoints[i];
    const p1 = sortedPoints[i + 1];
    // const cp1x = p0.x + h / 3;
    const cp1y = p0.y + (m[i] * h) / 3;
    // const cp2x = p1.x - h / 3;
    const cp2y = p1.y - (m[i + 1] * h) / 3;

    // Evaluate cubic Bézier curve at t
    const t = x[1] !== x[0] ? (x_spline - x[i]) / h : 0;
    const t_clamped = Math.max(0, Math.min(1, t)); // Clamp t for extrapolation
    const mt = 1 - t_clamped;
    const mt2 = mt * mt;
    const mt3 = mt2 * mt;
    const t2 = t_clamped * t_clamped;
    const t3 = t2 * t_clamped;
    return (
      mt3 * y[i] +
      3 * mt2 * t_clamped * cp1y +
      3 * mt * t2 * cp2y +
      t3 * y[i + 1]
    );
  };

  // Function to compute regression points
  const computeRegressionPoints = () => {
    const step = WIDTH / 187; // ~3.2086
    const regressionPoints: Point[] = [];
    const sortedPoints = [...points].sort((a, b) => a.x - b.x); // Sort by x for spline

    for (let x_canvas = 0; x_canvas <= WIDTH; x_canvas += step) {
      const x_spline = x_canvas - WIDTH/2; // Convert to spline coordinates
      const y_spline = evaluateMonotonicSpline(x_spline, sortedPoints);
      const y_canvas = HEIGHT/2 - y_spline; // Convert back to canvas coordinates
      if (regressionPoints.length < 187)
        regressionPoints.push({ x: x_canvas, y: y_canvas });
    }
    setRegPoints(regressionPoints);
    setRegPoints_output(regressionPoints.map((point) => ({ x: point.x/WIDTH, y: 1-point.y/HEIGHT }))); // Update regPoints_output
    return regressionPoints;
  };

  const drawGraph = (ctx: CanvasRenderingContext2D) => {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    // Draw grid
    ctx.strokeStyle = "#e0e0e0";
    ctx.lineWidth = 1;
    for (let x = 0; x <= ctx.canvas.width; x += 50) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, ctx.canvas.height);
      ctx.stroke();
    }
    for (let y = 0; y <= ctx.canvas.height; y += 50) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(ctx.canvas.width, y);
      ctx.stroke();
    }

    // Draw axes
    ctx.strokeStyle = "#000";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, ctx.canvas.height / 2);
    ctx.lineTo(ctx.canvas.width, ctx.canvas.height / 2);
    ctx.moveTo(ctx.canvas.width / 2, 0);
    ctx.lineTo(ctx.canvas.width / 2, ctx.canvas.height);
    ctx.stroke();

    // Draw lines
    ctx.strokeStyle = "#3b83f66b";
    ctx.lineWidth = 2;
    ctx.beginPath();
    const sortedPoints = [...points].sort((a, b) => a.x - b.x);
    sortedPoints.forEach((point, index) => {
      const canvasX = point.x + ctx.canvas.width / 2;
      const canvasY = ctx.canvas.height / 2 - point.y;
      if (index === 0) ctx.moveTo(canvasX, canvasY);
      else ctx.lineTo(canvasX, canvasY);
    });
    ctx.stroke();

    // Draw points with hover and drag effects
    points.forEach((point, index) => {
      const canvasX = point.x + ctx.canvas.width / 2;
      const canvasY = ctx.canvas.height / 2 - point.y;
      ctx.fillStyle = index === hoveredPoint || index === draggingPoint ? "#ff0000" : "#ef4444";
      ctx.beginPath();
      const radius = index === hoveredPoint || index === draggingPoint ? 10 : 6; // Enlarge on hover/drag
      ctx.arc(canvasX, canvasY, radius, 0, 2 * Math.PI);
      ctx.fill();
    });

    // Draw regression points
    if (isViewData){

      regPoints.forEach((point) => {
        const canvasX = point.x;
        const canvasY = point.y;
        // ctx.fillStyle = "#4e2100"; // Orange for regression points
        ctx.fillStyle = "rgba(78, 33, 0, 0.5)";
        ctx.beginPath();
        ctx.arc(canvasX, canvasY, 4, 0, 1.5 * Math.PI);
        ctx.fill();
      });
    }

    // Draw Bézier curve
    ctx.strokeStyle = "#10b981"; // Green for Bézier curve
    ctx.lineWidth = 2;
    ctx.setLineDash([10, 5]);
    ctx.beginPath();
    if (sortedPoints.length === 2) {
      // Straight line for 2 points
      const [p0, p1] = sortedPoints;
      const canvasX0 = p0.x + ctx.canvas.width / 2;
      const canvasY0 = ctx.canvas.height / 2 - p0.y;
      const canvasX1 = p1.x + ctx.canvas.width / 2;
      const canvasY1 = ctx.canvas.height / 2 - p1.y;
      ctx.moveTo(canvasX0, canvasY0);
      ctx.lineTo(canvasX1, canvasY1);
    } else if (sortedPoints.length === 3) {
      // Quadratic Bézier for 3 points
      const [p0, p1, p2] = sortedPoints;
      const canvasX0 = p0.x + ctx.canvas.width / 2;
      const canvasY0 = ctx.canvas.height / 2 - p0.y;
      const canvasX1 = p1.x + ctx.canvas.width / 2;
      const canvasY1 = ctx.canvas.height / 2 - p1.y;
      const canvasX2 = p2.x + ctx.canvas.width / 2;
      const canvasY2 = ctx.canvas.height / 2 - p2.y;
      ctx.moveTo(canvasX0, canvasY0);
      ctx.quadraticCurveTo(canvasX1, canvasY1, canvasX2, canvasY2);
    } else if (sortedPoints.length >= 4) {
      // Draw monotonic cubic spline using evaluated points
    ctx.strokeStyle = "#10b981"; // Green for spline
    ctx.lineWidth = 2;
    ctx.beginPath();
    const sortedPoints = [...points].sort((a, b) => a.x - b.x);
    if (sortedPoints.length === 2) {
      // Straight line for 2 points
      const [p0, p1] = sortedPoints;
      const canvasX0 = p0.x + ctx.canvas.width / 2;
      const canvasY0 = ctx.canvas.height / 2 - p0.y;
      const canvasX1 = p1.x + ctx.canvas.width / 2;
      const canvasY1 = ctx.canvas.height / 2 - p1.y;
      ctx.moveTo(canvasX0, canvasY0);
      ctx.lineTo(canvasX1, canvasY1);
    } else if (sortedPoints.length === 3) {
      // Quadratic Bézier for 3 points
      const [p0, p1, p2] = sortedPoints;
      const canvasX0 = p0.x + ctx.canvas.width / 2;
      const canvasY0 = ctx.canvas.height / 2 - p0.y;
      const canvasX1 = p1.x + ctx.canvas.width / 2;
      const canvasY1 = ctx.canvas.height / 2 - p1.y;
      const canvasX2 = p2.x + ctx.canvas.width / 2;
      const canvasY2 = ctx.canvas.height / 2 - p2.y;
      ctx.moveTo(canvasX0, canvasY0);
      ctx.quadraticCurveTo(canvasX1, canvasY1, canvasX2, canvasY2);
    } else if (sortedPoints.length >= 4) {
      // Monotonic cubic spline: evaluate at dense points
      const step = 1; // Sample every 1 pixel for smoothness
      for (let x_canvas = 0; x_canvas <= WIDTH; x_canvas += step) {
        const x_spline = x_canvas - WIDTH/2; // Convert to spline coordinates
        const y_spline = evaluateMonotonicSpline(x_spline, sortedPoints);
        const y_canvas = HEIGHT/2 - y_spline; // Convert to canvas coordinates
        if (x_canvas === 0) ctx.moveTo(x_canvas, y_canvas);
        else ctx.lineTo(x_canvas, y_canvas);
      }
    }
    }
    ctx.stroke();
    ctx.setLineDash([]);
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      if (ctx) drawGraph(ctx);
    }
  }, [points, hoveredPoint, draggingPoint]);

  const getMousePos = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    const canvasX = e.clientX - rect.left;
    const canvasY = e.clientY - rect.top;
    return {
      x: canvasX - canvas.width / 2,
      y: -(canvasY - canvas.height / 2),
    };
  };

  const findPointIndex = (mousePos: Point) => {
    const threshold = 15;
    return points.findIndex((point) => {
      const dx = point.x - mousePos.x;
      const dy = point.y - mousePos.y;
      return Math.sqrt(dx * dx + dy * dy) < threshold;
    });
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const mousePos = getMousePos(e);
    const pointIndex = findPointIndex(mousePos);

    if (e.button === 2) {
      // Right-click: remove point if one is near
      if (pointIndex >= 0) {
        e.preventDefault(); // Prevent context menu
        setPoints(points.filter((_, index) => index !== pointIndex));
      }
      return
    }

    if (pointIndex >= 0) {
      setDraggingPoint(pointIndex);
      setOffset({
        x: mousePos.x - points[pointIndex].x,
        y: mousePos.y - points[pointIndex].y,
      });
    } else {
      // Add a new point at the clicked position
      setPoints([...points, { x: mousePos.x, y: mousePos.y }]);
      setDraggingPoint(points.length); // Set the new point as the dragging point
      setOffset({
        x: mousePos.x - mousePos.x,
        y: mousePos.y - mousePos.y,
      });
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const mousePos = getMousePos(e);
    const pointIndex = findPointIndex(mousePos);

    // Update hovered point
    setHoveredPoint(pointIndex >= 0 ? pointIndex : null);

    // Update dragging point position
    if (draggingPoint !== null) {
      const newPoints = [...points];
      newPoints[draggingPoint] = {
        x: mousePos.x - offset.x,
        y: mousePos.y - offset.y,
      };
      setPoints(newPoints);
    }
  };

  const handleMouseUp = () => {
    setDraggingPoint(null);
  };

  useEffect(()=>{
    setPoints_input(points.map((point) => ({ x: point.x/WIDTH+0.5, y: point.y/HEIGHT+0.5 })).sort((a, b) => a.x - b.x));
  }, [points]);

  //https://chatgpt.com/c/68205dbe-9f20-800a-b9d0-f1befc8dae07
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const container = scrollContainerRef.current;
    if (container) {
      // Scroll to center on initial render
      container.scrollLeft = (container.scrollWidth - container.clientWidth) / 2;
    }
  }, []);

  React.useImperativeHandle(ref, () => ({
    setPointsFromInput: (inputPoints: Point[]) => {
      setPoints(inputPoints.map((point) => ({ x: (point.x-0.5) * WIDTH, y: (point.y-0.5) * HEIGHT })));
    }
  }));

  return (
    <div className="flex flex-col items-center p-4">
      <h1 className="text-2xl font-bold mb-4">Interactive Graph</h1>
      <div
        ref={scrollContainerRef}
        className="w-full overflow-x-auto"
        style={{ border: "1px solid #ccc" }}
      >
        <div style={{ width: `${WIDTH}px`, display: "flex", justifyContent: "center" }}>
          <canvas
            ref={canvasRef}
            width={WIDTH}
            height={HEIGHT}
            className="border border-gray-300"
            style={{ cursor: hoveredPoint !== null ? "pointer" : "default" }}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            onContextMenu={(e) => e.preventDefault()}
          />
        </div>
      </div>

      <div className="flex gap-4 mt-4">
        <button
          onClick={() => {
            // console.log(
            //   computeRegressionPoints().map(
            //     (point) =>
            //       `${(point.x / WIDTH).toFixed(2)},${(1 - point.y / HEIGHT).toFixed(2)}`
            //   )
            // );
            console.log("Anchor: ", JSON.stringify(points.map((point) => ({ x: point.x/WIDTH+0.5, y: point.y/HEIGHT+0.5 })).sort((a, b) => a.x - b.x)));
            computeRegressionPoints();
            // console.log(points.map((point) => `${(point.x / WIDTH).toFixed(2)},${(1 - point.y / HEIGHT).toFixed(2)}`));
          }}
          className="px-6 py-2 w-32 bg-blue-600 text-white rounded-md shadow-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-colors duration-200"
        >
          Update
        </button>

        <button
          onClick={() => {
            setPoints([]);
            setIsViewData(false);
          }}
          className="px-6 py-2 w-32 bg-gray-600 text-white rounded-md shadow-sm hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 transition-colors duration-200"
        >
          Reset
        </button>
      </div>

      {/* <div className="flex flex-col items-center justify-center h-screen gap-4">
      <Switch hidden={true}
        checked={isViewData}
        onCheckedChange={setIsViewData}
      />
      </div> */}

    </div>
  );
});


export default InteractiveGraph;