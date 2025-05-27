import { useEffect } from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

interface ModelResult {
  model: string;
  prediction: string;
  probabilities: number[];
}

interface ClassificationResultProps {
  classificationResult: {
    results: ModelResult[];
  } | null;
}

const ClassificationResult: React.FC<ClassificationResultProps> = ({
  classificationResult,
}) => {
  useEffect(() => {
    if (classificationResult) {
      console.log("Classification result hehe:", classificationResult);
    }
  }, [classificationResult]);

  if (!classificationResult) return null;
  const classesName = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unknown'];

  return (
    <div className="mt-6">
      <h2 className="text-2xl font-bold mb-4 text-center text-primary">
        🧠 Classification Results
      </h2>
      <div className="overflow-auto rounded-xl border shadow-sm">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="min-w-[150px]">Model</TableHead>
              <TableHead className="min-w-[150px]">Prediction</TableHead>
              <TableHead className="min-w-[300px]">Probabilities</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {classificationResult.results.map((result, index) => (
              <TableRow key={index}>
                <TableCell className="font-medium">{result.model}</TableCell>
                <TableCell>{result.prediction}</TableCell>
                <TableCell>
                  {result.probabilities && Array.isArray(result.probabilities) ? (
                    <div className="flex flex-wrap gap-x-4 gap-y-1">
                      {result.probabilities.map((prob, idx) => (
                        <span
                          key={idx}
                          className="inline-block px-2 py-1 text-sm rounded-full bg-secondary text-secondary-foreground"
                        >
                          {classesName[idx]}: { (prob * 100).toFixed(prob < 0.1 ? 3 : 2)}%
                        </span>
                      ))}
                    </div>
                  ) : (
                    <span className="text-muted-foreground">Not available</span>
                  )}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  );
};

export default ClassificationResult;
