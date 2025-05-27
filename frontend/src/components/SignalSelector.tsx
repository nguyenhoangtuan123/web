import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
// import { Sample } from "@/types/sample"; // Adjust path based on your project structure

interface Sample {
  id: string | number;
  label: string;
}

interface SignalSelectorProps {
  samples: Sample[];
  selectedSignal: number | null;
  onSelectChange: (value: string) => void; // Adjusted to match Shadcn Select's onValueChange
}

const SignalSelector: React.FC<SignalSelectorProps> = ({ samples, selectedSignal, onSelectChange }) => {
  return (
    <div className="mb-4">
      <label htmlFor="signalSelect" className="block text-sm font-medium text-foreground mb-2">
        Select Heartbeat Signal
      </label>
      <Select
        onValueChange={onSelectChange}
        value={selectedSignal !== null ? selectedSignal.toString() : ""}
      >
        <SelectTrigger
          id="signalSelect"
          className="w-full bg-background border-input text-foreground focus:ring-2 focus:ring-ring focus:ring-offset-2"
        >
          <SelectValue placeholder="Select a signal..." />
        </SelectTrigger>
        <SelectContent>
          {samples.map((sample, index) => (
            <SelectItem key={sample.id} value={index.toString()}>
              Signal {sample.id} (Label: {sample.label})
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
};

export default SignalSelector;