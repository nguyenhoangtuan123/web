import { useEffect, useRef, useState } from "react"
import InteractiveGraph, { type InteractiveGraphRef } from "./InteractiveGraph"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Button } from "@/components/ui/button"
import SignalChart from "@/components/SignalChart"
import SignalSelector from "@/components/SignalSelector"
import ClassificationResult from "@/components/ClassificationResult"
import { FileUpload } from "@/components/FileUpload"
import ShapValuesChart from "@/components/ShapValuesChart"

const tmp = [
    { x: 0.01, y: 0.01 },
    { x: 0.6, y: 0.4 },
    { x: 0.7, y: 0.7 },
    { x: 0.99, y: 0.99 }
]

interface Point {
    x: number;
    y: number;
}

interface AnchorData {
    name: string;
    anchor: Point[];
    label: number;
}

interface Sample {
    // Define the structure of a sample as returned by your backend
    // Example:
    // id: number;
    // name: string;
    // data: any;
    id: string | number;
    label: string;
    signal: number[];
}

interface FetchSamplesResponse {
    samples?: Sample[];
    error?: string;
}

type SetSamples = (samples: Sample[]) => void;

const fetchingSamples = async (setSamples: SetSamples): Promise<void> => {
    fetch('http://192.168.1.1:5000/get_samples')
    .then((response: Response) => {
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        return response.json() as Promise<FetchSamplesResponse>;
    })
    .then((data: FetchSamplesResponse) => {
        console.log('Loaded samples:', data);
        if (data.samples) {
            setSamples(data.samples);
        } else {
            alert('Không thể tải dữ liệu mẫu: ' + (data.error || 'Lỗi không xác định'));
        }
    })
    .catch((error: Error) => {
        console.error('Lỗi khi tải dữ liệu mẫu:', error);
        alert('Lỗi khi tải dữ liệu mẫu: ' + error.message);
    });
}

interface ClassificationResultType {
    // Define the expected structure of the classification result here
    // For example:
    results: {
        model: string;
    probabilities: number[];
    prediction: string;
    }[];
}

const fetchingResult = async (
    signalToClassify: number[],
    setClassificationResult: React.Dispatch<React.SetStateAction<ClassificationResultType | null>>,
    isLoadingResult: React.Dispatch<React.SetStateAction<boolean>>
) => {
    isLoadingResult(true);
    fetch('http://192.168.1.1:5000/classify', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ signal: signalToClassify }),
        })
        .then((response) => {
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            isLoadingResult(false);
            return response.json();
        })
        // .then((data) => {
        //     if (data.error) {
        //         alert('Lỗi từ server: ' + data.error);
        //         return;
        //     }
        //     console.log('Classification result:', data);
        //     setClassificationResult(data);
        // })
        .then((data) => {
            if (data.error) {
                alert('Lỗi từ server: ' + data.error);
                return;
            }
            console.log('Classification result:', data);
            setClassificationResult(data);

            // Gửi kết quả lên server
            const prediction = data?.results?.[0]?.prediction;
            if (prediction) {
                saveResultToDB(prediction);
            }
        })

        .catch((error) => {
            console.error('Classification error:', error.message);
            alert('Error classifying signal: ' + error.message);
        });

};

const InteractivePage = () => {
    const [points, setPoints] = useState<Point[]>(tmp)
    const [dataPoints, setDataPoints] = useState<Point[]>([])
    const [anchorData, setAnchorData] = useState<AnchorData[]>([])
    const [selectedAnchor, setSelectedAnchor] = useState<string>("")
    const [isLoadingResult, setIsLoadingResult] = useState<boolean>(false);

    const graphRef = useRef<InteractiveGraphRef>(null)

    // Fetch anchor data on component mount
    useEffect(() => {
        fetch("/sample_anchor.json")
            .then(response => response.json())
            .then(data => setAnchorData(data))
            .catch(error => console.error("Error fetching anchor data:", error))
    }, [])

    // Update graph when dataPoints change
    // useEffect(() => {
    //     console.log("points", JSON.stringify(points))
    //     graphRef.current?.setPointsFromInput()
    // }, [dataPoints])

    // Handle loading selected anchor
    const handleLoadAnchor = () => {
        if (selectedAnchor) {
            const selectedData = anchorData.find(data => data.name === selectedAnchor)
            if (selectedData?.anchor) {
                setPoints(selectedData.anchor)
                graphRef.current?.setPointsFromInput(selectedData.anchor)
            }
        }
    }

    const [classificationResult, setClassificationResult] = useState<ClassificationResultType | null>(null);
    
    const [selectedSignal, setSelectedSignal] = useState<number | null>(null);
    const [samples, setSamples] = useState<Sample[]>([]);
    const [, setSelectedSample] = useState<Sample | null>(null);
    useEffect(() => {
        fetchingSamples(setSamples);
    }, []);


    const handleSelectChange = (value: string) => {
        setSelectedSignal(value ? parseInt(value) : null);
        setClassificationResult(null);
        if (value === null) {
            setSelectedSample(null);
        } else {
            setSelectedSample(samples[parseInt(value)]);
            setDataPoints(samples[parseInt(value)].signal.map((x: number, idx: number) => ({ x: idx / samples[parseInt(value)].signal.length, y: x })));
        }
    };
    // luu kết quả phân loại vào cơ sở dữ liệu
    const saveResultToDB = async (result: string) => {
        try {
            const res = await fetch("http://localhost:5000/save-result", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ result })  // gửi chuỗi kết quả như "Normal", "AFib", v.v.
            });

            const data = await res.json();
            console.log("Saved result:", data);
        } catch (err) {
            console.error("Failed to save result", err);
        }
    };

    


    return (
        <div className="p-4 space-y-4">
            <div className="flex items-center gap-2">
                <Select 
                    value={selectedAnchor} 
                    onValueChange={setSelectedAnchor}
                >
                    <SelectTrigger className="w-[200px]">
                        <SelectValue placeholder="Select anchor data" />
                    </SelectTrigger>
                    <SelectContent>
                        {anchorData.map((data) => (
                            <SelectItem key={data.name} value={data.name}
							className="cursor-pointer">
                                {data.name || "Unnamed Anchor"}
                            </SelectItem>
                        ))}
                    </SelectContent>
                </Select>
                <Button 
                    onClick={handleLoadAnchor}
                    disabled={!selectedAnchor}
                >
                    Load Selected Anchor
                </Button>
            </div>
            <InteractiveGraph
                ref={graphRef}
                points_input={points}
                setPoints_input={setPoints}
                regPoints_output={dataPoints}
                setRegPoints_output={setDataPoints}
            />

            <SignalChart signalData={dataPoints.map((instance) =>  instance.y )} />
            <SignalSelector 
                samples={samples}
                selectedSignal={selectedSignal}
                onSelectChange={handleSelectChange}
            />
            <FileUpload
                onClassifySignals={(signals: number[][]) => {
                    const signal = signals[0];
                    // console.log("Signal to classify:", signal);
                    setDataPoints(signal.map((x: number, idx: number) => ({ x: idx / signal.length, y: x })));
                }}
            />
            <Button
                onClick={() => {
                    if (dataPoints.length>0) {
                        const signalToClassify = dataPoints.map((instance) =>  instance.y );
                        console.log("Signal to classify:", signalToClassify);
                        fetchingResult(signalToClassify, setClassificationResult, setIsLoadingResult);
                    } else {
                        alert("Please select a signal to classify.");
                    }
                }}
                disabled={dataPoints.length === 0}
            >
                Classify Signal
            </Button>
            {isLoadingResult && <div className="text-yellow-500">Classifying...</div>}
            <ClassificationResult classificationResult={classificationResult} />
            {classificationResult && (
                <ShapValuesChart signalToExplain={dataPoints.map((instance) => instance.y)} />
            )}
            {/* {dataPoints.length > 0 && (
                <ShapValuesChart signalToExplain={dataPoints.map((instance) => instance.y)} />
            )} */}
            <div className="h-20"/>
        </div>
    )
}

export default InteractivePage