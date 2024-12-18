export type TaskResult = DetectionResult | ClassificationResult | EmbeddingResult;

export interface DetectionResult {
  detections: {
    label: string;
    box: [number, number, number, number];
    confidence: number;
    squareness: number;
  }[];
}

export interface ClassificationResult {
  classifications: {
    label: string;
    confidence: number;
  }[];
}

export type EmbeddingResult = number[][];

export function isDetectionResult(result: TaskResult): result is DetectionResult {
  return 'detections' in result;
}