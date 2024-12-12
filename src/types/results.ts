export interface DetectionItem {
    label: string;
    box: number[];
    confidence: number;
    squareness: number;
}

export interface DetectionResult {
    detections: DetectionItem[];
}

export interface ClassificationItem {
    label: string;
    confidence: number;
}

export interface ClassificationResult {
    classifications: ClassificationItem[];
}

export type EmbeddingResult = number[];

// 각 작업별 결과 배열 타입
export type DetectionResults = DetectionResult[];
export type ClassificationResults = ClassificationResult[];
export type EmbeddingResults = EmbeddingResult[];

// Union type for all possible task results
export type TaskResult = DetectionResults | ClassificationResults | EmbeddingResults; 