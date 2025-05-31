import type { RuntimeSession } from './types/runtime';

export type TaskType = 'detection' | 'classification' | 'embedding';

export interface Box {
  x: number;
  y: number;
  width: number;
  height: number;
  confidence?: number;
}

export interface TaskOptions {
  labels?: string[];
  iouThreshold?: number;
  confidenceThreshold?: number;
  targetSize?: [number, number];
  dimension?: number;
  modelPath?: string;
  headless?: boolean;
  type?: 'image' | 'text';
  inputShape?: 'NCHW' | 'NHWC';
  sahi?: {
    overlap: number;
    minArea?: number;
    mergeThreshold?: number;
    aspectRatioThreshold?: number;
    sliceInfo?: Array<{
      x: number;
      y: number;
      width: number;
      height: number;
    }>;
  };
}

export interface ProcessOptions {
  confidenceThreshold: number;
  iouThreshold: number;
  targetSize: [number, number];
  originalSizes: [number, number][];
  labels: string[];
  taskType: TaskType;
  batch: number;
  shouldNormalize?: boolean;
  shouldMerge?: boolean;
  sahi?: {
    overlap: number;
    minArea?: number;
    mergeThreshold?: number;
    aspectRatioThreshold?: number;
    sliceInfo?: Array<{
      x: number;
      y: number;
      width: number;
      height: number;
    }>;
  };
}

export interface DrawOptions {
  labels: string[];
  taskType: TaskType;
}

export interface PreprocessResult {
  inputTensor: Float32Array;
  originalSizes: [number, number][];
  sliceInfo?: Array<{
    x: number;
    y: number;
    width: number;
    height: number;
    imageIndex: number;
  }>;
  slicesPerImage?: number[];
}

export interface DetectionOutput {
  boxes: Array<[number, number, number, number]>;  // [x, y, width, height]
  scores: number[];
  labels: number[];
}

export interface ClassificationOutput {
  label: number;
  confidence: number;
}

export interface EmbeddingOutput {
  embedding: number[];
}

export type ProcessedOutput = DetectionOutput | ClassificationOutput | EmbeddingOutput;

export interface DetectionResult {
  detections: Array<{
    label: string;
    box: [number, number, number, number];
    squareness: number;
    confidence: number;
  }>;
}

export interface ClassificationResult {
  classifications: Array<{
    label: string;
    confidence: number;
  }>;
}

export type TaskResult = DetectionResult | ClassificationResult | number[][] | number[];

export interface SessionManager {
  session: RuntimeSession | null;
  env: RuntimeSession | null;
  initialize(modelPath: string, options?: SessionOptions): Promise<void>;
  release(): Promise<void>;
}

export interface SessionOptions {
  enableCpuMemArena?: boolean;
  enableMemPattern?: boolean;
}