import { NMS } from '../utils';
import type { Tensor } from 'onnxruntime-node';

export function postprocessDetection(
  outputData: Float32Array,
  confThreshold: number,
  iouThreshold: number,
  targetSize: [number, number],
  originalSize: [number, number],
  numPredictions: number,
  channels: number,
) {
  const boxes: [number, number, number, number, number, number][] = [];
  const [targetWidth, targetHeight] = targetSize;
  const [originalHeight, originalWidth] = originalSize;

  for (let i = 0; i < numPredictions; i++) {
    const x_center = outputData[i];
    const y_center = outputData[i + numPredictions];
    const width = outputData[i + 2 * numPredictions];
    const height = outputData[i + 3 * numPredictions];

    let maxConfidence = Number.NEGATIVE_INFINITY;
    let classIndex = -1;

    // Find max confidence and class index directly
    for (let c = 4; c < channels; c++) {
      const confidence = outputData[i + c * numPredictions];
      if (confidence > maxConfidence) {
        maxConfidence = confidence;
        classIndex = c - 4;
      }
    }

    if (maxConfidence > confThreshold) {
      boxes.push([
        (x_center - width / 2) / targetWidth,
        (y_center - height / 2) / targetHeight,
        width / targetWidth,
        height / targetHeight,
        maxConfidence,
        classIndex
      ]);
    }
  }

  const nmsBoxes = NMS(boxes, iouThreshold) as [number, number, number, number, number, number][];

  // Optimize the rescaling loop
  return nmsBoxes.map(([x, y, w, h, conf, classIndex]) => [
    Math.round(x * originalWidth),
    Math.round(y * originalHeight),
    Math.round(w * originalWidth),
    Math.round(h * originalHeight),
    conf,
    classIndex
  ]);
}

export function processDetectionOutput(
  output: Tensor,
  confidenceThreshold: number,
  iouThreshold: number,
  targetSize: [number, number],
  originalSizes: [number, number][],
  batch: number,
) {
  const outputData = output.data as Float32Array;
  const results = [];
  const [, channels, numPredictions] = output.dims;

  for (let b = 0; b < batch; b++) {
    const slicedData = outputData.slice(
      b * channels * numPredictions,
      (b + 1) * channels * numPredictions,
    );
    const result = postprocessDetection(
      slicedData,
      confidenceThreshold,
      iouThreshold,
      targetSize,
      originalSizes[b],
      numPredictions,
      channels,
    );
    results.push(result);
  }

  return results;
} 