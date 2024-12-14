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
  const predictions = []

  for (let i = 0; i < numPredictions; i++) {
    const pred = []
    for (let c = 0; c < channels; c++) {
      const index = i + c * numPredictions
      pred.push(outputData[index])
    }
    predictions.push(pred)
  }

  let boxes: [number, number, number, number, number, number][] = []

  for (const prediction of predictions) {
    const [x_center, y_center, width, height, ...confidences] = prediction
    const maxConfidence = Math.max(...confidences)
    const classIndex = confidences.indexOf(maxConfidence)

    if (maxConfidence > confThreshold) {
      boxes.push([
        (x_center - width / 2) / targetSize[0],
        (y_center - height / 2) / targetSize[1],
        width / targetSize[0],
        height / targetSize[1],
        maxConfidence,
        classIndex
      ])
    }
  }

  boxes = NMS(boxes, iouThreshold) as [number, number, number, number, number, number][]

  return boxes.map((box) => {
    const [x, y, w, h, conf, classIndex] = box
    const rescaledX = Math.round(x * originalSize[1])
    const rescaledY = Math.round(y * originalSize[0])
    const rescaledW = Math.round(w * originalSize[1])
    const rescaledH = Math.round(h * originalSize[0])
    return [rescaledX, rescaledY, rescaledW, rescaledH, conf, classIndex]
  })
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