import { softmax } from '../utils';
import type { Tensor } from 'onnxruntime-node';

export function postprocessClassification(
  outputData: Float32Array,
  labels: string[],
  confidenceThreshold: number,
  isLogit: boolean,
): Array<{label: string, confidence: number}> {
  const probabilities = isLogit ? softmax(Array.from(outputData)) : outputData;

  return Array.from(probabilities)
    .map((confidence, index) => ({
      label: labels[index],
      confidence: confidence,
    }))
    .filter((item) => item.confidence >= confidenceThreshold)
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, 3);
}

export function processClassificationOutput(
  output: Tensor,
  labels: string[],
  confidenceThreshold: number,
  batch: number,
) {
  const outputData = output.data as Float32Array;
  const results = [];
  const isLogit = outputData[0] < 0 || outputData[0] > 1;

  for (let b = 0; b < batch; b++) {
    const slicedData = outputData.slice(
      b * labels.length,
      (b + 1) * labels.length,
    );
    results.push(
      postprocessClassification(
        slicedData,
        labels,
        confidenceThreshold,
        isLogit,
      ),
    );
  }

  return results;
} 