import type { Tensor } from 'onnxruntime-node';
import type { ProcessOptions } from '../types';

import { processDetectionOutput } from './postprocessing/detection';
import { processClassificationOutput } from './postprocessing/classification';
import { processEmbeddingOutput } from './postprocessing/embedding';

/**
 * 모델의 출력을 처리합니다.
 */
export function postprocess(output: Tensor, options: ProcessOptions) {
  const {
    confidenceThreshold,
    iouThreshold,
    targetSize,
    originalSizes,
    labels,
    taskType,
    batch,
    shouldNormalize,
    shouldMerge,
  } = options;

  switch (taskType) {
    case 'detection':
      return processDetectionOutput(
        output,
        confidenceThreshold,
        iouThreshold,
        targetSize,
        originalSizes,
        batch,
      );
    case 'classification':
      return processClassificationOutput(
        output,
        labels,
        confidenceThreshold,
        batch,
      );
    case 'embedding':
      return processEmbeddingOutput(
        output,
        batch,
        shouldNormalize,
        shouldMerge,
      );
    default:
      throw new Error(`Unsupported task: ${taskType}`);
  }
} 