import sharp from 'sharp';
import { NMS, softmax } from './utils';
import type { Tensor } from 'onnxruntime-node';
import type {
  TaskType,
  TaskResult,
  ProcessedOutput,
  ProcessOptions,
} from '../types';

/**
 * 이미지를 전처리하여 텐서로 변환합니다.
 * @param {Buffer[]} imageBuffers 입력 이미지 버퍼 배열.
 * @param {number[]} targetSize 이미지의 타겟 크기 [width, height].
 * @param {string} taskType 작업 유형 ('detection' 또는 'classification')
 * @returns {Promise<{inputTensor: Float32Array, originalSizes: number[][]}>} 전처리된 텐서와 원본 이미지 크기 배열.
 */
export async function preprocess(
  imageBuffers: Buffer[],
  targetSize: [number, number],
  taskType: TaskType
) {
  // ImageNet mean and std values for embedding task
  const mean = [0.48145466, 0.4578275, 0.40821073];
  const std = [0.26862954, 0.26130258, 0.27577711];

  const processImage = async (imageBuffer: Buffer) => {
    const metadata = await sharp(imageBuffer).metadata()
    const originalSize = [metadata.height, metadata.width]
    let processedImage: sharp.Sharp
    if (taskType === 'embedding') {
      // For embedding task: use affine with bilinear interpolation
      if (!metadata.width || !metadata.height) {
        throw new Error('Failed to get image dimensions')
      }
      const scaleX = targetSize[0] / metadata.width
      const scaleY = targetSize[1] / metadata.height
      processedImage = sharp(imageBuffer)
        .affine([scaleX, 0, 0, scaleY], {
          interpolator: 'bilinear',
          background: { r: 0, g: 0, b: 0, alpha: 1 }
        })
    } else {
      // For other tasks: use regular resize
      processedImage = sharp(imageBuffer)
        .resize(targetSize[0], targetSize[1], {
          fit: 'fill',
          kernel: 'lanczos3',
          position: 'center',
          background: { r: 0, g: 0, b: 0, alpha: 1 }
        })
    }

    const { data: buffer, info } = await processedImage
      .removeAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true })

    const { width, height, channels } = info

    // Create a temporary array in HWC format and normalize to [0,1]
    const hwcData = new Float32Array(height * width * channels)
    for (let i = 0; i < buffer.length; i++) {
      hwcData[i] = buffer[i] / 255.0
    }

    // Convert HWC to CHW format (matching Python's transpose(2, 0, 1))
    const data = new Float32Array(channels * height * width)
    for (let h = 0; h < height; h++) {
      for (let w = 0; w < width; w++) {
        for (let c = 0; c < channels; c++) {
          const srcIdx = (h * width + w) * channels + c  // HWC index
          const dstIdx = c * height * width + h * width + w  // CHW index
          
          if (taskType === 'embedding') {
            data[dstIdx] = (hwcData[srcIdx] - mean[c]) / std[c]
          } else {
            data[dstIdx] = hwcData[srcIdx]
          }
        }
      }
    }

    return { data, originalSize }
  }

  const results = await Promise.all(imageBuffers.map(processImage))

  const inputTensor = new Float32Array(
    results.length * 3 * targetSize[1] * targetSize[0],
  )
  const originalSizes: [number, number][] = []

  results.forEach(({ data, originalSize }, i) => {
    inputTensor.set(data, i * 3 * targetSize[1] * targetSize[0])
    originalSizes.push(originalSize as [number, number])
  })

  return { inputTensor, originalSizes }
}

/**
 * 모델의 출력을 처리합니다.
 * @param {ort.Tensor} output 모델의 출력 텐서.
 * @param {number} confThreshold 신뢰도 임계값.
 * @param {number[]} targetSize 모델 입력 이미지 크기 [width, height].
 * @param {number[]} originalSize - 원본 이미지 크기 [width, height].
 * @param {number} batch 배치 크기.
 * @returns {number[][]} 바운딩 박스 배열 [x, y, w, h, confidence, class].
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
  const results = [];

  const outputData = output.data as Float32Array;

  switch (taskType) {
    case 'detection':
      {
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
      }
      break;
    case 'classification':
      {
        const isLogit = outputData[0] < 0 || outputData[0] > 1
        for (let b = 0; b < batch; b++) {
          const slicedData = outputData.slice(
            b * labels.length,
            (b + 1) * labels.length,
          )
          results.push(
            postprocessClassification(
              slicedData,
              labels,
              confidenceThreshold,
              isLogit,
            ),
          )
        }
      }
      break
    case 'embedding':
      {
        const [batchSize, dimension] = output.dims;
        for (let b = 0; b < batch; b++) {
          const slicedData = outputData.slice(
            b * dimension,
            (b + 1) * dimension,
          );
          let embedding = Array.from(slicedData);
          
          if (shouldNormalize) {
            // L2 normalization
            const norm = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
            embedding = embedding.map(val => val / norm);
          }
          
          results.push(embedding);
        }
        
        if (shouldMerge && batch > 1) {
          // Average embeddings
          const mergedEmbedding = new Array(results[0].length).fill(0);
          for (const result of results) {
            for (let i = 0; i < result.length; i++) {
              mergedEmbedding[i] += result[i] / batch;
            }
          }
          return [mergedEmbedding];
        }
      }
      break;
    default:
      throw new Error(`Unsupported task: ${taskType}`);
  }
  return results;
}

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

/**
 * 결과를 형식화합니다.
 * @param {Array} processedOutput 처리된 출력 데이터
 * @param {string[]} labels 클래스 레이블 배열
 * @param {string} taskType 작업 유형 ('detection' 또는 'classification')
 * @returns {Object} 형식화된 결과 객체
 */
export function formatResult(
  processedOutput: ProcessedOutput | number[][],
  labels: string[],
  taskType: TaskType
): TaskResult {
  switch (taskType) {
    case 'detection': {
      if (!Array.isArray(processedOutput)) throw new Error('Invalid detection output');
      return {
        detections: processedOutput.map(([x, y, w, h, conf, classIndex]) => ({
          label: labels[classIndex],
          box: [x, y, w, h],
          squareness: Number((1 - Math.abs(1 - Math.min(w, h) / Math.max(w, h))).toFixed(4)),
          confidence: Number(conf.toFixed(4)),
        })),
      };
    }

    case 'classification': {
      if (!Array.isArray(processedOutput)) {
        throw new Error('Invalid classification output format');
      }

      // Check if it's the expected format
      if (processedOutput.length > 0 && typeof processedOutput[0] === 'object' && 'label' in processedOutput[0]) {
        const outputs = processedOutput as unknown as Array<{label: string, confidence: number}>;
        return {
          classifications: outputs.map(({ label, confidence }) => ({
            label,
            confidence: Number(confidence.toFixed(4)),
          })),
        };
      }
      
      throw new Error('Invalid classification output format');
    }

    case 'embedding': {
      if (!Array.isArray(processedOutput)) throw new Error('Invalid embedding output');
      return processedOutput;
    }

    default:
      throw new Error(`Unsupported task: ${taskType}`);
  }
} 