import { NMS, mergeNestedBoxes } from '../utils';
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
      // 좌표를 원본 이미지 크기로 변환
      const x = (x_center - width / 2) * (originalWidth / targetWidth);
      const y = (y_center - height / 2) * (originalHeight / targetHeight);
      const w = width * (originalWidth / targetWidth);
      const h = height * (originalHeight / targetHeight);

      boxes.push([x, y, w, h, maxConfidence, classIndex]);
    }
  }

  const nmsBoxes = NMS(boxes, iouThreshold) as [number, number, number, number, number, number][];
  return nmsBoxes;
}

export function processDetectionOutput(
  output: Tensor,
  confidenceThreshold: number,
  iouThreshold: number,
  targetSize: [number, number],
  originalSizes: [number, number][],
  batch: number,
  sliceInfo?: Array<{
    x: number;
    y: number;
    width: number;
    height: number;
    imageIndex: number;
  }>,
  mergeThreshold?: number
) {
  const outputData = output.data as Float32Array;
  const [, channels, numPredictions] = output.dims;

  // SAHI를 사용하지 않는 경우
  if (!sliceInfo) {
    const results: Array<[number, number, number, number, number, number][]> = [];
    
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

  // SAHI를 사용하는 경우
  const results: Map<number, Array<[number, number, number, number, number, number]>> = new Map();

  // 각 슬라이스 처리
  for (let b = 0; b < batch; b++) {
    const slicedData = outputData.slice(
      b * channels * numPredictions,
      (b + 1) * channels * numPredictions,
    );

    const { imageIndex, x: offsetX, y: offsetY, width: sliceWidth, height: sliceHeight } = sliceInfo[b];

    // 현재 이미지의 결과 배열 초기화
    if (!results.has(imageIndex)) {
      results.set(imageIndex, []);
    }

    // 기본 detection 수행 - 슬라이스 크기 전달
    const result = postprocessDetection(
      slicedData,
      confidenceThreshold,
      iouThreshold,
      targetSize,
      [sliceHeight, sliceWidth], // 슬라이스 크기 사용
      numPredictions,
      channels,
    ) as [number, number, number, number, number, number][];
    
    // 슬라이스 내 좌표를 원본 이미지 좌표로 변환
    for (const box of result) {
      box[0] += offsetX;  // x
      box[1] += offsetY;  // y
      // width와 height는 이미 올바른 스케일로 변환됨
    }

    // 현재 슬라이스의 결과를 해당 이미지의 결과 배열에 추가
    const currentResults = results.get(imageIndex);
    if (currentResults) {
      currentResults.push(...result);
    }
  }

  // 각 이미지별로 결과 병합
  const finalResults: Array<[number, number, number, number, number, number][]> = [];
  for (let i = 0; i < Math.max(...results.keys()) + 1; i++) {
    const imageResults = results.get(i) || [];
    if (imageResults.length > 0) {
      if (mergeThreshold) {
        const mergedResults = mergeNestedBoxes(imageResults, mergeThreshold);
        finalResults.push(mergedResults);
      } else {
        finalResults.push(imageResults);
      }
    } else {
      // 빈 결과 추가
      finalResults.push([]);
    }
  }

  return finalResults;
}