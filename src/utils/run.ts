import { InferenceSession, Tensor } from 'onnxruntime-node';
import { preprocess, postprocess, formatResult } from './process';
import { drawResult } from './draw';
import type { TaskResult, TaskType, TaskOptions, ProcessedOutput } from '../types';
import fs from 'node:fs';

/**
 * ONNX 모델을 로드하고 추론을 실행합니다.
 * @param {string} modelPath ONNX 모델 파일 경로
 * @param {Float32Array} inputTensor 입력 텐서 데이터
 * @param {number} batch 배치 크기
 * @param {number[]} targetSize 모델 입력 이미지 크기 [height, width]
 * @returns {Promise<ort.Tensor>} 모델의 출력 텐서
 */
export async function runInference(
  modelPath: string,
  inputTensor: Float32Array,
  batch: number,
  targetSize: [number, number]
): Promise<Tensor> {
  const session = await InferenceSession.create(modelPath);
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];
  const feeds = {
    [inputName]: new Tensor('float32', inputTensor, [batch, 3, ...targetSize]),
  };
  const results = await session.run(feeds);
  return results[outputName];
}

/**
 * 작업을 수행합니다.
 * @param {Buffer[]} imageBuffers 입력 이미지 버퍼 배열
 * @param {Object} options 설정 옵션
 * @param {string} taskType 작업 유형 ('detection' 또는 'classification')
 * @returns {Promise<string>} JSON 형식의 결과
 */
export async function runTask(
  imageBuffers: Buffer[],
  options: TaskOptions,
  taskType: TaskType
): Promise<TaskResult[]> {
  const {
    modelPath,
    labels = [],
    iouThreshold = 0.45,
    confidenceThreshold = 0.2,
    targetSize = [384, 384],
    headless = false,
  } = options;

  if (!modelPath) {
    throw new Error('Model path is required');
  }

  try {
    const { inputTensor, originalSizes } = await preprocess(
      imageBuffers,
      targetSize,
      taskType
    );
    const batch = imageBuffers.length;
    const output = await runInference(modelPath, inputTensor, batch, targetSize);
    
    const processedOutputs = postprocess(output, {
      confidenceThreshold,
      iouThreshold,
      targetSize,
      originalSizes,
      labels,
      taskType: taskType as 'detection' | 'classification',
      batch,
    });

    const results = await Promise.all(
      processedOutputs.map(async (processedOutput, bat) => {
        if (!headless && taskType !== 'embedding') {
          const drawOutput = taskType === 'detection' 
            ? processedOutput as number[][] 
            : processedOutput as { label: string; confidence: number; }[];
          await drawResult(
            imageBuffers[bat],
            drawOutput,
            `./output/${taskType}/${bat + 1}.png`,
            { labels, taskType: taskType as 'detection' | 'classification' },
          );
        }
        const formattedOutput = taskType === 'embedding' 
          ? processedOutput as number[][]
          : taskType === 'detection'
            ? { 
                boxes: (processedOutput as number[][]).map(([x, y, w, h]) => [x, y, w, h] as [number, number, number, number]),
                scores: (processedOutput as number[][]).map(box => box[4]),
                labels: (processedOutput as number[][]).map(box => box[5])
              }
            : { label: 0, confidence: 0 };
        return formatResult(formattedOutput, labels, taskType);
      })
    );

    return results;
  } catch (error) {
    throw new Error();
  }
} 