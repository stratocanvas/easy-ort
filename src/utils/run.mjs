import { InferenceSession, Tensor } from 'onnxruntime-node'
import { preprocess, postprocess, formatResult } from './process.mjs'
import { drawResult } from './draw.mjs'

/**
 * ONNX 모델을 로드하고 추론을 실행합니다.
 * @param {string} modelPath ONNX 모델 파일 경로
 * @param {Float32Array} inputTensor 입력 텐서 데이터
 * @param {number} batch 배치 크기
 * @param {number[]} targetSize 모델 입력 이미지 크기 [height, width]
 * @returns {Promise<ort.Tensor>} 모델의 출력 텐서
 */
export async function runInference(modelPath, inputTensor, batch, targetSize) {
  // ONNX 모델 로드
  const session = await InferenceSession.create(modelPath)
  const inputName = session.inputNames[0]
  const outputName = session.outputNames[0]

  // 추론 실행
  const feeds = {
    [inputName]: new Tensor('float32', inputTensor, [batch, 3, ...targetSize]),
  }
  const results = await session.run(feeds)
  return results[outputName]
}


/**
 * 작업을 수행합니다.
 * @param {Buffer[]} imageBuffers 입력 이미지 버퍼 배열
 * @param {Object} options 설정 옵션
 * @param {string} taskType 작업 유형 ('detection' 또는 'classification')
 * @returns {Promise<string>} JSON 형식의 결과
 */
export async function runTask(imageBuffers, options, taskType) {
  const {
    modelPath,
    labels,
    iouThreshold = 0.45,
    confidenceThreshold = 0.2,
    targetSize = [384, 384],
    headless = false,
  } = options

  try {
    const { inputTensor, originalSizes } = await preprocess(
      imageBuffers,
      targetSize,
    )
    const batch = imageBuffers.length
    const output = await runInference(modelPath, inputTensor, batch, targetSize)
    
    // 한 번의 postprocess 호출로 모든 배치 처리
    const processedOutputs = postprocess(output, {
      confidenceThreshold,
      iouThreshold,
      targetSize,
      originalSizes,
      labels,
      taskType,
      batch,
    })

    const results = await Promise.all(processedOutputs.map(async (processedOutput, bat) => {
      if (!headless) {
        await drawResult(
          imageBuffers[bat],
          processedOutput,
          `./output/output_${bat + 1}.jpg`,
          { labels, taskType },
        )
      }
      return formatResult(processedOutput, labels, taskType)
    }))

    return JSON.stringify(results, null, 2)
  } catch (error) {
    throw new Error(`${error.message}`)
  }
}