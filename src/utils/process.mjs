import sharp from 'sharp'
import { NMS, softmax } from './utils.mjs'

/**
 * 이미지를 전처리하여 텐서로 변환합니다.
 * @param {Buffer[]} imageBuffers 입력 이미지 버퍼 배열.
 * @param {number[]} targetSize 이미지의 타겟 크기 [width, height].
 * @returns {Promise<{inputTensor: Float32Array, originalSizes: number[][]}>} 전처리된 텐서와 원본 이미지 크기 배열.
 */
export async function preprocess(imageBuffers, targetSize) {
  const processImage = async (imageBuffer) => {
    const metadata = await sharp(imageBuffer).metadata()
    const originalSize = [metadata.height, metadata.width]

    const { data: buffer, info } = await sharp(imageBuffer)
      .resize(targetSize[0], targetSize[1], { fit: 'fill' })
      .raw()
      .toBuffer({ resolveWithObject: true })
    const { width, height, channels } = info

    const data = new Float32Array(channels * height * width)
    for (let i = 0; i < buffer.length; i++) {
      const c = i % channels
      const h = Math.floor(i / (width * channels))
      const w = Math.floor((i % (width * channels)) / channels)
      data[c * height * width + h * width + w] = buffer[i] / 255.0
    }

    return { data, originalSize }
  }

  const results = await Promise.all(imageBuffers.map(processImage))

  const inputTensor = new Float32Array(
    results.length * 3 * targetSize[1] * targetSize[0],
  )
  const originalSizes = []

  results.forEach(({ data, originalSize }, i) => {
    inputTensor.set(data, i * 3 * targetSize[1] * targetSize[0])
    originalSizes.push(originalSize)
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
export function postprocess(output, options) {
  const {
    confidenceThreshold,
    iouThreshold,
    targetSize,
    originalSizes,
    labels,
    taskType,
    batch,
  } = options
  const results = []

  switch (taskType) {
    case 'detection':
      {
        const [, channels, numPredictions] = output.dims
        for (let b = 0; b < batch; b++) {
          const outputData = output.data.slice(
            b * channels * numPredictions,
            (b + 1) * channels * numPredictions,
          )
          results.push(
            postprocessDetection(
              outputData,
              confidenceThreshold,
              iouThreshold,
              targetSize,
              originalSizes[b],
              numPredictions,
              channels,
            ),
          )
        }
      }
      break
    case 'classification':
      {
        const isLogit = output.data[0] < 0 || output.data[0] > 1
        for (let b = 0; b < batch; b++) {
          const outputData = output.data.slice(
            b * labels.length,
            (b + 1) * labels.length,
          )
          results.push(
            postprocessClassification(
              outputData,
              labels,
              confidenceThreshold,
              isLogit,
            ),
          )
        }
      }
      break
    default:
      throw new Error(`Unsupported task: ${taskType}`)
  }
  return results
}

export function postprocessDetection(
  outputData,
  confThreshold,
  iouThreshold,
  targetSize,
  originalSize,
  numPredictions,
  channels,
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

  let boxes = []
  for (const prediction of predictions) {
    const [x_center, y_center, width, height, ...confidences] = prediction
    const maxConfidence = Math.max(...confidences)
    const classIndex = confidences.indexOf(maxConfidence)

    if (maxConfidence > confThreshold) {
      const x = (x_center - width / 2) / targetSize[0]
      const y = (y_center - height / 2) / targetSize[1]
      const w = width / targetSize[0]
      const h = height / targetSize[1]
      boxes.push([x, y, w, h, maxConfidence, classIndex])
    }
  }

  boxes = NMS(boxes, iouThreshold)

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
  outputData,
  labels,
  confidenceThreshold,
  isLogit,
) {
  // logit인 경우 softmax 적용
  const probabilities = isLogit ? softmax(outputData) : outputData

  const results = Array.from(probabilities)
    .map((confidence, index) => ({
      label: labels[index],
      confidence: confidence,
    }))
    .filter((item) => item.confidence >= confidenceThreshold)
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, 3)

  return results.length > 0 ? results : [{ label: 'Unknown', confidence: 0 }]
}

/**
 * 결과를 형식화합니다.
 * @param {Array} processedOutput 처리된 출력 데이터
 * @param {string[]} labels 클래스 레이블 배열
 * @param {string} taskType 작업 유형 ('detection' 또는 'classification')
 * @returns {Object} 형식화된 결과 객체
 */
export function formatResult(processedOutput, labels, taskType) {
  switch (taskType) {
    case 'detection':
      return {
        detections: processedOutput.map(([x, y, w, h, conf, classIndex]) => ({
          label: labels[classIndex],
          box: [x, y, w, h],
          squareness: Number((1 - Math.abs(1 - Math.min(w, h) / Math.max(w, h))).toFixed(4)),
          confidence: Number(conf.toFixed(4)),
        })),
      }

    case 'classification':
      return {
        classifications: processedOutput.map(
          ({ label: LabelName, confidence }) => ({
            label: LabelName,
            confidence: Number(confidence.toFixed(4)),
          }),
        ),
      }

    default:
      throw new Error(`Unsupported task: ${taskType}`)
  }
}
