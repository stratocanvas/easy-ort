import sharp from 'sharp';
import type { TaskType } from '../../types';

const EMBEDDING_MEAN = [0.48145466, 0.4578275, 0.40821073];
const EMBEDDING_STD = [0.26862954, 0.26130258, 0.27577711];

/**
 * 이미지를 전처리하여 텐서로 변환합니다.
 */
export async function preprocess(
  imageBuffers: Buffer[],
  targetSize: [number, number],
  taskType: TaskType
) {
  const processImage = async (imageBuffer: Buffer) => {
    const metadata = await sharp(imageBuffer).metadata()
    const originalSize = [metadata.height, metadata.width]
    let processedImage: sharp.Sharp
    if (taskType === 'embedding') {
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

    const hwcData = new Float32Array(height * width * channels)
    for (let i = 0; i < buffer.length; i++) {
      hwcData[i] = buffer[i] / 255.0
    }

    const data = new Float32Array(channels * height * width)
    for (let h = 0; h < height; h++) {
      for (let w = 0; w < width; w++) {
        for (let c = 0; c < channels; c++) {
          const srcIdx = (h * width + w) * channels + c
          const dstIdx = c * height * width + h * width + w
          
          if (taskType === 'embedding') {
            data[dstIdx] = (hwcData[srcIdx] - EMBEDDING_MEAN[c]) / EMBEDDING_STD[c]
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