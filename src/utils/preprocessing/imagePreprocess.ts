import sharp from 'sharp';
import type { TaskType } from '../../types';

const EMBEDDING_MEAN = [0.48145466, 0.4578275, 0.40821073];
const EMBEDDING_STD = [0.26862954, 0.26130258, 0.27577711];

interface ImageSlice {
  data: Float32Array;
  originalSize: [number, number];
  slice: {
    x: number;
    y: number;
    width: number;
    height: number;
    imageIndex: number;
  };
}

interface ProcessedImage {
  data: Float32Array;
  originalSize: [number, number];
}

async function processImageToTensor(
  imageBuffer: Buffer,
  targetSize: [number, number],
  taskType: TaskType,
  options?: {
    x?: number;
    y?: number;
    width?: number;
    height?: number;
    inputShape?: 'NCHW' | 'NHWC';
  }
): Promise<ProcessedImage> {
  const metadata = await sharp(imageBuffer).metadata();
  if (!metadata.width || !metadata.height) {
    throw new Error('Failed to get image dimensions');
  }

  const inputShape = options?.inputShape || 'NCHW';

  let processedImage: sharp.Sharp;
  if (taskType === 'embedding') {
    const scaleX = targetSize[0] / metadata.width;
    const scaleY = targetSize[1] / metadata.height;
    processedImage = sharp(imageBuffer, {
      failOnError: false,
      sequentialRead: true
    }).affine([scaleX, 0, 0, scaleY], {
      interpolator: 'bilinear',
      background: { r: 0, g: 0, b: 0, alpha: 1 }
    });
  } else {
    processedImage = sharp(imageBuffer, {
      failOnError: false,
      sequentialRead: true
    });

    if (options && (options.x !== undefined || options.y !== undefined || options.width !== undefined || options.height !== undefined)) {
      processedImage = processedImage.extract({
        left: options.x || 0,
        top: options.y || 0,
        width: options.width || metadata.width,
        height: options.height || metadata.height
      });
    }

    processedImage = processedImage.resize(targetSize[0], targetSize[1], {
      fit: 'fill',
      kernel: 'lanczos3'
    });
  }

  const { data: buffer, info } = await processedImage
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const { width, height, channels } = info;
  const pixelCount = height * width;
  const data = new Float32Array(channels * height * width);

  for (let i = 0; i < pixelCount; i++) {
    for (let c = 0; c < channels; c++) {
      const srcIdx = i * channels + c;
      let dstIdx: number;
      
      if (inputShape === 'NHWC') {
        // NHWC: [batch, height, width, channels]
        dstIdx = i * channels + c;
      } else {
        // NCHW: [batch, channels, height, width] (기본값)
        dstIdx = c * pixelCount + i;
      }
      
      const normalizedValue = buffer[srcIdx] / 255.0;

      if (taskType === 'embedding') {
        data[dstIdx] = (normalizedValue - EMBEDDING_MEAN[c]) / EMBEDDING_STD[c];
      } else {
        data[dstIdx] = normalizedValue;
      }
    }
  }

  return {
    data,
    originalSize: [
      options?.height || metadata.height,
      options?.width || metadata.width
    ] as [number, number]
  };
}

async function sliceDetectionImage(
  imageBuffer: Buffer,
  overlap: number,
  targetSize: [number, number],
  imageIndex: number,
  inputShape?: 'NCHW' | 'NHWC'
): Promise<ImageSlice[]> {
  const metadata = await sharp(imageBuffer).metadata();
  if (!metadata.width || !metadata.height) {
    throw new Error('Failed to get image dimensions');
  }

  // 1. 슬라이스 크기는 원본 이미지의 짧은 변 길이로 설정
  const sliceSize = Math.min(metadata.width, metadata.height);
  const stride = Math.floor(sliceSize * (1 - overlap));
  const slices: ImageSlice[] = [];

  // 가로/세로 슬라이스 개수 계산
  const numCols = Math.ceil((metadata.width - sliceSize) / stride) + 1;
  const numRows = Math.ceil((metadata.height - sliceSize) / stride) + 1;

  for (let row = 0; row < numRows; row++) {
    for (let col = 0; col < numCols; col++) {
      // 슬라이스 위치 계산
      let x = col * stride;
      let y = row * stride;
      
      // 마지막 슬라이스가 이미지 경계를 넘어가지 않도록 조정
      if (x + sliceSize > metadata.width) {
        x = metadata.width - sliceSize;
      }
      if (y + sliceSize > metadata.height) {
        y = metadata.height - sliceSize;
      }

      // 2. 정사각형 슬라이스를 추출하고 targetSize로 리사이즈
      const processed = await processImageToTensor(imageBuffer, targetSize, 'detection', {
        x, y, width: sliceSize, height: sliceSize, inputShape
      });

      slices.push({
        ...processed,
        slice: { x, y, width: sliceSize, height: sliceSize, imageIndex }
      });
    }
  }

  return slices;
}

export async function preprocess(
  imageBuffers: Buffer[],
  targetSize: [number, number],
  taskType: TaskType,
  sahi?: { overlap: number; aspectRatioThreshold?: number },
  inputShape?: 'NCHW' | 'NHWC'
) {
  // SAHI는 Detection에서만 사용
  if (taskType === 'detection' && sahi) {
    const allSlices: ImageSlice[] = [];
    const sliceInfos: { x: number; y: number; width: number; height: number; imageIndex: number }[] = [];
    const slicesPerImage: number[] = [];

    for (let i = 0; i < imageBuffers.length; i++) {
      // 이미지 메타데이터를 가져와서 비율 확인
      const metadata = await sharp(imageBuffers[i]).metadata();
      if (!metadata.width || !metadata.height) {
        throw new Error('Failed to get image dimensions');
      }

      const aspectRatio = Math.max(metadata.width, metadata.height) / Math.min(metadata.width, metadata.height);
      
      // aspectRatioThreshold가 설정되어 있고, 비율이 임계값보다 작으면 슬라이스 처리하지 않음
      if (sahi.aspectRatioThreshold && aspectRatio < sahi.aspectRatioThreshold) {
        const processed = await processImageToTensor(imageBuffers[i], targetSize, taskType, { inputShape });
        allSlices.push({
          ...processed,
          slice: { x: 0, y: 0, width: metadata.width, height: metadata.height, imageIndex: i }
        });
        slicesPerImage.push(1);
        sliceInfos.push({ x: 0, y: 0, width: metadata.width, height: metadata.height, imageIndex: i });
        continue;
      }

      const slices = await sliceDetectionImage(imageBuffers[i], sahi.overlap, targetSize, i, inputShape);
      allSlices.push(...slices);
      slicesPerImage.push(slices.length);
      sliceInfos.push(...slices.map(s => ({ ...s.slice, imageIndex: i })));
    }


    const inputTensor = new Float32Array(
      allSlices.length * 3 * targetSize[1] * targetSize[0]
    );

    allSlices.forEach(({ data }, i) => {
      inputTensor.set(data, i * 3 * targetSize[1] * targetSize[0]);
    });

    return {
      inputTensor,
      originalSizes: allSlices.map(s => s.originalSize),
      sliceInfo: sliceInfos,
      slicesPerImage
    };
  }

  // 일반적인 이미지 처리
  const results = await Promise.all(
    imageBuffers.map(buffer => processImageToTensor(buffer, targetSize, taskType, { inputShape }))
  );

  const inputTensor = new Float32Array(
    results.length * 3 * targetSize[1] * targetSize[0]
  );

  results.forEach(({ data }, i) => {
    inputTensor.set(data, i * 3 * targetSize[1] * targetSize[0]);
  });

  return {
    inputTensor,
    originalSizes: results.map(r => r.originalSize)
  };
} 