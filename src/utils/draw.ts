import sharp from 'sharp'
import { Buffer } from 'node:buffer'

interface DrawOptions {
  labels: string[]
  taskType: 'detection' | 'classification'
}

interface ClassificationPrediction {
  label: string
  confidence: number
}

/**
 * 이미지에 inference 결과를 그려서 저장합니다.
 * @param {Buffer} imageBuffer - 입력 이미지 버퍼.
 * @param {number[][] | ClassificationPrediction[]} results - 추론 결과.
 * @param {string} outputPath - 출력 이미지의 경로.
 * @param {DrawOptions} options - 드로잉 옵션.
 */
export async function drawResult(
  imageBuffer: Buffer,
  results: number[][] | ClassificationPrediction[],
  outputPath: string,
  options: DrawOptions
): Promise<void> {
  const { labels, taskType } = options
  const image = sharp(imageBuffer)
  const metadata = await image.metadata()
  const { width = 0, height = 0 } = metadata

  let svgBuffer: Buffer
  if (taskType === 'detection') {
    svgBuffer = drawDetection(results as number[][], width, height, labels)
  } else {
    svgBuffer = drawClassification(results as ClassificationPrediction[], width, height)
  }

  await image
    .composite([{ input: svgBuffer, blend: 'over' }])
    .toFile(outputPath)
}

/**
 * 바운딩 박스를 SVG 오버레이로 그립니다.
 * @param {number[][]} boxes 바운딩 박스 배열 [x, y, w, h, confidence].
 * @param {number} imageWidth 이미지의 너비.
 * @param {number} imageHeight 이미지의 높이.
 * @param {string[]} labels 레이블 배열.
 * @returns {Buffer} SVG 이미지 데이터 버퍼.
 */
function drawDetection(
  boxes: number[][],
  imageWidth: number,
  imageHeight: number,
  labels: string[]
): Buffer {
  const colorPalette = generateColorPalette(labels.length)
  
  const svgRectangles = boxes
    .map((box, index) => {
      const [x, y, w, h, conf, classIndex] = box
      const label = `${labels[classIndex]}: ${conf.toFixed(2)}`
      const color = colorPalette[classIndex % colorPalette.length]
      
      // Calculate label width and position
      const labelWidth = label.length * 7.5 + 16
      const labelX = Math.min(x, imageWidth - labelWidth)
      const labelY = Math.max(y - 30, 10)
      
      return `
        <g class="detection-group">
          <rect x="${x}" y="${y}" width="${w}" height="${h}" rx="4" ry="4"
                class="bounding-box" stroke="${color}" stroke-dasharray="8 4" />
          <rect x="${labelX}" y="${labelY}" width="${labelWidth}" height="24" rx="12" ry="12"
                class="label-background" fill="${color}" />
          <text x="${labelX + 8}" y="${labelY + 17}" class="label-text">${label}</text>
        </g>
      `
    })
    .join('\n')

  const svgImage = `
    <svg width="${imageWidth}" height="${imageHeight}" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <filter id="dropShadow">
          <feDropShadow dx="0" dy="2" stdDeviation="3" flood-opacity="0.3" />
        </filter>
        <style>
          .bounding-box {
            fill: none;
            stroke-width: 2.5;
            stroke-linecap: round;
            stroke-linejoin: round;
            filter: url(#dropShadow);
          }
          .label-background {
            fill-opacity: 0.9;
            filter: url(#dropShadow);
          }
          .label-text {
            fill: white;
            font-size: 13px;
            font-family: Arial, sans-serif;
            font-weight: bold;
            text-shadow: 0px 1px 2px rgba(0,0,0,0.2);
          }
        </style>
      </defs>
      ${svgRectangles}
    </svg>
  `
  return Buffer.from(svgImage)
}

/**
 * 색상 팔레트를 생성합니다.
 * @param {number} count 색상 팔레트의 색상 개수
 * @returns {string[]} 색상 팔레트 배열
 */
function generateColorPalette(count: number): string[] {
  const palette: string[] = []
  for (let i = 0; i < count; i++) {
    const hue = (i * 137.508) % 360
    palette.push(`hsl(${hue}, 80%, 55%)`)
  }
  return palette
}

/**
 * 이미지 분류 결과를 SVG 오버레이로 그립니다.
 * @param {ClassificationPrediction[]} predictions 분류 결과 배열
 * @param {number} imageWidth 이미지의 너비
 * @param {number} imageHeight 이미지의 높이
 * @returns {Buffer} SVG 이미지 데이터 버퍼
 */
function drawClassification(
  predictions: ClassificationPrediction[],
  imageWidth: number,
  imageHeight: number,
): Buffer {
  const svgContent = `
    <svg width="${imageWidth}" height="${imageHeight}" xmlns="http://www.w3.org/2000/svg">
      <rect x="10" y="10" width="320" height="${70 + predictions.length * 50}" rx="15" ry="15"
            fill="rgba(0,0,0,0.7)" />
      <rect x="10" y="10" width="320" height="${70 + predictions.length * 50}" rx="15" ry="15"
            fill="none" stroke="rgba(255,255,255,0.3)" stroke-width="2"/>
      <text x="25" y="45" fill="white" font-size="22" font-weight="bold" font-family="Arial, sans-serif">
        Classification Results
      </text>
      ${predictions
        .map(
          (pred, index) => `
        <g transform="translate(0,${index * 50})">
          <text x="25" y="${95}" fill="white" font-size="16" font-family="Arial, sans-serif">
            ${index + 1}. ${pred.label}
          </text>
          <text x="300" y="${95}" fill="white" font-size="16" font-family="Arial, sans-serif" text-anchor="end">
            ${(pred.confidence * 100).toFixed(1)}%
          </text>
          <rect x="25" y="${105}" width="280" height="12" rx="6" ry="6" fill="rgba(255,255,255,0.1)"/>
          <rect x="25" y="${105}" width="${pred.confidence * 280}" height="12" rx="6" ry="6" 
                fill="${index === 0 ? 'rgba(29,209,161,0.8)' : 'rgba(255,255,255,0.8)'}"/>
        </g>
      `
        )
        .join('')}
    </svg>
  `
  return Buffer.from(svgContent)
} 