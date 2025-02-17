/**
 * Calculate IoU (Intersection over Union) between two bounding boxes
 */
export function iou(box1: [number, number, number, number], box2: [number, number, number, number]): number {
  const [x1, y1, w1, h1] = box1;
  const [x2, y2, w2, h2] = box2;

  const xi1 = Math.max(x1, x2);
  const yi1 = Math.max(y1, y2);
  const xi2 = Math.min(x1 + w1, x2 + w2);
  const yi2 = Math.min(y1 + h1, y2 + h2);
  const interArea = Math.max(0, xi2 - xi1) * Math.max(0, yi2 - yi1);

  const box1Area = w1 * h1;
  const box2Area = w2 * h2;
  const unionArea = box1Area + box2Area - interArea;

  return interArea / unionArea;
}

/**
 * Apply Soft Non-Maximum Suppression to adjust confidence scores of overlapping boxes
 * using Gaussian decay function
 */
export function NMS(
  boxes: [number, number, number, number, number, number][],
  iouThreshold: number,
  sigma = 0.5,
  scoreThreshold = 0.3
): [number, number, number, number, number, number][] {
  if (boxes.length === 0) return [];

  // 점수에 따라 인덱스 정렬
  let indices = Array.from({ length: boxes.length }, (_, i) => i)
    .sort((a, b) => boxes[b][4] - boxes[a][4]);
  
  const keepBoxes: [number, number, number, number, number, number][] = [];
  const scores = boxes.map(box => box[4]);

  while (indices.length > 0) {
    const currentIdx = indices[0];
    keepBoxes.push(boxes[currentIdx]);

    if (indices.length === 1) break;

    // 현재 박스를 제외한 나머지 박스들의 인덱스
    indices.splice(0, 1);

    // 현재 박스와 나머지 박스들 간의 IoU 계산
    const ious = indices.map(idx => 
      iou(
        [boxes[currentIdx][0], boxes[currentIdx][1], boxes[currentIdx][2], boxes[currentIdx][3]],
        [boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]]
      )
    );

    // Soft-NMS를 사용하여 점수 감소
    const weights = ious.map(iouValue => 
      Math.exp(-(iouValue * iouValue) / sigma)
    );

    // 남은 박스들의 점수 업데이트
    for (let i = 0; i < indices.length; i++) {
      scores[indices[i]] *= weights[i];
    }

    // scoreThreshold보다 낮은 점수를 가진 박스 제거
    indices = indices.filter(idx => scores[idx] > scoreThreshold);
  }

  return keepBoxes;
}

/**
 * Apply softmax to convert logits to probabilities
 */
export function softmax(arr: number[]): number[] {
  const maxLogit = Math.max(...arr);
  const scores = arr.map((logit) => Math.exp(logit - maxLogit));
  const sum = scores.reduce((a, b) => a + b, 0);
  return scores.map((score) => score / sum);
} 