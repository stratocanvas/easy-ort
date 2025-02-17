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
 * 한 박스가 다른 박스에 얼마나 포함되는지 계산합니다.
 */
function isContained(
  boxA: [number, number, number, number],
  boxB: [number, number, number, number],
  threshold = 0.7
): boolean {
  const [xa1, ya1, wa, ha] = boxA;
  const [xb1, yb1, wb, hb] = boxB;
  const xa2 = xa1 + wa;
  const ya2 = ya1 + ha;
  const xb2 = xb1 + wb;
  const yb2 = yb1 + hb;

  // 겹치는 영역 계산
  const interX1 = Math.max(xa1, xb1);
  const interY1 = Math.max(ya1, yb1);
  const interX2 = Math.min(xa2, xb2);
  const interY2 = Math.min(ya2, yb2);
  const interArea = Math.max(0, interX2 - interX1) * Math.max(0, interY2 - interY1);

  // boxB의 넓이 계산
  const areaB = wb * hb;
  if (areaB <= 0) return false;

  // 포함 비율 계산 및 threshold와 비교
  return (interArea / areaB) >= threshold;
}

/**
 * SAHI: 중첩된 박스들을 병합합니다.
 */
export function mergeNestedBoxes(
  boxes: Array<[number, number, number, number, number, number]>,
  containmentThreshold = 0.7
): Array<[number, number, number, number, number, number]> {
  const merged = new Array(boxes.length).fill(false);
  const mergedBoxes: Array<[number, number, number, number, number, number]> = [];

  for (let i = 0; i < boxes.length; i++) {
    if (merged[i]) continue;

    const [x1, y1, w1, h1, score1, label1] = boxes[i];
    let unionBox: [number, number, number, number] = [x1, y1, w1, h1];
    let maxScore = score1;
    let finalLabel = label1;

    for (let j = i + 1; j < boxes.length; j++) {
      if (merged[j]) continue;

      const [x2, y2, w2, h2, score2, label2] = boxes[j];
      const box2: [number, number, number, number] = [x2, y2, w2, h2];

      if (isContained(unionBox, box2, containmentThreshold) || 
          isContained(box2, unionBox, containmentThreshold)) {
        // union_box 업데이트 (두 box의 최소/최대 좌표)
        unionBox = [
          Math.min(unionBox[0], x2),
          Math.min(unionBox[1], y2),
          Math.max(unionBox[0] + unionBox[2], x2 + w2) - Math.min(unionBox[0], x2),
          Math.max(unionBox[1] + unionBox[3], y2 + h2) - Math.min(unionBox[1], y2)
        ];

        if (score2 > maxScore) {
          maxScore = score2;
          finalLabel = label2;
        }
        merged[j] = true;
      }
    }

    mergedBoxes.push([...unionBox, maxScore, finalLabel]);
  }

  return mergedBoxes;
}


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