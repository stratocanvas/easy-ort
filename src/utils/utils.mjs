/**
 * 두 바운딩 박스 사이의 IoU(교집합/합집합)를 계산합니다.
 * @param {number[]} box1 첫 번째 바운딩 박스 [x, y, w, h].
 * @param {number[]} box2 두 번째 바운딩 박스 [x, y, w, h].
 * @returns {number} IoU 값.
 */
export function iou(box1, box2) {
  const [x1, y1, w1, h1] = box1
  const [x2, y2, w2, h2] = box2

  const xi1 = Math.max(x1, x2)
  const yi1 = Math.max(y1, y2)
  const xi2 = Math.min(x1 + w1, x2 + w2)
  const yi2 = Math.min(y1 + h1, y2 + h2)
  const interArea = Math.max(0, xi2 - xi1) * Math.max(0, yi2 - yi1)

  const box1Area = w1 * h1
  const box2Area = w2 * h2
  const unionArea = box1Area + box2Area - interArea

  return interArea / unionArea
}

/**
 * 비최대 억제(NMS)를 적용하여 중복되는 바운딩 박스를 제거합니다.
 * @param {number[][]} boxes 바운딩 박스 배열 [x, y, w, h, confidence].
 * @param {number} iouThreshold IoU 임계값.
 * @returns {number[][]} NMS가 적용된 바운딩 박스 배열.
 */
export function NMS(boxes, iouThreshold) {
  // 신뢰도 점수를 기준으로 내림차순 정렬
  const sortedBoxes = boxes.sort((a, b) => b[4] - a[4])
  const nmsBoxes = []

  while (sortedBoxes.length > 0) {
    const chosenBox = sortedBoxes.shift()
    nmsBoxes.push(chosenBox)

    const remainingBoxes = sortedBoxes.filter(
      (box) => iou(chosenBox, box) < iouThreshold,
    )
    sortedBoxes.length = 0
    sortedBoxes.push(...remainingBoxes)
  }
  return nmsBoxes
}

/**
 * Logit을 softmax를 적용하여 확률로 변환합니다.
 * @param {number[]} arr logit 배열
 * @returns {number[]} softmax 배열
 */
export function softmax(arr) {
  const maxLogit = Math.max(...arr)
  const scores = arr.map((logit) => Math.exp(logit - maxLogit))
  const sum = scores.reduce((a, b) => a + b, 0)
  return scores.map((score) => score / sum)
}
