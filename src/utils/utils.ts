import type { Box } from '../types';

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
 * Apply Non-Maximum Suppression to remove overlapping boxes
 */
export function NMS(boxes: [number, number, number, number, number, number][], iouThreshold: number): [number, number, number, number, number, number][] {
  const sortedBoxes = boxes.sort((a, b) => b[4] - a[4]);
  const nmsBoxes: [number, number, number, number, number, number][] = [];

  while (sortedBoxes.length > 0) {
    const chosenBox = sortedBoxes.shift();
    if (chosenBox) {
      nmsBoxes.push(chosenBox);

      const remainingBoxes = sortedBoxes.filter(
        (box) => iou(
          [chosenBox[0], chosenBox[1], chosenBox[2], chosenBox[3]], 
          [box[0], box[1], box[2], box[3]]
        ) < iouThreshold
      );
      sortedBoxes.length = 0;
      sortedBoxes.push(...remainingBoxes);
    }
  }
  return nmsBoxes;
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