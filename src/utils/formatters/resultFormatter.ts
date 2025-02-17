import type {
  TaskType,
  TaskResult,
  ProcessedOutput
} from '../../types';

/**
 * 결과를 형식화합니다.
 */
export function formatResult(
  processedOutput: ProcessedOutput | number[][],
  labels: string[],
  taskType: TaskType
): TaskResult {
  switch (taskType) {
    case 'detection': {
      if (!Array.isArray(processedOutput)) throw new Error('Invalid detection output');
      return {
        detections: processedOutput.map(([x, y, w, h, conf, classIndex]) => ({
          label: labels[classIndex],
          box: [
            Math.round(x),
            Math.round(y),
            Math.round(w),
            Math.round(h)
          ],
          squareness: Number((1 - Math.abs(1 - Math.min(w, h) / Math.max(w, h))).toFixed(4)),
          confidence: Number(conf.toFixed(4)),
        })),
      };
    }

    case 'classification': {
      if (!Array.isArray(processedOutput)) {
        throw new Error('Invalid classification output format');
      }

      if (processedOutput.length > 0 && typeof processedOutput[0] === 'object' && 'label' in processedOutput[0]) {
        const outputs = processedOutput as unknown as Array<{label: string, confidence: number}>;
        return {
          classifications: outputs.map(({ label, confidence }) => ({
            label,
            confidence: Number(confidence.toFixed(4)),
          })),
        };
      }
      
      throw new Error('Invalid classification output format');
    }

    case 'embedding': {
      if (!Array.isArray(processedOutput)) throw new Error('Invalid embedding output');
      return processedOutput;
    }

    default:
      throw new Error(`Unsupported task: ${taskType}`);
  }
} 