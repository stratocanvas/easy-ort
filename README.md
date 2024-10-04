# easy-ort

A simple wrapper for simple ONNX inference tasks

## Installation 

```bash
npm i @stratocanvas/easy-ort
```

## Usage
### Object Detection

```javascript
import { runDetection } from '@stratocanvas/easy-ort'

const options = {
  modelPath: './path/to/model.onnx',
  labels: ['person', 'cat', 'dog', ...],
  iouThreshold: 0.45,
  confidenceThreshold: 0.2,
  targetSize: [384, 384],
  headless: false,
}

// Image should be provided as buffer
// You can batch process multiple images
const imageBuffers = [...]
const result = await runDetection(imageBuffers, options)
console.log(JSON.stringify(result))

/*result:
[
  {
    "detections": [
      {
        "label": "cat",
        "box": [x, y, w, h]
        "squareness": 0.833,
        "confidence": 0.986
      },
      {
        "label": "dog",
        "box": [x, y, w, h]
        "squareness": 0.933,
        "confidence": 0.564
      }
    ]
  }
]
*/
```

### Image Classification
```javascript
import { runClassification } from '@stratocanvas/easy-ort'

// Configuration options for the classification
const options = {
  modelPath: './path/to/model.onnx',
  labels: ['happy', 'sad', 'neutral', ...],
  confidenceThreshold: 0.2,
  targetSize: [384, 384],
  headless: false,
}

// Image should be provided as buffer
// You can batch process multiple images
const imageBuffers = [...]
const result = await runClassification(imageBuffers, options)
console.log(JSON.stringify(result))

/*result:
[
  {
    "classifications": [
      {
        "label": "happy",
        "confidence": 0.753
      },
      {
        "label": "neutral",
        "confidence": 0.215
      }
    ]
  }
]
*/

```

### Options
- `modelPath` string: Path to ONNX model file

- `labels` string[]: Detection or classification labels

- `iouThreshold` number: IoU threshold for detection inference (default: `0.45`)

- `confidenceThreshold` number: Confidence threshold (default: `0.2`)

- `targetSize` number[]: Model inference size [height, width] (default: `[384, 384]`)

- `headless` bool: Disable drawing and saving inference result (default: `false`)

## Precautions

- This library only runs in a Node.js environment (it depends on `onnxruntime-node` library)
- The model must have only one input node and one output node
- Result images are saved in the `./output/` directory by default
- Input images should be passed as buffers

## License

MIT
