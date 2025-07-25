![](./og.png)

# easy-ort &middot; [![Test](https://github.com/stratocanvas/easy-ort/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/stratocanvas/easy-ort/actions/workflows/test.yml) [![Build](https://github.com/stratocanvas/easy-ort/actions/workflows/npm-publish.yml/badge.svg)](https://github.com/stratocanvas/easy-ort/actions/workflows/npm-publish.yml) [![codecov](https://codecov.io/gh/stratocanvas/easy-ort/graph/badge.svg?token=9ODSUZ09UU)](https://codecov.io/gh/stratocanvas/easy-ort) [![version](https://img.shields.io/npm/v/%40stratocanvas%2Feasy-ort?logo=npm)](https://www.npmjs.com/package/@stratocanvas/easy-ort) [![downloads](https://img.shields.io/npm/dw/%40stratocanvas%2Feasy-ort?logo=npm)](https://www.npmjs.com/package/@stratocanvas/easy-ort)

A lightweight and intuitive wrapper for ONNX Runtime in Node.js. Supports object detection, image classification, and vision embeddings with a clean, chainable API.

## Features
- 🚀 Fluent chainable API
- 🖼️ Batch processing for images
- 📊 Result visualization
- 🎯 Built-in NMS for object detection
- 🧬 Vision embedding support

## Installation 

```bash
npm i @stratocanvas/easy-ort

# Install one of the following runtimes:
npm i onnxruntime-node   # For Node.js
# OR
npm i onnxruntime-web    # For browser
```

## Runtime Options

easy-ort supports both Node.js and Web environments through different ONNX Runtime implementations:

```javascript
// For Node.js
const nodeOrt = new EasyORT('node')

// For Web
const webOrt = new EasyORT('web')
```

Choose the appropriate runtime based on your environment:
- Use `'node'` for Node.js applications (default)
- Use `'web'` for browser applications

## Quick Examples

### Object Detection
```javascript
import EasyORT from '@stratocanvas/easy-ort'

const result = await new EasyORT('node')
  .detect(['person', 'car'])
  .in(imageBuffers)
  .using('./model.onnx')
  .withOptions({
    confidenceThreshold: 0.3,
    iouThreshold: 0.45,
    targetSize: [640, 640]
  })
  // Optional: Configure ONNX Runtime memory optimizations
  .withMemoryOptions({
    enableCpuMemArena: true,    // Enable CPU memory arena allocation
    enableMemPattern: true      // Enable memory pattern optimization
  })
  .andDraw()
  .now()

/* Output example:
[
  {
    "detections": [
      {
        "label": "person",
        "box": [120, 30, 50, 100],     // [x, y, width, height] in pixels
        "confidence": 0.92,             // 0-1 confidence score
        "squareness": 0.85             // Box aspect ratio (0-1)
      },
      {
        "label": "car",
        "box": [200, 150, 120, 80],
        "confidence": 0.88,
        "squareness": 0.75
      }
    ]
  }
]
*/
```

### Image Classification
```javascript
const result = await new EasyORT('node')
  .classify(['cat', 'dog', 'bird'])
  .in(imageBuffers)
  .using('./classifier.onnx')
  .withOptions({
    confidenceThreshold: 0.2,
    targetSize: [224, 224]
  })
  .andDraw()
  .now()

/* Output example:
[
  {
    "classifications": [
      {
        "label": "dog",
        "confidence": 0.95    // 0-1 confidence score
      },
      {
        "label": "cat",
        "confidence": 0.03
      }
    ]
  }
]
*/
```

### Image Embeddings
```javascript
const result = await new EasyORT('node')
  .createEmbeddings()
  .in(imageBuffers)
  .using('./vision_model.onnx')
  .withOptions({
    dimension: 768,
    targetSize: [384, 384]
  })
  .andNormalize()
  .now()

/* Output example:
[
  [0.15, -0.28, 0.91, ...],  // 768-dimensional vector for first image
  [0.33, 0.12, -0.67, ...]   // 768-dimensional vector for second image
]

// With .andMerge():
[[0.24, -0.08, 0.12, ...]]     // Single averaged 768-dimensional vector
*/
```

## Batch Processing

The library automatically handles batch processing for both single and multiple inputs. Here's a utility function to load multiple images:

```typescript
import fs from 'node:fs/promises'
import path from 'node:path'
import sharp from 'sharp'

async function loadImagesAsBuffers(directoryPath: string): Promise<Buffer[]> {
  const files = await fs.readdir(directoryPath);
  const imageBuffers: Buffer[] = [];

  for (const file of files) {
    if (file.match(/\.(jpg|jpeg|png|gif|webp)$/i)) {
      const filePath = path.join(directoryPath, file);
      const buffer = await sharp(filePath)
        .toBuffer();
      imageBuffers.push(buffer);
    }
  }
  return imageBuffers;
}

// Usage example
const imageBuffers = await loadImagesAsBuffers('./images')
const result = await new EasyORT()
  .detect(['person', 'car'])
  .in(imageBuffers)  // Pass single Buffer or Buffer[] for batch processing
  .using('./model.onnx')
  .withOptions({ /* ... */ })
  .now()

// Result will be an array matching the input batch size
```

## API Reference

### Task Initialization
- `.detect(labels: string[])` - Start object detection task
- `.classify(labels: string[])` - Start image classification task
- `.createEmbeddings()` - Start embedding extraction task

### Chain Methods
- `.withOptions(options)` - Set task-specific options
- `.withMemoryOptions(options)` - Set ONNX Runtime memory optimization options
- `.in(inputs)` - Provide input data (Buffer[] for images)
- `.using(modelPath)` - Specify ONNX model path
- `.andDraw()` - Enable result visualization (detection/classification only)
- `.andNormalize()` - Enable L2 normalization (embeddings only)
- `.andMerge()` - Merge embeddings (embeddings only)
- `.now()` - Execute the task

### Options

```typescript
// Detection
{
  confidenceThreshold?: number;  // Default: 0.2
  iouThreshold?: number;        // Default: 0.45
  targetSize?: [number, number]; // Default: [384, 384]
  inputShape?: 'NCHW' | 'NHWC'; // Default: 'NCHW', tensor format for input images
  sahi?: {                      // SAHI (Slicing Aided Hyper Inference)
    overlap: number;            // Default: 0.1, overlap ratio between slices
    mergeThreshold?: number;    // Threshold for merging overlapped detections
    aspectRatioThreshold?: number; // Only apply SAHI when image aspect ratio exceeds this value
  }
}

// Classification
{
  confidenceThreshold?: number;  // Default: 0.2
  targetSize?: [number, number]; // Default: [384, 384]
  inputShape?: 'NCHW' | 'NHWC'; // Default: 'NCHW', tensor format for input images
}

// Embeddings
{
  dimension?: number;           // Default: 768
  targetSize?: [number, number]; // Default: [384, 384]
  inputShape?: 'NCHW' | 'NHWC'; // Default: 'NCHW', tensor format for input images
}

// Memory Options
{
  enableCpuMemArena?: boolean;  // Default: true, enables CPU memory arena allocation
  enableMemPattern?: boolean;   // Default: true, enables memory pattern optimization
}
```

### Input Tensor Format

The `inputShape` option controls how image data is arranged in the input tensor. Use [Netron](https://netron.app/) to visualize your ONNX model and check the expected input format:

```javascript
// Check your model's input format using Netron at https://netron.app/
// Then set the appropriate inputShape option

const result = await new EasyORT('node')
  .detect(['person', 'car'])
  .withOptions({
    inputShape: 'NCHW'  // or 'NHWC' based on your model
  })
  .in(imageBuffers)
  .using('./model.onnx')
  .now()
```

## Advanced Features

### SAHI (Slicing Aided Hyper Inference)

SAHI is a technique that improves object detection performance on images with extreme aspect ratios or small objects. It works by:
1. Slicing the input image into smaller, overlapping pieces
2. Running detection on each slice
3. Merging the results back together

```javascript
const result = await new EasyORT('node')
  .detect(['person'])
  .withOptions({
    iouThreshold: 0.45,
    confidenceThreshold: 0.2,
    targetSize: [384, 384],
  })
  .withSahi({
    overlap: 0.2,              // 20% overlap between slices
    mergeThreshold: 0.5,       // Threshold for merging overlapped detections
    aspectRatioThreshold: 4.0  // Only apply SAHI when image aspect ratio > 4.0
  })
  .in(imageBuffers)
  .using('./model.onnx')
  .andDraw()                   // Optional: visualize results
  .now()
```

SAHI is particularly useful for:
- Images with extreme aspect ratios (e.g., panoramas)
- Images containing many small objects
- Satellite or aerial imagery

The `aspectRatioThreshold` option allows you to selectively apply SAHI only to images that need it:
- If image aspect ratio > threshold: Image is sliced and processed using SAHI
- If image aspect ratio ≤ threshold: Image is processed normally without slicing

## Requirements

### Models
- Single input/output nodes
- Input formats:
  - Vision: NCHW format, normalized to 0-1
- Output formats:
  - Detection: [batch_size, num_boxes, 5 + num_classes]
  - Classification: [batch_size, num_classes]
  - Embedding: [batch_size, dimension]

### System
- Node.js or Web environment
- Appropriate ONNX Runtime installed (`onnxruntime-node` or `onnxruntime-web`)
- Write access to `./output/` for visualization

## Acknowledgements
I would like to acknowledge the following open-source projects and resources that have been instrumental in the development of this project:

* **[`deepghs/imgutils`](https://github.com/deepghs/imgutils)**
 * I adopted the embedding merge (aggregation) algorithm from this project.
 * I also utilized its image preprocessing algorithms for embeddings.


## License

MIT
