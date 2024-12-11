import { runTask } from './utils/run.mjs'

/**
 * @typedef {Object} DetectionConfig
 * @property {string[]} labels - Labels for detection classes
 * @property {number} [iouThreshold=0.45] - IoU threshold for NMS
 * @property {number} [confidenceThreshold=0.2] - Confidence threshold for detections
 * @property {[number, number]} [targetSize=[384, 384]] - Target size [width, height]
 */

/**
 * @typedef {Object} ClassificationConfig
 * @property {string[]} labels - Labels for classification classes
 * @property {number} [confidenceThreshold=0.2] - Confidence threshold for classification
 * @property {[number, number]} [targetSize=[384, 384]] - Target size [width, height]
 */

class EasyORT {
  constructor() {
    this.imageBuffers = []
    this.modelPath = null
    this.taskConfig = null
    this.taskType = null
    this.shouldDraw = false
  }

  /**
   * Set the model path
   * @param {string} modelPath - Path to the ONNX model
   * @returns {EasyORT}
   */
  model(modelPath) {
    this.modelPath = modelPath
    return this
  }

  /**
   * Set input data
   * @param {Buffer[]} imageBuffers - Array of image buffers
   * @returns {EasyORT}
   */
  data(imageBuffers) {
    this.imageBuffers = imageBuffers
    return this
  }

  /**
   * Run inference with specified task and configuration
   * @param {'detection' | 'classification'} taskName - Name of the task
   * @param {DetectionConfig | ClassificationConfig} config - Task configuration
   * @returns {Promise<Object>}
   */
  async run(taskName, config) {
    if (!['detection', 'classification'].includes(taskName)) {
      throw new Error('Invalid task name. Must be either "detection" or "classification"')
    }
    if (!this.imageBuffers.length) {
      throw new Error('No images provided. Call data() first.')
    }
    if (!this.modelPath) {
      throw new Error('No model path provided. Call model() first.')
    }

    this.taskType = taskName
    this.taskConfig = {
      modelPath: this.modelPath,
      labels: config.labels,
      iouThreshold: config.iouThreshold ?? 0.45,
      confidenceThreshold: config.confidenceThreshold ?? 0.2,
      targetSize: config.targetSize ?? [384, 384],
      headless: !this.shouldDraw,
    }

    return runTask(this.imageBuffers, this.taskConfig, this.taskType)
  }

  /**
   * Enable drawing results
   * @returns {EasyORT}
   */
  draw() {
    this.shouldDraw = true
    return this
  }
}

export default EasyORT
