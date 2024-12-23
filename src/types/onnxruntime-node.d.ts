declare module 'onnxruntime-node' {
  export class InferenceSession {
    static create(modelPath: string): Promise<InferenceSession>
    inputNames: string[]
    outputNames: string[]
    run(feeds: { [key: string]: Tensor }): Promise<{ [key: string]: Tensor }>
  }

  export class Tensor {
    constructor(
      type: 'float32' | 'int64',
      data: Float32Array | BigInt64Array,
      dims: number[]
    )
    data: Float32Array | BigInt64Array
    dims: number[]
    type: string
  }
} 