declare module 'onnxruntime-node' {
  export class InferenceSession {
    static create(
      modelPath: string,
      options?: { enableCpuMemArena: boolean; enableMemPattern: boolean }
    ): Promise<InferenceSession>
    inputNames: string[]
    outputNames: string[]
    run(feeds: { [key: string]: Tensor }): Promise<{ [key: string]: Tensor }>
    release(): Promise<void>
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
    dispose(): void;
  }
} 