import type { InferenceSession as NodeSession, Tensor as NodeTensor } from "onnxruntime-node";
import type { InferenceSession as WebSession, Tensor as WebTensor } from "onnxruntime-web";

export type RuntimeType = "node" | "web";
export type RuntimeTensor = NodeTensor | WebTensor;
export type RuntimeSession = NodeSession | WebSession;
export type FeedsType = { [key: string]: RuntimeTensor };

export interface RuntimeProvider {
  createSession(
    modelPath: string,
    options?: { enableCpuMemArena: boolean; enableMemPattern: boolean }
  ): Promise<RuntimeSession>;
  createTensor(
    type: "float32" | "int64",
    data: Float32Array | BigInt64Array,
    dims: number[]
  ): RuntimeTensor;
  run(session: RuntimeSession, feeds: { [key: string]: RuntimeTensor }): Promise<{ [key: string]: RuntimeTensor }>;
  release(session: RuntimeSession): Promise<void>;
} 