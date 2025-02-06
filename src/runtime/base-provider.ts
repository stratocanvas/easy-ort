import type { RuntimeProvider, RuntimeSession, RuntimeTensor } from "../types/runtime";
import type { InferenceSession as NodeInferenceSession, Tensor as NodeTensor } from "onnxruntime-node";
import type { InferenceSession as WebInferenceSession, Tensor as WebTensor } from "onnxruntime-web";

export abstract class BaseRuntimeProvider implements RuntimeProvider {
  abstract createSession(
    modelPath: string,
    options?: { enableCpuMemArena: boolean; enableMemPattern: boolean }
  ): Promise<NodeInferenceSession | WebInferenceSession>;
  abstract createTensor(type: "float32" | "int64", data: Float32Array | BigInt64Array, dims: number[]): NodeTensor | WebTensor;
  abstract run(session: RuntimeSession, feeds: { [key: string]: RuntimeTensor }): Promise<{ [key: string]: RuntimeTensor }>;
  abstract release(session: RuntimeSession): Promise<void>;
} 