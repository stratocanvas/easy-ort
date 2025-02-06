import { InferenceSession, Tensor } from "onnxruntime-node";
import type { RuntimeProvider, RuntimeSession, RuntimeTensor } from "../types/runtime";

export class NodeRuntimeProvider implements RuntimeProvider {
  async createSession(
    modelPath: string,
    options: { enableCpuMemArena: boolean; enableMemPattern: boolean } = {
      enableCpuMemArena: true,
      enableMemPattern: true,
    }
  ): Promise<RuntimeSession> {
    return await InferenceSession.create(modelPath, options);
  }

  createTensor(type: "float32" | "int64", data: Float32Array | BigInt64Array, dims: number[]): RuntimeTensor {
    return new Tensor(type, data, dims);
  }

  async run(session: RuntimeSession, feeds: { [key: string]: RuntimeTensor }): Promise<{ [key: string]: RuntimeTensor }> {
    const nodeSession = session as InferenceSession;
    const nodeTensorFeeds: { [key: string]: Tensor } = feeds as { [key: string]: Tensor };
    return await nodeSession.run(nodeTensorFeeds);
  }

  async release(session: RuntimeSession): Promise<void> {
    await (session as InferenceSession).release();
  }
} 