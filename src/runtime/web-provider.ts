import { InferenceSession, Tensor } from "onnxruntime-web";
import type { RuntimeProvider, RuntimeSession, RuntimeTensor } from "../types/runtime";

export class WebRuntimeProvider implements RuntimeProvider {
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
    return await (session as InferenceSession).run(feeds as { [key: string]: Tensor });
  }

  async release(session: RuntimeSession): Promise<void> {
    await (session as InferenceSession).release();
  }
} 