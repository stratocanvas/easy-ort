import { InferenceSession, Tensor, env } from "onnxruntime-web";
import type { RuntimeSession, RuntimeTensor } from "../types/runtime";
import { BaseProvider } from "./base-provider";

export class WebRuntimeProvider extends BaseProvider {
  async createSession(
    modelPath: string,
    options: { enableCpuMemArena: boolean; enableMemPattern: boolean }
  ): Promise<RuntimeSession> {
    const session = await super.createSession(modelPath, options);
    
    // Initialize session with ONNX Runtime Web
    env.wasm.numThreads = 1;
    env.wasm.simd = true;
    
    this.sessionManager.session = await InferenceSession.create(modelPath, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
      enableCpuMemArena: options.enableCpuMemArena,
      enableMemPattern: options.enableMemPattern
    }) as RuntimeSession;
    
    return this.sessionManager.session;
  }

  createTensor(
    type: "float32" | "int64",
    data: Float32Array | BigInt64Array,
    dims: number[]
  ): RuntimeTensor {
    return new Tensor(type, data, dims) as RuntimeTensor;
  }

  async run(
    session: RuntimeSession,
    feeds: { [key: string]: RuntimeTensor }
  ): Promise<{ [key: string]: RuntimeTensor }> {
    const webFeeds = feeds as { [key: string]: Tensor };
    const results = await (session as InferenceSession).run(webFeeds);
    return results as { [key: string]: RuntimeTensor };
  }

  async release(session: RuntimeSession): Promise<void> {
    await super.release(session);
    // Additional cleanup if needed
  }
} 