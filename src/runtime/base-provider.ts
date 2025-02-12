import type { RuntimeProvider, RuntimeSession, RuntimeTensor } from "../types/runtime";
import type { InferenceSession as NodeInferenceSession, Tensor as NodeTensor } from "onnxruntime-node";
import type { InferenceSession as WebInferenceSession, Tensor as WebTensor } from "onnxruntime-web";
import { OrtSessionManager } from './session-manager';

export abstract class BaseProvider implements RuntimeProvider {
  protected sessionManager: OrtSessionManager;

  constructor() {
    this.sessionManager = OrtSessionManager.getInstance();
  }

  async createSession(modelPath: string, options: { enableCpuMemArena: boolean; enableMemPattern: boolean }): Promise<RuntimeSession> {
    await this.sessionManager.initialize(modelPath, options);
    if (!this.sessionManager.session) {
      throw new Error('Failed to initialize session');
    }
    return this.sessionManager.session;
  }

  abstract createTensor(type: "float32" | "int64", data: Float32Array | BigInt64Array, dims: number[]): RuntimeTensor;

  async run(session: RuntimeSession, feeds: { [key: string]: RuntimeTensor }): Promise<{ [key: string]: RuntimeTensor }> {
    // Implementation will be provided by concrete providers
    throw new Error('Method not implemented');
  }

  async release(session: RuntimeSession): Promise<void> {
    await this.sessionManager.release();
  }
} 