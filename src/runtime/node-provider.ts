import * as ort from "onnxruntime-node";
import type { RuntimeSession, RuntimeTensor } from "../types/runtime";
import { BaseProvider } from "./base-provider";



export class NodeRuntimeProvider extends BaseProvider {
  private async disposeTensor(tensor: ort.Tensor) {
    try {
      if (tensor && typeof tensor.dispose === 'function') {
        await tensor.dispose();
      }
    } catch (error) {
      console.error('Error disposing tensor:', error);
    }
  }

  async createSession(
    modelPath: string,
    options: { enableCpuMemArena: boolean; enableMemPattern: boolean }
  ): Promise<RuntimeSession> {
    
    // Initialize session with ONNX Runtime Node
    const sessionOptions = {
      executionProviders: ['cpu'],
      graphOptimizationLevel: 'all',
      enableCpuMemArena: false, // Disable CPU arena to prevent memory accumulation
      enableMemPattern: false,   // Disable memory pattern to force immediate cleanup
      executionMode: 'sequential'
    };
    
    this.sessionManager.session = await ort.InferenceSession.create(modelPath, sessionOptions) as RuntimeSession;
    
    return this.sessionManager.session;
  }

  createTensor(
    type: "float32" | "int64",
    data: Float32Array | BigInt64Array,
    dims: number[]
  ): RuntimeTensor {
    const tensor = new ort.Tensor(type, data, dims) as RuntimeTensor;
    return tensor;
  }

  async run(
    session: RuntimeSession,
    feeds: { [key: string]: RuntimeTensor }
  ): Promise<{ [key: string]: RuntimeTensor }> {
    
    try {
      const nodeFeeds = feeds as { [key: string]: ort.Tensor };
      const results = await (session as ort.InferenceSession).run(nodeFeeds);
      
      // Dispose input tensors immediately
      for (const tensor of Object.values(nodeFeeds)) {
        await this.disposeTensor(tensor);
      }
      
      return results as { [key: string]: RuntimeTensor };
    } catch (error) {
      // Ensure tensors are disposed even if inference fails
      for (const tensor of Object.values(feeds)) {
        await this.disposeTensor(tensor as ort.Tensor);
      }
      throw error;
    }
  }

  async release(session: RuntimeSession): Promise<void> {
    try {
      // Release session
      await (session as ort.InferenceSession).release();
      
      // Force garbage collection
      if (global.gc) {
        global.gc();
      }
      
      // Wait a bit for memory to be released
      await new Promise(resolve => setTimeout(resolve, 100));
    } catch (error) {
      console.error('Error releasing session:', error);
    }
  }
} 