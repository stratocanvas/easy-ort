import type { SessionManager, SessionOptions } from '../types';
import type { RuntimeSession } from '../types/runtime';

export class OrtSessionManager implements SessionManager {
  session: RuntimeSession | null = null;
  env: RuntimeSession | null = null;
  private static instance: OrtSessionManager;
  private static readonly DEFAULT_OPTIONS: SessionOptions = {
    enableCpuMemArena: true,
    enableMemPattern: true
  };

  private constructor() {}

  static getInstance(): OrtSessionManager {
    if (!OrtSessionManager.instance) {
      OrtSessionManager.instance = new OrtSessionManager();
    }
    return OrtSessionManager.instance;
  }

  async initialize(modelPath: string, options: SessionOptions = OrtSessionManager.DEFAULT_OPTIONS): Promise<void> {
    if (this.session) {
      return; // Session already initialized
    }
  }

  async release(): Promise<void> {
    try {
      if (this.session) {
        this.session = null;
      }
      if (this.env) {
        this.env = null;
      }
    } catch (error) {
      console.error('Error releasing session:', error);
    }
  }
} 