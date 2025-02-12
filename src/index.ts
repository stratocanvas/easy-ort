import { preprocess } from "./utils/preprocessing/imagePreprocess";
import { postprocess } from "./utils/process";
import { formatResult } from "./utils/formatters/resultFormatter";
import { drawResult } from "./utils/draw";
import type {
	TaskResult,
	TaskType,
	TaskOptions,
  ProcessedOutput
} from "./types";
import fs from 'node:fs';
import path from 'node:path';
import type { RuntimeProvider, RuntimeType, RuntimeSession, RuntimeTensor } from "./types/runtime";
import { NodeRuntimeProvider } from "./runtime/node-provider";
import { WebRuntimeProvider } from "./runtime/web-provider";
import type { Tensor } from "onnxruntime-node";
import type { DetectionResult, ClassificationResult, EmbeddingResult } from "./types/results";
type FeedsType = { [key: string]: RuntimeTensor };

class TaskBuilder<T extends TaskResult> {
	private inputs: Buffer[] | string[] = [];
	private modelPath = "";
	private options: TaskOptions = {};
	private shouldDraw = false;
	private shouldNormalize = false;
	private shouldMerge = false;
	private taskType: TaskType;
	private inputType: "image" | "text";
	private easyOrt: EasyORT;
	private static MAX_BATCH_SIZE = 32;
	private memoryOptions: { enableCpuMemArena: boolean; enableMemPattern: boolean } = {
		enableCpuMemArena: true,
		enableMemPattern: true,
	};

	
	constructor(taskType: TaskType, inputType: "image" | "text", easyOrt: EasyORT) {
		this.taskType = taskType;
		this.inputType = inputType;
		this.easyOrt = easyOrt;
	}

	withOptions(options: TaskOptions): TaskBuilder<T> {
		this.options = { ...this.options, ...options };
		return this;
	}

	withMemoryOptions(options: { enableCpuMemArena?: boolean; enableMemPattern?: boolean }): TaskBuilder<T> {
		this.memoryOptions = {
			...this.memoryOptions,
			...options
		};
		return this;
	}

	in(inputs: Buffer[] | string[]): TaskBuilder<T> {
		this.inputs = inputs;
		return this;
	}

	using(modelPath: string): TaskBuilder<T> {
		this.modelPath = modelPath;
		return this;
	}

	andDraw(): TaskBuilder<T> {
		this.shouldDraw = true;
		return this;
	}

	andNormalize(): TaskBuilder<T> {
		this.shouldNormalize = true;
		return this;
	}

	andMerge(): TaskBuilder<T> {
		this.shouldMerge = true;
		return this;
	}

	private resolvePath(modelPath: string): string {
		if (path.isAbsolute(modelPath)) {
			return modelPath;
		}

		const commonDirs = [
			'public',
			'static',
			'assets',
			'src/assets',
			'app/assets',
			'static/assets',
			''  // project root as fallback
		];

		for (const dir of commonDirs) {
			const tryPath = dir 
				? path.join(process.cwd(), dir, modelPath)
				: path.join(process.cwd(), modelPath);
			
			if (fs.existsSync(tryPath)) {
				return tryPath;
			}
		}

		return modelPath;
	}

	private async createSession(modelPath: string) {
		return await this.easyOrt.createSession(modelPath, this.memoryOptions);
	}

	private async processBatch(
		inputs: Buffer[] | string[],
		session: RuntimeSession,
		startIdx: number,
		batchSize: number
	): Promise<T[]> {
		let tensor: RuntimeTensor | undefined;
		try {
			const batchInputs = inputs.slice(startIdx, startIdx + batchSize);
			let inputTensor: Float32Array | BigInt64Array;
			let originalSizes: [number, number][] = [];

			if (this.inputType === "image") {
				const preprocessed = await preprocess(
					batchInputs as Buffer[],
					this.options.targetSize || [384, 384],
					this.taskType
				);
				inputTensor = preprocessed.inputTensor;
				originalSizes = preprocessed.originalSizes;
			} else {
				// Text input processing
				const maxLength = Math.max(
					...(this.inputs as string[]).map((text) => text.length),
				);
				inputTensor = new BigInt64Array(batchSize * maxLength);

				// Simple tokenization (you might want to use a proper tokenizer)
				(this.inputs as string[]).forEach((text, i) => {
					const chars = Array.from(text);
					chars.forEach((char, j) => {
						inputTensor[i * maxLength + j] = BigInt(char.charCodeAt(0));
					});
				});
			}

			const inputName = session.inputNames[0];
			const outputName = session.outputNames[0];

			tensor = this.easyOrt.createTensor(
				this.inputType === "image" ? "float32" : "int64",
				inputTensor,
				this.inputType === "image"
					? [batchSize, 3, ...(this.options.targetSize || [384, 384])]
					: [batchSize, inputTensor.length / batchSize]
			);

			const feeds = { [inputName]: tensor };
			const results = await this.easyOrt.run(session, feeds);

			const output = results[outputName];

			const processedOutputs = postprocess(output as Tensor, {
				confidenceThreshold: this.options.confidenceThreshold || 0.2,
				iouThreshold: this.options.iouThreshold || 0.45,
				targetSize: this.options.targetSize || [384, 384],
				originalSizes,
				labels: this.options.labels || [],
				taskType: this.taskType,
				batch: batchSize,
				shouldNormalize: this.shouldNormalize,
				shouldMerge: this.shouldMerge,
			});

			// Dispose all tensors in results
			for (const resultTensor of Object.values(results)) {
				if ('dispose' in resultTensor && typeof (resultTensor as { dispose: () => void }).dispose === 'function') {
					(resultTensor as { dispose: () => void }).dispose();
				}
			}

			if (this.shouldDraw && this.taskType !== "embedding") {
				await Promise.all(
					processedOutputs.map(async (processedOutput, idx) => {
						const outputPath = `./output/${this.taskType}/${startIdx + idx + 1}.png`;
						const dir = outputPath.substring(0, outputPath.lastIndexOf('/'));
						if (!fs.existsSync(dir)) {
							fs.mkdirSync(dir, { recursive: true });
						}
						await drawResult(
							batchInputs[idx] as Buffer,
							processedOutput as number[][] | { label: string; confidence: number }[],
							outputPath,
							{
								labels: this.options.labels || [],
								taskType: this.taskType as "detection" | "classification",
							}
						);
					})
				);
			}

			return processedOutputs.map((output) =>
				formatResult(
					output as ProcessedOutput | number[][],
					this.options.labels || [],
					this.taskType
				) as T
			);
		} finally {
			// Ensure tensor is disposed
			if (tensor && 'dispose' in tensor && typeof (tensor as { dispose: () => void }).dispose === 'function') {
				(tensor as { dispose: () => void }).dispose();
			}
		}
	}

	async now(): Promise<T[]> {
		if (!this.inputs.length) {
			throw new Error("No inputs provided. Call in() first.");
		}
		if (!this.modelPath) {
			throw new Error("No model path provided. Call using() first.");
		}

		try {
			const resolvedModelPath = this.resolvePath(this.modelPath);
			const session = await this.createSession(resolvedModelPath);
			
			const totalInputs = this.inputs.length;
			const results: T[] = [];

			try {
				// Process in optimal batch sizes
				for (let i = 0; i < totalInputs; i += TaskBuilder.MAX_BATCH_SIZE) {
					const batchSize = Math.min(TaskBuilder.MAX_BATCH_SIZE, totalInputs - i);
					const batchResults = await this.processBatch(this.inputs, session, i, batchSize);
					results.push(...batchResults);
				}
				return results;
			} finally {
				// Session을 캐시하지 않는 경우에만 해제
				if (!this.easyOrt.sessionCache.has(resolvedModelPath)) {
					await this.easyOrt.releaseSession(session);
				}
			}
		} catch (error: unknown) {
			if (error instanceof Error) {
				throw new Error(`Failed to load model: ${error.message}`);
			}
			throw new Error("An unknown error occurred while loading the model");
		}
	}
}

export default class EasyORT {
	private provider: RuntimeProvider;
	private runtime: 'node' | 'web';
	public sessionCache = new Map<string, RuntimeSession>();
	private static readonly DEFAULT_MAX_SESSIONS = 5;
	private static readonly DEFAULT_MAX_MEMORY_MB = 1024; // 1GB
	private maxSessions: number;
	private maxMemoryMB: number;
	private currentMemoryUsageMB = 0;

	public getRuntime() {
		return this.runtime;
	}

	constructor(
		runtime: 'node' | 'web' = 'node',
		options?: {
			maxSessions?: number;
			maxMemoryMB?: number;
		}
	) {
		this.runtime = runtime;
		this.provider = runtime === 'node' 
			? new NodeRuntimeProvider()
			: new WebRuntimeProvider();
		this.maxSessions = options?.maxSessions || EasyORT.DEFAULT_MAX_SESSIONS;
		this.maxMemoryMB = options?.maxMemoryMB || EasyORT.DEFAULT_MAX_MEMORY_MB;
	}

	private async removeOldestSession(): Promise<void> {
		const oldestSession = this.sessionCache.values().next().value;
		if (oldestSession) {
			await this.releaseSession(oldestSession);
		}
	}

	private async ensureResourceAvailability(modelSizeInBytes: number): Promise<void> {
		// 세션 수 제한 확인
		if (this.sessionCache.size >= this.maxSessions) {
			await this.removeOldestSession();
		}

		// 메모리 사용량 제한 확인
		const modelSizeInMB = modelSizeInBytes / (1024 * 1024);
		if (this.currentMemoryUsageMB + modelSizeInMB > this.maxMemoryMB) {
			// 메모리 확보를 위해 가장 오래된 세션 제거
			while (this.currentMemoryUsageMB + modelSizeInMB > this.maxMemoryMB && this.sessionCache.size > 0) {
				await this.removeOldestSession();
			}
		}
	}

	public async createSession(
		modelPath: string,
		options: { enableCpuMemArena: boolean; enableMemPattern: boolean } = {
			enableCpuMemArena: true,
			enableMemPattern: true
		}
	): Promise<RuntimeSession> {
		if (this.sessionCache.has(modelPath)) {
			return this.sessionCache.get(modelPath) as RuntimeSession;
		}

		// 모델 파일 크기 확인
		const stats = await fs.promises.stat(modelPath);
		await this.ensureResourceAvailability(stats.size);

		const session = await this.provider.createSession(modelPath, options);
		this.sessionCache.set(modelPath, session);
		this.currentMemoryUsageMB += stats.size / (1024 * 1024);
		return session;
	}

	public createTensor(type: "float32" | "int64", data: Float32Array | BigInt64Array, dims: number[]): RuntimeTensor {
		return this.provider.createTensor(type, data, dims);
	}

	public async releaseSession(session: RuntimeSession): Promise<void> {
		await this.provider.release(session);
		// Find and remove from cache
		for (const [path, cachedSession] of this.sessionCache.entries()) {
			if (cachedSession === session) {
				this.sessionCache.delete(path);
				// 메모리 사용량 업데이트
				try {
					const stats = await fs.promises.stat(path);
					this.currentMemoryUsageMB -= stats.size / (1024 * 1024);
				} catch (error) {
					console.warn(`Failed to update memory usage for ${path}`);
				}
				break;
			}
		}
	}

	public getSessionStats(): {
		currentSessions: number;
		maxSessions: number;
		currentMemoryMB: number;
		maxMemoryMB: number;
	} {
		return {
			currentSessions: this.sessionCache.size,
			maxSessions: this.maxSessions,
			currentMemoryMB: Math.round(this.currentMemoryUsageMB),
			maxMemoryMB: this.maxMemoryMB
		};
	}

	public async releaseAllSessions(): Promise<void> {
		const releasePromises: Promise<void>[] = [];
		for (const [path, session] of this.sessionCache.entries()) {
			releasePromises.push(this.provider.release(session));
		}
		await Promise.all(releasePromises);
		this.sessionCache.clear();
		this.currentMemoryUsageMB = 0;
	}

	detect(labels: string[]): TaskBuilder<DetectionResult> {
		return new TaskBuilder<DetectionResult>("detection", "image", this).withOptions({ labels });
	}

	classify(labels: string[]): TaskBuilder<ClassificationResult> {
		return new TaskBuilder<ClassificationResult>("classification", "image", this).withOptions({ labels });
	}

	createEmbeddingsFor(type: "image" | "text"): TaskBuilder<number[]> {
		return new TaskBuilder<number[]>("embedding", type, this);
	}

	public async run(session: RuntimeSession, feeds: FeedsType): Promise<{ [key: string]: RuntimeTensor }> {
		return await this.provider.run(session, feeds);
	}
}
