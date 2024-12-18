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

	constructor(taskType: TaskType, inputType: "image" | "text", easyOrt: EasyORT) {
		this.taskType = taskType;
		this.inputType = inputType;
		this.easyOrt = easyOrt;
	}

	withOptions(options: TaskOptions): TaskBuilder<T> {
		this.options = { ...this.options, ...options };
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
		return await this.easyOrt.createSession(modelPath);
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
			
			let inputTensor: Float32Array | BigInt64Array;
			let originalSizes: [number, number][] = [];
			const batch = this.inputs.length;

			if (this.inputType === "image") {
				const preprocessed = await preprocess(
					this.inputs as Buffer[],
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
				inputTensor = new BigInt64Array(batch * maxLength);

				// Simple tokenization (you might want to use a proper tokenizer)
				(this.inputs as string[]).forEach((text, i) => {
					const chars = Array.from(text);
					chars.forEach((char, j) => {
						inputTensor[i * maxLength + j] = BigInt(char.charCodeAt(0));
					});
				});
			}

			const session = await this.createSession(resolvedModelPath);
			const inputName = session.inputNames[0];
			const outputName = session.outputNames[0];

			const tensor = this.easyOrt.createTensor(
				this.inputType === "image" ? "float32" : "int64",
				inputTensor,
				this.inputType === "image"
					? [batch, 3, ...(this.options.targetSize || [384, 384])]
					: [batch, inputTensor.length / batch],
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
				batch,
				shouldNormalize: this.shouldNormalize,
				shouldMerge: this.shouldMerge,
			});

			if (this.shouldDraw && this.taskType !== "embedding") {
				await Promise.all(
					processedOutputs.map(async (processedOutput, bat) => {
						const outputPath = `./output/${this.taskType}/${bat + 1}.png`;
						const dir = outputPath.substring(0, outputPath.lastIndexOf('/'));
						if (!fs.existsSync(dir)) {
							fs.mkdirSync(dir, { recursive: true });
						}
						await drawResult(
							this.inputs[bat] as Buffer,
							processedOutput as number[][] | { label: string; confidence: number }[],
							outputPath,
							{
								labels: this.options.labels || [],
								taskType: this.taskType as "detection" | "classification",
							},
						);
					}),
				);
			}

			return processedOutputs.map((output) =>
				formatResult(
					output as ProcessedOutput | number[][],
					this.options.labels || [],
					this.taskType
				) as T
			);
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

	public getRuntime() {
		return this.runtime;
	}

	constructor(runtime: 'node' | 'web' = 'node') {
		this.runtime = runtime;
		this.provider = runtime === 'node' 
			? new NodeRuntimeProvider()
			: new WebRuntimeProvider();
	}

	public async createSession(modelPath: string): Promise<RuntimeSession> {
		return await this.provider.createSession(modelPath);
	}

	public createTensor(type: "float32" | "int64", data: Float32Array | BigInt64Array, dims: number[]): RuntimeTensor {
		return this.provider.createTensor(type, data, dims);
	}

	detect(labels: string[]): TaskBuilder<DetectionResult> {
		return new TaskBuilder<DetectionResult>("detection", "image", this).withOptions({ labels });
	}

	classify(labels: string[]): TaskBuilder<ClassificationResult> {
		return new TaskBuilder<ClassificationResult>("classification", "image", this).withOptions({ labels });
	}

	createEmbeddingsFor(type: "image" | "text"): TaskBuilder<EmbeddingResult> {
		return new TaskBuilder<EmbeddingResult>("embedding", type, this);
	}

	public async run(session: RuntimeSession, feeds: FeedsType): Promise<{ [key: string]: RuntimeTensor }> {
		return await this.provider.run(session, feeds);
	}
}
