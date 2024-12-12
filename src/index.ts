import { InferenceSession, Tensor } from "onnxruntime-node";
import { preprocess, postprocess, formatResult } from "./utils/process";
import { drawResult } from "./utils/draw";
import type {
	TaskResult,
	TaskType,
	TaskOptions,
  ProcessedOutput
} from "./types";
import fs from 'node:fs';

class TaskBuilder {
	private inputs: Buffer[] | string[] = [];
	private modelPath = "";
	private options: TaskOptions = {};
	private shouldDraw = false;
	private shouldNormalize = false;
	private shouldMerge = false;
	private taskType: TaskType;
	private inputType: "image" | "text";

	constructor(taskType: TaskType, inputType: "image" | "text") {
		this.taskType = taskType;
		this.inputType = inputType;
	}

	withOptions(options: TaskOptions) {
		this.options = { ...this.options, ...options };
		return this;
	}

	in(inputs: Buffer[] | string[]) {
		this.inputs = inputs;
		return this;
	}

	using(modelPath: string) {
		this.modelPath = modelPath;
		return this;
	}

	andDraw() {
		this.shouldDraw = true;
		return this;
	}

	andNormalize() {
		this.shouldNormalize = true;
		return this;
	}

	andMerge() {
		this.shouldMerge = true;
		return this;
	}

	async now(): Promise<TaskResult[]> {
		if (!this.inputs.length) {
			throw new Error("No inputs provided. Call in() first.");
		}
		if (!this.modelPath) {
			throw new Error("No model path provided. Call using() first.");
		}

		try {
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

			const session = await InferenceSession.create(this.modelPath);
			const inputName = session.inputNames[0];
			const outputName = session.outputNames[0];

			const feeds = {
				[inputName]: new Tensor(
					this.inputType === "image" ? "float32" : "int64",
					inputTensor,
					this.inputType === "image"
						? [batch, 3, ...(this.options.targetSize || [384, 384])]
						: [batch, inputTensor.length / batch],
				),
			};

			const results = await session.run(feeds);
			const output = results[outputName];

			const processedOutputs = postprocess(output, {
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
					this.taskType,
				),
			);
		} catch (error: unknown) {
			if (error instanceof Error) {
				throw new Error(error.message);
			}
			throw new Error("An unknown error occurred");
		}
	}
}

export default class EasyORT {
	detect(labels: string[]) {
		return new TaskBuilder("detection", "image").withOptions({ labels });
	}

	classify(labels: string[]) {
		return new TaskBuilder("classification", "image").withOptions({ labels });
	}

	createEmbeddingsFor(type: "image" | "text") {
		return new TaskBuilder("embedding", type);
	}
}
