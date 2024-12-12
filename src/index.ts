import { InferenceSession, Tensor } from "onnxruntime-node";
import { preprocess, postprocess, formatResult } from "./utils/process";
import { drawResult } from "./utils/draw";
import type {
	TaskResult,
	TaskType,
	TaskOptions,
  ProcessedOutput,
  DetectionOutput,
  ClassificationOutput
} from "./types";

class TaskBuilder {
	private inputs: Buffer[] | string[] = [];
	private modelPath = "";
	private options: TaskOptions = {};
	private shouldDraw = false;
	private taskType: TaskType;


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
				);
				inputTensor = preprocessed.inputTensor;
				originalSizes = preprocessed.originalSizes;
			} else {
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
			});

			if (this.shouldDraw && this.taskType !== "embedding") {
				await Promise.all(
					processedOutputs.map(async (processedOutput, bat) => {
						await drawResult(
							this.inputs[bat] as Buffer,
							processedOutput as number[][] | { label: string; confidence: number }[],
							`./output/${bat + 1}.png`,
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

}
