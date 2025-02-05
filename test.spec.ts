import { describe, it, expect, beforeAll, afterEach, afterAll } from "vitest";
import axios from "axios";
import fs from "node:fs/promises";
import path from "node:path";
import EasyORT from "./src/index";
import { existsSync } from "node:fs";
import dotenv from "dotenv";
dotenv.config();

const TEST_TIMEOUT = 60000; // 60초
async function downloadImage(url: string): Promise<Buffer> {
	const response = await axios.get(url, { responseType: "arraybuffer" });
	return Buffer.from(response.data);
}

async function downloadModel(url: string, fileName: string): Promise<string> {
	const modelsDir = path.join(process.cwd(), "models");
	const filePath = path.join(modelsDir, fileName);

	try {
		await fs.access(modelsDir);
	} catch {
		await fs.mkdir(modelsDir);
	}

	try {
		await fs.access(filePath);
		console.log(`Model ${fileName} already exists`);
	} catch {
		console.log(`Downloading model ${fileName}...`);
		const response = await axios.get(url, { responseType: "arraybuffer" });
		await fs.writeFile(filePath, Buffer.from(response.data));
		console.log(`Model ${fileName} downloaded successfully`);
	}

	return filePath;
}

async function ensureOutputDir() {
	const outputDir = path.join(process.cwd(), "output");
	try {
		await fs.access(outputDir);
	} catch {
		await fs.mkdir(outputDir, { recursive: true });
	}
	return outputDir;
}

describe(
	"EasyORT Tests",
	() => {
		let model: EasyORT;
		const testImages = process.env.TEST_IMAGES?.split(",") ?? [];

		beforeAll(async () => {
			model = new EasyORT('node');
		});

		afterEach(async () => {
			// 각 테스트 후 세션 정리
			await model.releaseAllSessions();
		});

		describe(
			"Object Detection",
			() => {
				const MODEL_URL = process.env.DETECTION_MODEL_URL ?? "";
				let modelPath: string;
				let outputDir: string;

				beforeAll(async () => {
					modelPath = await downloadModel(MODEL_URL, "detection.onnx");
					outputDir = await ensureOutputDir();
				});

				it("should download detection model", async () => {
					expect(modelPath).toBeDefined();
				});

				it("should detect objects in a single image", async () => {
					const imageBuffer = await downloadImage(testImages[0]);

					const result = await model
						.detect(["person"])
						.in([imageBuffer])
						.using(modelPath)
						.withOptions({
							iouThreshold: 0.45,
							confidenceThreshold: 0.2,
							targetSize: [384, 384],
						})
						.now();

					expect(result).toHaveLength(1);
					expect(result[0].detections).toBeDefined();
				});

				it("should detect objects and save drawn results", async () => {
					const imageBuffers = await Promise.all(
						testImages.map((url) => downloadImage(url)),
					);

					const result = await model
						.detect(["person"])
						.in(imageBuffers)
						.using(modelPath)
						.withOptions({
							iouThreshold: 0.45,
							confidenceThreshold: 0.2,
							targetSize: [384, 384],
						})
						.andDraw()
						.now();

					// 결과 검증
					expect(result).toHaveLength(imageBuffers.length);

					// 파일 생성 확인 전 약간의 지연
					await new Promise((resolve) => setTimeout(resolve, 1000));

					// 파일 생성 확인
					for (let i = 0; i < imageBuffers.length; i++) {
						const outputPath = path.join(outputDir, 'detection', `${i + 1}.png`);
						const exists = existsSync(outputPath);
						expect(exists, `File ${outputPath} should exist`).toBe(true);
					}
				});

				it("should detect objects in images", async () => {
					const imageBuffers = await Promise.all(
						testImages.map((url) => downloadImage(url)),
					);

					const result = await model
						.detect(["person"])
						.withOptions({
							iouThreshold: 0.45,
							confidenceThreshold: 0.2,
							targetSize: [384, 384],
						})
						.in(imageBuffers)
						.using(modelPath)
						.andDraw()
						.now();

					expect(result).toBeDefined();
					expect(Array.isArray(result)).toBe(true);

					for (const detection of result) {
						expect(detection).toHaveProperty("detections");
						expect(Array.isArray(detection.detections)).toBe(true);

						for (const item of detection.detections) {
							expect(item).toHaveProperty("label");
							expect(item).toHaveProperty("box");
							expect(item).toHaveProperty("confidence");
							expect(item).toHaveProperty("squareness");

							expect(Array.isArray(item.box)).toBe(true);
							expect(item.box).toHaveLength(4);
							expect(item.confidence).toBeGreaterThanOrEqual(0);
							expect(item.confidence).toBeLessThanOrEqual(1);
							expect(item.squareness).toBeGreaterThanOrEqual(0);
							expect(item.squareness).toBeLessThanOrEqual(1);
						}
					}
				});
			},
			TEST_TIMEOUT,
		);

		describe(
			"Image Classification",
			() => {
				const MODEL_URL = process.env.CLASSIFICATION_MODEL_URL ?? "";
				let modelPath: string;
				let outputDir: string;

				beforeAll(async () => {
					modelPath = await downloadModel(MODEL_URL, "classification.onnx");
					outputDir = await ensureOutputDir(); 
				});

				it("should download classification model", async () => {
					expect(modelPath).toBeDefined();
				});

				it("should classify a single image without drawing", async () => {
					const imageBuffer = await downloadImage(testImages[0]); 

					const result = await model
						.classify(["general", "sensitive", "questionable", "explicit"])
						.in([imageBuffer])
						.using(modelPath)
						.withOptions({
							confidenceThreshold: 0.3,
							iouThreshold: 0.5,
							targetSize: [384, 384],
						})
						.now();

					expect(result).toHaveLength(1);
					const classification = result[0];
					expect(classification.classifications).toBeDefined();
					expect(classification.classifications.length).toBeGreaterThan(0);
				});

				it("should classify multiple images and save drawn results", async () => {
					const imageBuffers = await Promise.all(
						testImages.map((url) => downloadImage(url)),
					);

					const result = await model
						.classify(["general", "sensitive", "questionable", "explicit"])
						.in(imageBuffers)
						.using(modelPath)
						.withOptions({
							confidenceThreshold: 0.2,
							targetSize: [384, 384],
						})
						.andDraw()
						.now();

					// 결과 검증
					expect(result).toHaveLength(imageBuffers.length);

					// 파일 생성 확인 전 약간의 지연
					await new Promise((resolve) => setTimeout(resolve, 1000));

					// 파일 생성 확인
					for (let i = 0; i < imageBuffers.length; i++) {
						const outputPath = path.join(outputDir, 'classification', `${i + 1}.png`);
						const exists = existsSync(outputPath);
						expect(exists, `File ${outputPath} should exist`).toBe(true);
					}
				});

				it("should classify images", async () => {
					const imageBuffers = await Promise.all(
						testImages.map((url) => downloadImage(url)),
					);

					const result = await model
						.classify(["general", "sensitive", "questionable", "explicit"])
						.in(imageBuffers)
						.using(modelPath)
						.withOptions({
							confidenceThreshold: 0.2,
							targetSize: [384, 384],
						})
						.now();

					expect(result).toBeDefined();
					expect(Array.isArray(result)).toBe(true);

					for (const classification of result) {
						expect(classification).toHaveProperty("classifications");
						expect(Array.isArray(classification.classifications)).toBe(true);

						for (const item of classification.classifications) {
							expect(item).toHaveProperty("label");
							expect(item).toHaveProperty("confidence");
							expect(item.confidence).toBeGreaterThanOrEqual(0);
							expect(item.confidence).toBeLessThanOrEqual(1);
						}
					}
				});
			},
			TEST_TIMEOUT,
		);

		describe("Image Embedding", () => {
			const MODEL_URL = process.env.EMBEDDING_MODEL_URL ?? "";
			let modelPath: string;

			beforeAll(async () => {
				modelPath = await downloadModel(MODEL_URL, "embedding.onnx");
			});

			const validateEmbedding = (embedding: number[]) => {
				expect(Array.isArray(embedding)).toBe(true);
				expect(embedding).toHaveLength(768);
				for (const value of embedding) {
					expect(typeof value).toBe("number");
					expect(Number.isFinite(value)).toBe(true);
				}
			};

			it("should create embeddings for images", async () => {
				const imageBuffers = await Promise.all(
					testImages.map((url) => downloadImage(url))
				);

				const result = await model
					.createEmbeddingsFor("image")
					.withOptions({
						dimension: 768,
						targetSize: [384, 384],
					})
					.in(imageBuffers)
					.using(modelPath)
					.andNormalize()
					.now();

				expect(result).toBeDefined();
				expect(Array.isArray(result)).toBe(true);
				
				// 각 이미지의 임베딩 검증
				for (const embedding of result) {
					expect(Array.isArray(embedding)).toBe(true);
					expect(embedding).toHaveLength(768);
					for (const value of embedding) {
						expect(typeof value).toBe("number");
					}
				}
			});

			it("should create merged embeddings for images", async () => {
				const imageBuffers = await Promise.all(
					testImages.map((url) => downloadImage(url)),
				);

				const result = await model
					.createEmbeddingsFor("image")
					.in(imageBuffers)
					.using(modelPath)
					.withOptions({
							dimension: 768,
							targetSize: [384, 384],
						})
						.andNormalize()
						.andMerge()
						.now();

				expect(result).toBeDefined();
				expect(Array.isArray(result)).toBe(true);
				expect(result).toHaveLength(1);
				for (const embedding of result) {
					expect(Array.isArray(embedding)).toBe(true);
					expect(embedding).toHaveLength(768);
					for (const value of embedding) {
						expect(typeof value).toBe("number");
					}
				}
			});

			it("should handle invalid images gracefully", async () => {
				const invalidBuffer = Buffer.from("invalid data");

				await expect(
					model
						.createEmbeddingsFor("image")
						.withOptions({
							dimension: 768,
							targetSize: [384, 384],
						})
						.in([invalidBuffer])
						.using(modelPath)
						.andNormalize()
						.now(),
				).rejects.toThrow();
			});

			it("should process large batch of images", async () => {
				const largeImageSet = Array(10).fill(testImages[0]);
				const imageBuffers = await Promise.all(
					largeImageSet.map((url) => downloadImage(url)),
				);

				const startTime = Date.now();
				const result = await model
					.createEmbeddingsFor("image")
					.in(imageBuffers)
					.using(modelPath)
					.withOptions({
						dimension: 768,
						targetSize: [384, 384],
					})
					.andNormalize()
					.now();

				const processingTime = Date.now() - startTime;
				expect(processingTime).toBeLessThan(30000); // 30초 이내 처리
				expect(result).toHaveLength(10);
			});
		});

		describe("Session Management", () => {
			const MODEL_URL = process.env.DETECTION_MODEL_URL ?? "";
			let modelPath: string;

			beforeAll(async () => {
				modelPath = await downloadModel(MODEL_URL, "detection.onnx");
			});

			it("should reuse cached session for the same model", async () => {
				const session1 = await model.createSession(modelPath);
				const session2 = await model.createSession(modelPath);
				
				expect(session1).toBe(session2);
			});

			it("should properly release individual sessions", async () => {
				const session = await model.createSession(modelPath);
				await model.releaseSession(session);

				// 새로운 세션 생성 시 다른 인스턴스여야 함
				const newSession = await model.createSession(modelPath);
				expect(newSession).not.toBe(session);
			});

			it("should handle multiple models and release all sessions", async () => {
				const detectionSession = await model.createSession(modelPath);
				
				const classificationModelPath = await downloadModel(
					process.env.CLASSIFICATION_MODEL_URL ?? "",
					"classification.onnx"
				);
				const classificationSession = await model.createSession(classificationModelPath);

				expect(detectionSession).not.toBe(classificationSession);

				await model.releaseAllSessions();

				// 새로운 세션들은 이전 세션들과 달라야 함
				const newDetectionSession = await model.createSession(modelPath);
				const newClassificationSession = await model.createSession(classificationModelPath);

				expect(newDetectionSession).not.toBe(detectionSession);
				expect(newClassificationSession).not.toBe(classificationSession);
			});

			it("should handle session cleanup after task completion", async () => {
				const imageBuffer = await downloadImage(testImages[0]);
				
				// 첫 번째 실행
				await model.detect(["person"])
					.in([imageBuffer])
					.using(modelPath)
					.now();

				// 동일한 모델로 두 번째 실행 - 세션이 재사용되어야 함
				await model.detect(["person"])
					.in([imageBuffer])
					.using(modelPath)
					.now();

				// 세션 해제
				await model.releaseAllSessions();

				// 세션 해제 후 새로운 실행 - 새로운 세션이 생성되어야 함
				await model.detect(["person"])
					.in([imageBuffer])
					.using(modelPath)
					.now();
			});
		});
	},
	TEST_TIMEOUT,
);
