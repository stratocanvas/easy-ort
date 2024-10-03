import { runTask } from './utils/run.mjs'

/**
 * 객체 감지를 수행하는 함수
 * @param {Buffer[]} imageBuffers 입력 이미지 버퍼 배열
 * @param {Object} options 설정 옵션
 * @returns {Promise<string>} JSON 형식의 감지 결과
 */
export async function runDetection(imageBuffers, options) {
  return runTask(imageBuffers, options, 'detection')
}

/**
 * 이미지 분류를 수행하는 함수
 * @param {Buffer[]} imageBuffers 입력 이미지 버퍼 배열
 * @param {Object} options 설정 옵션
 * @returns {Promise<string>} JSON 형식의 분류 결과
 */
export async function runClassification(imageBuffers, options) {
  return runTask(imageBuffers, options, 'classification')
}
