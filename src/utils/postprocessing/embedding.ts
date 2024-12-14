import type { Tensor } from 'onnxruntime-node';

function normalizeEmbedding(embedding: number[]): number[] {
  const norm = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
  return embedding.map(val => val / norm);
}

export function processEmbeddingOutput(
  output: Tensor,
  batch: number,
  shouldNormalize = false,
  shouldMerge = false,
) {
  const outputData = output.data as Float32Array;
  const [, dimension] = output.dims;
  const results: number[][] = [];

  for (let b = 0; b < batch; b++) {
    const slicedData = outputData.slice(
      b * dimension,
      (b + 1) * dimension,
    );
    let embedding = Array.from(slicedData);
    
    if (shouldNormalize) {
      embedding = normalizeEmbedding(embedding);
    }
    
    results.push(embedding);
  }
  
  if (shouldMerge && batch > 1) {
    // Calculate L2 norms of original embeddings
    const originalNorms = results.map(emb => 
      Math.sqrt(emb.reduce((sum, val) => sum + val * val, 0))
    );
    
    // L2 normalize each embedding
    const normalizedEmbs = results.map(normalizeEmbedding);
    
    // Average the normalized embeddings
    const mergedEmbedding = new Array(results[0].length).fill(0);
    for (const emb of normalizedEmbs) {
      for (let i = 0; i < emb.length; i++) {
        mergedEmbedding[i] += emb[i] / batch;
      }
    }
    
    // L2 normalize the merged embedding
    const normalizedMerged = normalizeEmbedding(mergedEmbedding);
    
    // Scale by mean of original norms
    const meanNorm = originalNorms.reduce((sum, norm) => sum + norm, 0) / batch;
    const finalEmbedding = normalizedMerged.map(val => val * meanNorm);
    
    return [finalEmbedding];
  }

  return results;
} 