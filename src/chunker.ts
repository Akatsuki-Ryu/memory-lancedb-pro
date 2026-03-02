/**
 * Long Context Chunking System
 * Splits documents that exceed embedding model context limits into manageable chunks.
 * Uses semantic-aware chunking with configurable overlap to preserve context.
 */

// ============================================================================
// Types & Constants
// ============================================================================

export interface ChunkResult {
  chunks: string[];
  metadatas: Array<{ startIndex: number; endIndex: number; length: number }>;
  totalOriginalLength: number;
  chunkCount: number;
}

export interface ChunkerConfig {
  /** Maximum characters per chunk (default: 4000 for most models) */
  maxChunkSize: number;
  /** Overlap between chunks in characters (default: 200) */
  overlapSize: number;
  /** Minimum chunk size below which we don't split further (default: 100) */
  minChunkSize: number;
  /** Attempt to split on sentence boundaries for better semantic coherence (default: true) */
  semanticSplit: boolean;
  /** Max lines per chunk before forced split (default: 50) */
  maxLinesPerChunk: number;
}

// Common embedding context limits
export const EMBEDDING_CONTEXT_LIMITS: Record<string, number> = {
  "jina-embeddings-v5-text-small": 8192,
  "jina-embeddings-v5-text-nano": 8192,
  "text-embedding-3-small": 8192,
  "text-embedding-3-large": 8192,
  "text-embedding-004": 8192,
  "gemini-embedding-001": 2048,
  "nomic-embed-text": 8192,
  "all-MiniLM-L6-v2": 512,
  "all-mpnet-base-v2": 512,
};

// Default configuration
export const DEFAULT_CHUNKER_CONFIG: ChunkerConfig = {
  maxChunkSize: 4000,
  overlapSize: 200,
  minChunkSize: 200,
  semanticSplit: true,
  maxLinesPerChunk: 50,
};

const SENTENCE_ENDING_PATTERNS = /[.!?。！？。、。][ \t\n\r]*/g;
const LINE_BREAK_PATTERN = /\r\n|\n|\r/;

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Calculate safe chunk size considering overlap
 */
function calculateChunkStep(config: ChunkerConfig): number {
  return config.maxChunkSize - config.overlapSize;
}

/**
 * Find best split point that respects semantic boundaries
 */
function findBestSplitPoint(text: string, maxPos: number): number {
  // Try to split at sentence boundaries first
  const sentencePattern = new RegExp(SENTENCE_ENDING_PATTERNS.source);
  let lastMatch: RegExpExecArray | null;
  let bestSplit = 0;

  while ((lastMatch = sentencePattern.exec(text.slice(0, maxPos))) !== null) {
    bestSplit = Math.min(lastMatch.index + lastMatch[0].length, maxPos);
  }

  if (bestSplit === 0) {
    // Fall back to word boundaries
    const wordPattern = /\s+$/;
    const match = text.slice(0, maxPos).match(wordPattern);
    if (match) {
      bestSplit = Math.min(maxPos - match[0].length, maxPos);
    } else {
      bestSplit = maxPos;
    }
  }

  return bestSplit;
}

// ============================================================================
// Chunking Core
// ============================================================================

/**
 * Split a document into chunks respecting semantic boundaries
 */
export function chunkDocument(
  text: string,
  config: ChunkerConfig = DEFAULT_CHUNKER_CONFIG
): ChunkResult {
  if (!text || text.trim().length === 0) {
    return { chunks: [], metadatas: [], totalOriginalLength: 0, chunkCount: 0 };
  }

  const totalOriginalLength = text.length;
  const chunks: string[] = [];
  const metadatas: Array<{ startIndex: number; endIndex: number; length: number }> = [];

  const step = calculateChunkStep(config);
  let currentPosition = 0;
  let splitCount = 0;

  while (currentPosition < text.length) {
    const remainingText = text.slice(currentPosition);
    const remainingLength = remainingText.length;

    if (remainingLength <= config.maxChunkSize || remainingLength <= config.minChunkSize) {
      // Last chunk - take everything remaining
      const chunk = remainingText.trim();
      if (chunk.length > 0) {
        chunks.push(chunk);
        metadatas.push({
          startIndex: currentPosition,
          endIndex: currentPosition + chunk.length,
          length: chunk.length,
        });
      }
      break;
    }

    // Find optimal split point
    const splitPoint = Math.min(currentPosition + config.maxChunkSize,
      currentPosition + config.minChunkSize + config.overlapSize);
    
    const bestSplit = findBestSplitPoint(remainingText, splitPoint - currentPosition);
    const newCurrentPosition = currentPosition + bestSplit;

    // Extract chunk with overlap
    const endPos = Math.min(newCurrentPosition + config.overlapSize, text.length);
    const chunk = text.slice(currentPosition, newCurrentPosition).trim();

    if (chunk.length >= config.minChunkSize) {
      chunks.push(chunk);
      metadatas.push({
        startIndex: currentPosition,
        endIndex: currentPosition + chunk.length,
        length: chunk.length,
      });
      splitCount++;
    }

    // Move to next position (overlap for context)
    currentPosition = newCurrentPosition;
    
    // Safety check to prevent infinite loops
    if (newCurrentPosition === currentPosition || currentPosition >= text.length) {
      break;
    }
  }

  return {
    chunks,
    metadatas,
    totalOriginalLength,
    chunkCount: chunks.length + (splitCount > 0 ? splitCount : 1),
  };
}

/**
 * Smart chunker that adapts to model context limits
 */
export function smartChunk(text: string, embedderModel?: string): ChunkResult {
  const limit = embedderModel ? EMBEDDING_CONTEXT_LIMITS[embedderModel] : 8192;
  
  const config = {
    maxChunkSize: Math.max(1000, Math.floor(limit * 0.7)), // 70% of context limit
    overlapSize: Math.floor(limit * 0.05), // 5% overlap
    minChunkSize: Math.max(100, Math.floor(limit * 0.1)),
    semanticSplit: true,
    maxLinesPerChunk: 50,
  };

  return chunkDocument(text, config);
}

// ============================================================================
// Export
// ============================================================================

export default chunkDocument;
