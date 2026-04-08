import type { CompressionResult, Matrix } from '../types';
import { computeSVD, type SVDMode } from './svd';
import { clampMatrix, zeros } from './matrix';
import { imageDataToRGBMatrices, rgbMatricesToImageData } from './image-io';
import { clampRankByShape, estimateSVDCompressionRatio } from './compression-ratio';

interface NonBlockingOptions {
    blockSize?: number;
    onProgress?: (progress01: number) => void;
}

function reconstructFromFactors(U: Matrix, S: number[], Vt: Matrix, k: number): Matrix {
    const rows = U.length;
    const cols = Vt[0]?.length ?? 0;
    const rank = Math.max(1, Math.min(k, S.length));
    const out = zeros(rows, cols);

    for (let i = 0; i < rows; i += 1) {
        for (let t = 0; t < rank; t += 1) {
            const scaled = U[i][t] * S[t];
            if (scaled === 0) continue;
            const vtRow = Vt[t];
            for (let j = 0; j < cols; j += 1) {
                out[i][j] += scaled * vtRow[j];
            }
        }
    }

    return clampMatrix(out, 0, 255);
}

function reconstructByRankK(channel: Matrix, k: number, svdMode: SVDMode = 'full'): Matrix {
    const { U, S, Vt } = computeSVD(channel, {
        copyInput: false,
        mode: svdMode,
        rank: k,
    });
    return reconstructFromFactors(U, S, Vt, k);
}

export function compressImageBySVD(
    imageData: ImageData,
    k: number,
    svdMode: SVDMode = 'full',
): CompressionResult {
    // Defensive copy: avoid mutating input image data.
    const safeImageData = new ImageData(new Uint8ClampedArray(imageData.data), imageData.width, imageData.height);
    const { r, g, b, width, height } = imageDataToRGBMatrices(safeImageData);
    const safeRank = clampRankByShape(k, width, height);

    const r2 = reconstructByRankK(r, safeRank, svdMode);
    const g2 = reconstructByRankK(g, safeRank, svdMode);
    const b2 = reconstructByRankK(b, safeRank, svdMode);

    const compressedImageData = rgbMatricesToImageData(r2, g2, b2, width, height);

    return {
        imageData: compressedImageData,
        rankUsed: safeRank,
        estimatedCompressionRatio: estimateSVDCompressionRatio(width, height, safeRank),
    };
}

function extractBlock(source: Matrix, r0: number, c0: number, rows: number, cols: number): Matrix {
    const out = zeros(rows, cols);
    for (let i = 0; i < rows; i += 1) {
        for (let j = 0; j < cols; j += 1) {
            out[i][j] = source[r0 + i][c0 + j];
        }
    }
    return out;
}

function writeBlock(target: Matrix, block: Matrix, r0: number, c0: number): void {
    const rows = block.length;
    const cols = block[0]?.length ?? 0;
    for (let i = 0; i < rows; i += 1) {
        for (let j = 0; j < cols; j += 1) {
            target[r0 + i][c0 + j] = block[i][j];
        }
    }
}

function yieldToUI(): Promise<void> {
    return new Promise((resolve) => {
        setTimeout(resolve, 0);
    });
}

/**
 * Non-blocking fallback when worker is unavailable: block-wise SVD.
 * Goal: keep UI responsive, not strict equivalence to full-image SVD.
 */
export async function compressImageBySVDNonBlocking(
    imageData: ImageData,
    k: number,
    options: NonBlockingOptions = {},
    svdMode: SVDMode = 'full',
): Promise<CompressionResult> {
    const blockSize = Math.max(16, options.blockSize ?? 96);

    const safeImageData = new ImageData(new Uint8ClampedArray(imageData.data), imageData.width, imageData.height);
    const { r, g, b, width, height } = imageDataToRGBMatrices(safeImageData);
    const safeRank = clampRankByShape(k, width, height);

    const r2 = zeros(height, width);
    const g2 = zeros(height, width);
    const b2 = zeros(height, width);

    const rowBlocks = Math.ceil(height / blockSize);
    const colBlocks = Math.ceil(width / blockSize);
    const totalBlocks = rowBlocks * colBlocks * 3;
    let finished = 0;

    const processChannel = async (src: Matrix, dst: Matrix) => {
        for (let br = 0; br < rowBlocks; br += 1) {
            const r0 = br * blockSize;
            const h = Math.min(blockSize, height - r0);
            for (let bc = 0; bc < colBlocks; bc += 1) {
                const c0 = bc * blockSize;
                const w = Math.min(blockSize, width - c0);

                const block = extractBlock(src, r0, c0, h, w);
                const kLocal = Math.max(1, Math.min(safeRank, Math.min(h, w)));
                const rec = reconstructByRankK(block, kLocal, svdMode);
                writeBlock(dst, rec, r0, c0);

                finished += 1;
                options.onProgress?.(finished / totalBlocks);

                await yieldToUI();
            }
        }
    };

    await processChannel(r, r2);
    await processChannel(g, g2);
    await processChannel(b, b2);

    const compressedImageData = rgbMatricesToImageData(r2, g2, b2, width, height);

    return {
        imageData: compressedImageData,
        rankUsed: safeRank,
        estimatedCompressionRatio: estimateSVDCompressionRatio(width, height, safeRank),
    };
}
