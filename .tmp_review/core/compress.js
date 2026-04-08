import { computeSVD } from './svd';
import { clampMatrix, zeros } from './matrix';
import { imageDataToRGBMatrices, rgbMatricesToImageData } from './image-io';
function reconstructFromFactors(U, S, Vt, k) {
    const rows = U.length;
    const cols = Vt[0]?.length ?? 0;
    const rank = Math.max(1, Math.min(k, S.length));
    const out = zeros(rows, cols);
    for (let i = 0; i < rows; i += 1) {
        for (let t = 0; t < rank; t += 1) {
            const scaled = U[i][t] * S[t];
            if (scaled === 0)
                continue;
            const vtRow = Vt[t];
            for (let j = 0; j < cols; j += 1) {
                out[i][j] += scaled * vtRow[j];
            }
        }
    }
    return clampMatrix(out, 0, 255);
}
function reconstructByRankK(channel, k, svdMode = 'full') {
    const { U, S, Vt } = computeSVD(channel, {
        copyInput: false,
        mode: svdMode,
        rank: k,
    });
    return reconstructFromFactors(U, S, Vt, k);
}
export function compressImageBySVD(imageData, k, svdMode = 'full') {
    // 再次防御式拷贝，确保输入 imageData 不会被修改。
    const safeImageData = new ImageData(new Uint8ClampedArray(imageData.data), imageData.width, imageData.height);
    const { r, g, b, width, height } = imageDataToRGBMatrices(safeImageData);
    const r2 = reconstructByRankK(r, k, svdMode);
    const g2 = reconstructByRankK(g, k, svdMode);
    const b2 = reconstructByRankK(b, k, svdMode);
    const compressedImageData = rgbMatricesToImageData(r2, g2, b2, width, height);
    // 原图存储量（忽略 alpha）：3mn
    const originalStorage = 3 * width * height;
    // rank-k SVD 估计存储量（RGB 三通道）：3k(m+n+1)
    const compressedStorage = 3 * k * (height + width + 1);
    return {
        imageData: compressedImageData,
        rankUsed: k,
        estimatedCompressionRatio: originalStorage / compressedStorage,
    };
}
function extractBlock(source, r0, c0, rows, cols) {
    const out = zeros(rows, cols);
    for (let i = 0; i < rows; i += 1) {
        for (let j = 0; j < cols; j += 1) {
            out[i][j] = source[r0 + i][c0 + j];
        }
    }
    return out;
}
function writeBlock(target, block, r0, c0) {
    const rows = block.length;
    const cols = block[0]?.length ?? 0;
    for (let i = 0; i < rows; i += 1) {
        for (let j = 0; j < cols; j += 1) {
            target[r0 + i][c0 + j] = block[i][j];
        }
    }
}
function yieldToUI() {
    return new Promise((resolve) => {
        setTimeout(resolve, 0);
    });
}
/**
 * Worker 不可用时的非阻塞回退方案：块 SVD + 分段让出主线程。
 * 目标是"不卡死页面"，而不是数学上与整图 SVD 完全等价。
 */
export async function compressImageBySVDNonBlocking(imageData, k, options = {}, svdMode = 'full') {
    const blockSize = Math.max(16, options.blockSize ?? 96);
    const safeImageData = new ImageData(new Uint8ClampedArray(imageData.data), imageData.width, imageData.height);
    const { r, g, b, width, height } = imageDataToRGBMatrices(safeImageData);
    const r2 = zeros(height, width);
    const g2 = zeros(height, width);
    const b2 = zeros(height, width);
    const rowBlocks = Math.ceil(height / blockSize);
    const colBlocks = Math.ceil(width / blockSize);
    const totalBlocks = rowBlocks * colBlocks * 3;
    let finished = 0;
    const processChannel = async (src, dst) => {
        for (let br = 0; br < rowBlocks; br += 1) {
            const r0 = br * blockSize;
            const h = Math.min(blockSize, height - r0);
            for (let bc = 0; bc < colBlocks; bc += 1) {
                const c0 = bc * blockSize;
                const w = Math.min(blockSize, width - c0);
                const block = extractBlock(src, r0, c0, h, w);
                const kLocal = Math.max(1, Math.min(k, Math.min(h, w)));
                const rec = reconstructByRankK(block, kLocal, svdMode);
                writeBlock(dst, rec, r0, c0);
                finished += 1;
                options.onProgress?.(finished / totalBlocks);
                // 每个 block 后主动让出主线程，避免页面假死。
                await yieldToUI();
            }
        }
    };
    await processChannel(r, r2);
    await processChannel(g, g2);
    await processChannel(b, b2);
    const compressedImageData = rgbMatricesToImageData(r2, g2, b2, width, height);
    const originalStorage = 3 * width * height;
    const compressedStorage = 3 * k * (height + width + 1);
    return {
        imageData: compressedImageData,
        rankUsed: k,
        estimatedCompressionRatio: originalStorage / compressedStorage,
    };
}
