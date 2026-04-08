import type { QualityMetrics } from '../types';

const LUMA_R = 0.299;
const LUMA_G = 0.587;
const LUMA_B = 0.114;

function toLuma(data: Uint8ClampedArray, width: number, height: number): Float64Array {
    const out = new Float64Array(width * height);
    for (let i = 0; i < width * height; i += 1) {
        const idx = i * 4;
        out[i] = LUMA_R * data[idx] + LUMA_G * data[idx + 1] + LUMA_B * data[idx + 2];
    }
    return out;
}

function mean(arr: Float64Array): number {
    let sum = 0;
    for (let i = 0; i < arr.length; i += 1) {
        sum += arr[i];
    }
    return sum / arr.length;
}

function computeGlobalSsim(x: Float64Array, y: Float64Array): number {
    const n = x.length;
    if (n === 0) return 1;

    const muX = mean(x);
    const muY = mean(y);

    let sigmaX2 = 0;
    let sigmaY2 = 0;
    let sigmaXY = 0;
    for (let i = 0; i < n; i += 1) {
        const dx = x[i] - muX;
        const dy = y[i] - muY;
        sigmaX2 += dx * dx;
        sigmaY2 += dy * dy;
        sigmaXY += dx * dy;
    }

    sigmaX2 /= n;
    sigmaY2 /= n;
    sigmaXY /= n;

    const L = 255;
    const c1 = (0.01 * L) ** 2;
    const c2 = (0.03 * L) ** 2;

    const numerator = (2 * muX * muY + c1) * (2 * sigmaXY + c2);
    const denominator = (muX * muX + muY * muY + c1) * (sigmaX2 + sigmaY2 + c2);
    if (denominator === 0) return numerator === 0 ? 1 : 0;
    return numerator / denominator;
}

interface IntegralBundle {
    stride: number;
    sumX: Float64Array;
    sumY: Float64Array;
    sumX2: Float64Array;
    sumY2: Float64Array;
    sumXY: Float64Array;
}

function buildIntegrals(x: Float64Array, y: Float64Array, width: number, height: number): IntegralBundle {
    const stride = width + 1;
    const size = (width + 1) * (height + 1);

    const sumX = new Float64Array(size);
    const sumY = new Float64Array(size);
    const sumX2 = new Float64Array(size);
    const sumY2 = new Float64Array(size);
    const sumXY = new Float64Array(size);

    for (let row = 1; row <= height; row += 1) {
        let rowSumX = 0;
        let rowSumY = 0;
        let rowSumX2 = 0;
        let rowSumY2 = 0;
        let rowSumXY = 0;

        for (let col = 1; col <= width; col += 1) {
            const idx = (row - 1) * width + (col - 1);
            const xv = x[idx];
            const yv = y[idx];

            rowSumX += xv;
            rowSumY += yv;
            rowSumX2 += xv * xv;
            rowSumY2 += yv * yv;
            rowSumXY += xv * yv;

            const off = row * stride + col;
            const prevRow = (row - 1) * stride + col;

            sumX[off] = sumX[prevRow] + rowSumX;
            sumY[off] = sumY[prevRow] + rowSumY;
            sumX2[off] = sumX2[prevRow] + rowSumX2;
            sumY2[off] = sumY2[prevRow] + rowSumY2;
            sumXY[off] = sumXY[prevRow] + rowSumXY;
        }
    }

    return { stride, sumX, sumY, sumX2, sumY2, sumXY };
}

function regionSum(integral: Float64Array, stride: number, x0: number, y0: number, x1: number, y1: number): number {
    const a = integral[y0 * stride + x0];
    const b = integral[y0 * stride + x1];
    const c = integral[y1 * stride + x0];
    const d = integral[y1 * stride + x1];
    return d - b - c + a;
}

function computeLocalSsim(x: Float64Array, y: Float64Array, width: number, height: number): number {
    if (width < 2 || height < 2) {
        return computeGlobalSsim(x, y);
    }

    const windowSize = Math.min(8, width, height);
    if (windowSize < 2) {
        return computeGlobalSsim(x, y);
    }

    const { stride, sumX, sumY, sumX2, sumY2, sumXY } = buildIntegrals(x, y, width, height);
    const area = windowSize * windowSize;

    const L = 255;
    const c1 = (0.01 * L) ** 2;
    const c2 = (0.03 * L) ** 2;

    let ssimSum = 0;
    let count = 0;

    for (let y0 = 0; y0 <= height - windowSize; y0 += 1) {
        const y1 = y0 + windowSize;
        for (let x0 = 0; x0 <= width - windowSize; x0 += 1) {
            const x1 = x0 + windowSize;

            const sx = regionSum(sumX, stride, x0, y0, x1, y1);
            const sy = regionSum(sumY, stride, x0, y0, x1, y1);
            const sx2 = regionSum(sumX2, stride, x0, y0, x1, y1);
            const sy2 = regionSum(sumY2, stride, x0, y0, x1, y1);
            const sxy = regionSum(sumXY, stride, x0, y0, x1, y1);

            const muX = sx / area;
            const muY = sy / area;
            const ex2 = sx2 / area;
            const ey2 = sy2 / area;
            const exy = sxy / area;

            const sigmaX2 = Math.max(0, ex2 - muX * muX);
            const sigmaY2 = Math.max(0, ey2 - muY * muY);
            const sigmaXY = exy - muX * muY;

            const numerator = (2 * muX * muY + c1) * (2 * sigmaXY + c2);
            const denominator = (muX * muX + muY * muY + c1) * (sigmaX2 + sigmaY2 + c2);

            const localSsim = denominator === 0 ? (numerator === 0 ? 1 : 0) : numerator / denominator;
            ssimSum += localSsim;
            count += 1;
        }
    }

    if (count === 0) return computeGlobalSsim(x, y);
    return ssimSum / count;
}

function computeNccRgb(original: Uint8ClampedArray, reconstructed: Uint8ClampedArray): number {
    const samples = Math.floor(original.length / 4) * 3;
    if (samples <= 0) return 1;

    let sumX = 0;
    let sumY = 0;
    let identical = true;

    for (let i = 0; i < original.length; i += 4) {
        const xr = original[i];
        const xg = original[i + 1];
        const xb = original[i + 2];
        const yr = reconstructed[i];
        const yg = reconstructed[i + 1];
        const yb = reconstructed[i + 2];

        sumX += xr + xg + xb;
        sumY += yr + yg + yb;

        if (identical && (xr !== yr || xg !== yg || xb !== yb)) {
            identical = false;
        }
    }

    const meanX = sumX / samples;
    const meanY = sumY / samples;

    let numerator = 0;
    let denomX = 0;
    let denomY = 0;

    for (let i = 0; i < original.length; i += 4) {
        const dxr = original[i] - meanX;
        const dxg = original[i + 1] - meanX;
        const dxb = original[i + 2] - meanX;
        const dyr = reconstructed[i] - meanY;
        const dyg = reconstructed[i + 1] - meanY;
        const dyb = reconstructed[i + 2] - meanY;

        numerator += dxr * dyr + dxg * dyg + dxb * dyb;
        denomX += dxr * dxr + dxg * dxg + dxb * dxb;
        denomY += dyr * dyr + dyg * dyg + dyb * dyb;
    }

    if (denomX === 0 && denomY === 0) {
        return identical ? 1 : 0;
    }
    if (denomX === 0 || denomY === 0) {
        return 0;
    }

    const ncc = numerator / Math.sqrt(denomX * denomY);
    return Math.max(-1, Math.min(1, ncc));
}

export function evaluateQuality(original: ImageData, reconstructed: ImageData): QualityMetrics {
    if (original.width !== reconstructed.width || original.height !== reconstructed.height) {
        throw new Error('Quality evaluation failed: image dimensions do not match');
    }

    const pixels = original.width * original.height;
    const samples = pixels * 3;

    let mseSum = 0;
    let maeSum = 0;

    for (let i = 0; i < pixels; i += 1) {
        const idx = i * 4;

        const dr = original.data[idx] - reconstructed.data[idx];
        const dg = original.data[idx + 1] - reconstructed.data[idx + 1];
        const db = original.data[idx + 2] - reconstructed.data[idx + 2];

        mseSum += dr * dr + dg * dg + db * db;
        maeSum += Math.abs(dr) + Math.abs(dg) + Math.abs(db);
    }

    const mse = mseSum / samples;
    const mae = maeSum / samples;
    const rmse = Math.sqrt(mse);
    const psnr = mse === 0 ? Infinity : 10 * Math.log10((255 * 255) / mse);

    const lumaOriginal = toLuma(original.data, original.width, original.height);
    const lumaReconstructed = toLuma(reconstructed.data, reconstructed.width, reconstructed.height);
    const ssim = computeLocalSsim(lumaOriginal, lumaReconstructed, original.width, original.height);

    const ncc = computeNccRgb(original.data, reconstructed.data);

    return {
        mse,
        rmse,
        mae,
        psnr,
        ssim,
        ncc,
    };
}
