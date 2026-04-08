import type { CompressionResult, DWTParams, DWTWavelet, Matrix } from '../types';
import { imageDataToRGBMatrices, rgbMatricesToImageData } from './image-io';
import {
    estimateDWTCompressedBytes,
    estimateDWTCompressionRatio,
    estimateOriginalImageBytes,
} from './compression-ratio';


interface WaveletFilter {
    lo_d: number[];
    hi_d: number[];
    lo_r: number[];
    hi_r: number[];
}

/**
 *   lo_r = reverse(lo_d)
 *   hi_r = reverse(hi_d)
 */
function buildOrthogonalFilters(lo_d: number[]): WaveletFilter {
    const L = lo_d.length;
    const hi_d = new Array<number>(L);
    for (let n = 0; n < L; n++) {
        hi_d[n] = ((n & 1) === 0 ? 1 : -1) * lo_d[L - 1 - n];
    }
    const lo_r = [...lo_d].reverse();
    const hi_r = [...hi_d].reverse();
    return { lo_d, hi_d, lo_r, hi_r };
}

const WAVELET_FILTERS: Record<DWTWavelet, WaveletFilter> = {
    haar: buildOrthogonalFilters([
        1 / Math.SQRT2,
        1 / Math.SQRT2,
    ]),
    db2: buildOrthogonalFilters([
        -0.12940952255092145,
        0.22414386804185735,
        0.83651630373746899,
        0.48296291314469025,
    ]),
    db4: buildOrthogonalFilters([
        -0.01059740178499728,
        0.03288301166698295,
        0.03084138183598697,
        -0.18703481171888114,
        -0.02798376941698385,
        0.63088076792959036,
        0.71484657055254153,
        0.23037781330885523,
    ]),
};

function getFilter(wavelet: DWTWavelet): WaveletFilter {
    return WAVELET_FILTERS[wavelet];
}

function computeMaxWaveletLevels(width: number, height: number, filterLength: number): number {
    const minDim = Math.max(1, Math.min(width, height));
    const dyadicBound = Math.floor(Math.log2(minDim));
    const filterBound = filterLength > 1 ? Math.floor(Math.log2(minDim / (filterLength - 1))) : dyadicBound;
    return Math.max(1, Math.min(dyadicBound, Math.max(1, filterBound)));
}

/**
 * Periodic extension.
 * For orthogonal wavelets with the current correlation-form implementation,
 * periodic extension avoids one-sided phase bias and improves reconstruction consistency.
 */
function periodicRef(arr: number[], idx: number): number {
    const N = arr.length;
    if (N <= 1) return arr[0] ?? 0;
    let i = idx % N;
    if (i < 0) i += N;
    return arr[i];
}

/**
 *
 * Boundary uses periodic extension. Output length is ceil(N / 2). */
function dwt1d(signal: number[], lo: number[], hi: number[]): { approx: number[]; detail: number[] } {
    const N = signal.length;
    const L = lo.length;
    const halfLen = Math.ceil(N / 2);

    const approx = new Array<number>(halfLen);
    const detail = new Array<number>(halfLen);

    for (let k = 0; k < halfLen; k++) {
        let sumLo = 0;
        let sumHi = 0;
        for (let j = 0; j < L; j++) {
            const val = periodicRef(signal, 2 * k + j);
            sumLo += lo[j] * val;
            sumHi += hi[j] * val;
        }
        approx[k] = sumLo;
        detail[k] = sumHi;
    }

    return { approx, detail };
}

/**
 *
 * Use analysis filters (lo_d, hi_d) to match the correlation-form forward transform.
 * Boundary uses periodic extension. */
function idwt1d(approx: number[], detail: number[], lo: number[], hi: number[], outLen: number): number[] {
    const L = lo.length;
    const result = new Array<number>(outLen);

    for (let n = 0; n < outLen; n++) {
        let sum = 0;
        for (let j = 0; j < L; j++) {
            // k may be negative; out-of-range samples are handled by periodic extension.
            const diff = n - j;
            if ((diff & 1) !== 0) continue; // must be even
            const k = diff >> 1;
            sum += lo[j] * periodicRef(approx, k) + hi[j] * periodicRef(detail, k);
        }
        result[n] = sum;
    }

    return result;
}


/** Transpose a `number[][]` matrix. */
function transposeArr(mat: number[][]): number[][] {
    const rows = mat.length;
    const cols = mat[0]?.length ?? 0;
    const out: number[][] = Array.from({ length: cols }, () => new Array<number>(rows));
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            out[j][i] = mat[i][j];
        }
    }
    return out;
}

interface DWT2DLevel {
    /** Approximation sub-band LL. */
    LL: number[][];
    /** Horizontal detail sub-band LH. */
    LH: number[][];
    /** Vertical detail sub-band HL. */
    HL: number[][];
    /** Diagonal detail sub-band HH. */
    HH: number[][];
    /** Original row count before this level decomposition. */
    origRows: number;
    /** Original column count before this level decomposition. */
    origCols: number;
}

/** Single-level 2D forward DWT. */
function dwt2dSingle(input: number[][], filter: WaveletFilter): DWT2DLevel {
    const rows = input.length;
    const cols = input[0]?.length ?? 0;
    const { lo_d, hi_d } = filter;
    const halfCols = Math.ceil(cols / 2);

    // Step 1: transform by rows.
    const rowLo: number[][] = new Array(rows);
    const rowHi: number[][] = new Array(rows);
    for (let r = 0; r < rows; r++) {
        const { approx, detail } = dwt1d(input[r], lo_d, hi_d);
        rowLo[r] = approx;
        rowHi[r] = detail;
    }

    // Step 2: transform by columns.
    // Apply row-wise DWT on transposed matrices, equivalent to column-wise DWT.
    const rowLoT = transposeArr(rowLo);  // halfCols x rows
    const rowHiT = transposeArr(rowHi);  // halfCols 脳 rows

    const halfRows = Math.ceil(rows / 2);
    const LL: number[][] = new Array(halfRows);
    const LH: number[][] = new Array(halfRows);
    const HL: number[][] = new Array(halfRows);
    const HH: number[][] = new Array(halfRows);

    for (let c = 0; c < halfCols; c++) {
        const { approx: ll_col, detail: lh_col } = dwt1d(rowLoT[c], lo_d, hi_d);
        const { approx: hl_col, detail: hh_col } = dwt1d(rowHiT[c], lo_d, hi_d);

        for (let r = 0; r < halfRows; r++) {
            if (!LL[r]) {
                LL[r] = new Array(halfCols);
                LH[r] = new Array(halfCols);
                HL[r] = new Array(halfCols);
                HH[r] = new Array(halfCols);
            }
            LL[r][c] = ll_col[r];
            LH[r][c] = lh_col[r];
            HL[r][c] = hl_col[r];
            HH[r][c] = hh_col[r];
        }
    }

    return { LL, LH, HL, HH, origRows: rows, origCols: cols };
}

/** Single-level 2D inverse DWT. */
function idwt2dSingle(level: DWT2DLevel, filter: WaveletFilter): number[][] {
    const { LL, LH, HL, HH, origRows, origCols } = level;
    // Inverse transform uses analysis filters, matching the correlation convention.
    const { lo_d, hi_d } = filter;
    const halfRows = LL.length;
    const halfCols = LL[0]?.length ?? 0;

    const rowLo: number[][] = Array.from({ length: origRows }, () => new Array<number>(halfCols));
    const rowHi: number[][] = Array.from({ length: origRows }, () => new Array<number>(halfCols));

    for (let c = 0; c < halfCols; c++) {
        const ll_col = new Array<number>(halfRows);
        const lh_col = new Array<number>(halfRows);
        const hl_col = new Array<number>(halfRows);
        const hh_col = new Array<number>(halfRows);

        for (let r = 0; r < halfRows; r++) {
            ll_col[r] = LL[r][c];
            lh_col[r] = LH[r][c];
            hl_col[r] = HL[r][c];
            hh_col[r] = HH[r][c];
        }

        const recLo = idwt1d(ll_col, lh_col, lo_d, hi_d, origRows);
        const recHi = idwt1d(hl_col, hh_col, lo_d, hi_d, origRows);

        for (let r = 0; r < origRows; r++) {
            rowLo[r][c] = recLo[r];
            rowHi[r][c] = recHi[r];
        }
    }

    const result: number[][] = new Array(origRows);
    for (let r = 0; r < origRows; r++) {
        result[r] = idwt1d(rowLo[r], rowHi[r], lo_d, hi_d, origCols);
    }

    return result;
}


interface DWT2DDecomposition {
    /** Detail sub-bands for each level (LH/HL/HH), from fine to coarse. */
    detailLevels: Array<{
        LH: number[][];
        HL: number[][];
        HH: number[][];
        origRows: number;
        origCols: number;
    }>;
    /** Final low-frequency approximation LL. */
    approx: Array<Array<number>>;
}

/** Multi-level 2D forward DWT decomposition. */
function dwtDecompose2D(input: number[][], wavelet: DWTWavelet, levels: number): DWT2DDecomposition {
    const filter = getFilter(wavelet);
    const detailLevels: DWT2DDecomposition['detailLevels'] = [];
    let current = input;

    for (let l = 0; l < levels; l++) {
        // Stop when matrix is too small to continue (minimum 2x2).
        if (current.length < 2 || (current[0]?.length ?? 0) < 2) break;

        const level = dwt2dSingle(current, filter);
        detailLevels.push({
            LH: level.LH,
            HL: level.HL,
            HH: level.HH,
            origRows: level.origRows,
            origCols: level.origCols,
        });
        current = level.LL;
    }

    return { detailLevels, approx: current };
}

/** Reconstruct matrix from multi-level 2D DWT decomposition. */
function dwtReconstruct2D(decomp: DWT2DDecomposition, wavelet: DWTWavelet): number[][] {
    const filter = getFilter(wavelet);
    let current = decomp.approx;

    for (let l = decomp.detailLevels.length - 1; l >= 0; l--) {
        const dl = decomp.detailLevels[l];
        const level: DWT2DLevel = {
            LL: current,
            LH: dl.LH,
            HL: dl.HL,
            HH: dl.HH,
            origRows: dl.origRows,
            origCols: dl.origCols,
        };
        current = idwt2dSingle(level, filter);
    }

    return current;
}


/** Hard thresholding: set coefficients with |x| < lambda to zero. */
function hardThreshold(mat: number[][], lambda: number): number[][] {
    return mat.map(row =>
        row.map(v => Math.abs(v) < lambda ? 0 : v),
    );
}

/** Soft thresholding: sign(x) * max(0, |x| - lambda). */
function softThreshold(mat: number[][], lambda: number): number[][] {
    return mat.map(row =>
        row.map(v => {
            const abs = Math.abs(v);
            if (abs < lambda) return 0;
            return Math.sign(v) * (abs - lambda);
        }),
    );
}

/** Apply thresholding to detail sub-bands; keep LL unchanged. */
function applyThreshold(
    decomp: DWT2DDecomposition,
    threshold: number,
    mode: 'hard' | 'soft',
): DWT2DDecomposition {
    const threshFn = mode === 'hard' ? hardThreshold : softThreshold;

    const newDetails = decomp.detailLevels.map(dl => ({
        LH: threshFn(dl.LH, threshold),
        HL: threshFn(dl.HL, threshold),
        HH: threshFn(dl.HH, threshold),
        origRows: dl.origRows,
        origCols: dl.origCols,
    }));

    return {
        approx: decomp.approx, // Keep LL unchanged.
        detailLevels: newDetails,
    };
}


/** Count total/non-zero coefficients in a decomposition. */
function countNonZeroCoefficients(decomp: DWT2DDecomposition): { total: number; nonZero: number } {
    let total = 0;
    let nonZero = 0;

    const countMat = (mat: number[][]) => {
        for (const row of mat) {
            for (const v of row) {
                total++;
                if (v !== 0) nonZero++;
            }
        }
    };

    countMat(decomp.approx);

    for (const dl of decomp.detailLevels) {
        countMat(dl.LH);
        countMat(dl.HL);
        countMat(dl.HH);
    }

    return { total, nonZero };
}

interface DWTThresholdProfile {
    width: number;
    height: number;
    safeLevels: number;
    totalCoefficients: number;
    approxNonZero: number;
    thresholdMin: number;
    thresholdMax: number;
    hardDetailNonZeroByThreshold: number[];
    softDetailNonZeroByThreshold: number[];
}

function buildThresholdProfile(
    imageData: ImageData,
    wavelet: DWTWavelet,
    levels: number,
    thresholdMin: number,
    thresholdMax: number,
): DWTThresholdProfile {
    const safe = new ImageData(new Uint8ClampedArray(imageData.data), imageData.width, imageData.height);
    const { r, g, b, width, height } = imageDataToRGBMatrices(safe);

    const filterLength = getFilter(wavelet).lo_d.length;
    const maxLevels = computeMaxWaveletLevels(width, height, filterLength);
    const safeLevels = Math.min(levels, maxLevels);
    const minT = Math.max(0, Math.floor(thresholdMin));
    const maxT = Math.max(minT, Math.floor(thresholdMax));
    const thresholdSpan = maxT - minT + 1;
    const hardDiff = new Int32Array(thresholdSpan + 1);
    const softDiff = new Int32Array(thresholdSpan + 1);

    const addAliveThresholdRange = (diff: Int32Array, maxAliveThreshold: number) => {
        if (maxAliveThreshold < minT) return;
        const end = Math.min(maxAliveThreshold, maxT) - minT;
        diff[0] += 1;
        diff[end + 1] -= 1;
    };

    const processChannel = (channel: Matrix): { total: number; approxNonZero: number } => {
        const decomp = dwtDecompose2D(channel, wavelet, safeLevels);

        let total = 0;
        let approxNonZero = 0;

        for (const row of decomp.approx) {
            for (const v of row) {
                total += 1;
                if (v !== 0) approxNonZero += 1;
            }
        }

        for (const level of decomp.detailLevels) {
            const mats = [level.LH, level.HL, level.HH];
            for (const mat of mats) {
                for (const row of mat) {
                    for (const v of row) {
                        total += 1;
                        const abs = Math.abs(v);
                        if (abs <= 0) continue;
                        // hard: |x| >= t
                        addAliveThresholdRange(hardDiff, Math.floor(abs));
                        // soft: |x| > t
                        addAliveThresholdRange(softDiff, Math.ceil(abs) - 1);
                    }
                }
            }
        }

        return { total, approxNonZero };
    };

    const channels = [processChannel(r), processChannel(g), processChannel(b)];
    const totalCoefficients = channels[0].total + channels[1].total + channels[2].total;
    const approxNonZero = channels[0].approxNonZero + channels[1].approxNonZero + channels[2].approxNonZero;

    const hardDetailNonZeroByThreshold = new Array<number>(thresholdSpan);
    const softDetailNonZeroByThreshold = new Array<number>(thresholdSpan);
    let hardRunning = 0;
    let softRunning = 0;
    for (let i = 0; i < thresholdSpan; i++) {
        hardRunning += hardDiff[i];
        softRunning += softDiff[i];
        hardDetailNonZeroByThreshold[i] = hardRunning;
        softDetailNonZeroByThreshold[i] = softRunning;
    }

    return {
        width,
        height,
        safeLevels,
        totalCoefficients,
        approxNonZero,
        thresholdMin: minT,
        thresholdMax: maxT,
        hardDetailNonZeroByThreshold,
        softDetailNonZeroByThreshold,
    };
}

function estimateRatioByThreshold(
    profile: DWTThresholdProfile,
    threshold: number,
    thresholdMode: 'hard' | 'soft',
): number {
    const clampedThreshold = Math.max(
        profile.thresholdMin,
        Math.min(profile.thresholdMax, Math.floor(threshold)),
    );
    const idx = clampedThreshold - profile.thresholdMin;
    const detailNonZero = thresholdMode === 'hard'
        ? profile.hardDetailNonZeroByThreshold[idx]
        : profile.softDetailNonZeroByThreshold[idx];

    const nonZeroCoefficients = profile.approxNonZero + detailNonZero;
    return estimateDWTCompressionRatio(
        profile.width,
        profile.height,
        nonZeroCoefficients,
        profile.totalCoefficients,
    );
}

function ratioDistance(actual: number, target: number): number {
    if (!Number.isFinite(actual) && !Number.isFinite(target)) return 0;
    if (!Number.isFinite(actual)) return Number.POSITIVE_INFINITY;
    if (!Number.isFinite(target)) return Number.POSITIVE_INFINITY;
    return Math.abs(actual - target);
}

export interface DWTThresholdSuggestion {
    threshold: number;
    estimatedCompressionRatio: number;
    safeLevels: number;
    minReachableRatio: number;
    maxReachableRatio: number;
}

export function suggestDWTThresholdForTargetRatio(
    imageData: ImageData,
    params: Pick<DWTParams, 'wavelet' | 'levels' | 'thresholdMode'>,
    targetRatio: number,
    thresholdMin = 0,
    thresholdMax = 100,
): DWTThresholdSuggestion {
    const minT = Math.max(0, Math.floor(thresholdMin));
    const maxT = Math.max(minT, Math.floor(thresholdMax));
    const profile = buildThresholdProfile(imageData, params.wavelet, params.levels, minT, maxT);
    const target = Number.isFinite(targetRatio) && targetRatio > 0 ? targetRatio : 1;

    const ratioAt = (t: number) => estimateRatioByThreshold(profile, t, params.thresholdMode);
    const minReachableRatio = ratioAt(minT);
    const maxReachableRatio = ratioAt(maxT);

    let threshold = minT;
    let estimatedCompressionRatio = minReachableRatio;

    if (target <= minReachableRatio) {
        threshold = minT;
        estimatedCompressionRatio = minReachableRatio;
    } else if (target >= maxReachableRatio) {
        threshold = maxT;
        estimatedCompressionRatio = maxReachableRatio;
    } else {
        let lo = minT;
        let hi = maxT;
        while (lo < hi) {
            const mid = lo + ((hi - lo) >> 1);
            const ratioMid = ratioAt(mid);
            if (ratioMid >= target) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }

        const candidateA = lo;
        const candidateB = Math.max(minT, lo - 1);
        const ratioA = ratioAt(candidateA);
        const ratioB = ratioAt(candidateB);

        if (ratioDistance(ratioB, target) <= ratioDistance(ratioA, target)) {
            threshold = candidateB;
            estimatedCompressionRatio = ratioB;
        } else {
            threshold = candidateA;
            estimatedCompressionRatio = ratioA;
        }
    }

    return {
        threshold,
        estimatedCompressionRatio,
        safeLevels: profile.safeLevels,
        minReachableRatio,
        maxReachableRatio,
    };
}


/** DWT image compression entry. */
export function compressImageByDWT(imageData: ImageData, params: DWTParams): CompressionResult {
    const { wavelet, levels, threshold, thresholdMode } = params;

    // Defensive copy.
    const safe = new ImageData(new Uint8ClampedArray(imageData.data), imageData.width, imageData.height);
    const { r, g, b, width, height } = imageDataToRGBMatrices(safe);

    const filterLength = getFilter(wavelet).lo_d.length;
    const maxLevels = computeMaxWaveletLevels(width, height, filterLength);
    const safeLevels = Math.min(levels, maxLevels);

    const processChannel = (channel: Matrix): { result: Matrix; nonZero: number; total: number } => {
        const decomp = dwtDecompose2D(channel, wavelet, safeLevels);
        const thresholded = applyThreshold(decomp, threshold, thresholdMode);
        const { total, nonZero } = countNonZeroCoefficients(thresholded);
        const reconstructed = dwtReconstruct2D(thresholded, wavelet);

        const clamped = reconstructed.map(row =>
            row.map(v => Math.max(0, Math.min(255, Math.round(v)))),
        );
        return { result: clamped, nonZero, total };
    };

    const rResult = processChannel(r);
    const gResult = processChannel(g);
    const bResult = processChannel(b);

    const totalCoeffs = rResult.total + gResult.total + bResult.total;
    const nonZeroCoeffs = rResult.nonZero + gResult.nonZero + bResult.nonZero;

    const reconstructedImageData = rgbMatricesToImageData(
        rResult.result,
        gResult.result,
        bResult.result,
        width,
        height,
    );

    const ratio = estimateDWTCompressionRatio(width, height, nonZeroCoeffs, totalCoeffs);
    const originalBytes = estimateOriginalImageBytes(width, height);
    const compressedBytes = estimateDWTCompressedBytes(nonZeroCoeffs, totalCoeffs);

    return {
        imageData: reconstructedImageData,
        rankUsed: safeLevels,
        estimatedCompressionRatio: ratio,
        method: 'dwt',
        notes: `${wavelet.toUpperCase()} · L${safeLevels} · λ=${threshold} · ${thresholdMode === 'hard' ? 'hard' : 'soft'} · nonzero ${nonZeroCoeffs}/${totalCoeffs} · ${(originalBytes / 1024).toFixed(1)}KB -> ${(compressedBytes / 1024).toFixed(1)}KB`,
    };
}


