import type { CompressionResult, DWTParams, DWTWavelet, Matrix } from '../types';
import { imageDataToRGBMatrices, rgbMatricesToImageData } from './image-io';
import {
    estimateDWTCompressedBytes,
    estimateDWTCompressionRatio,
    estimateOriginalImageBytes,
} from './compression-ratio';

// 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€ 灏忔尝婊ゆ尝鍣?鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

interface WaveletFilter {
    lo_d: number[];
    hi_d: number[];
    lo_r: number[];
    hi_r: number[];
}

/**
 * 浠庡垎瑙ｄ綆閫氭护娉㈠櫒鎺ㄥ瀹屾暣鐨勬浜?鍙屾浜ゆ护娉㈠櫒缁勩€? * 瀵逛簬姝ｄ氦灏忔尝锛圚aar, Daubechies 绯诲垪锛夛細
 *   hi_d[n] = (-1)^n 路 lo_d[L-1-n]     (浜ゆ浛鍙嶈浆)
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
 * 涓€缁村墠鍚?DWT锛堝垎瑙ｄ竴灞傦級銆? *
 * 浣跨敤 **鐩稿叧锛坈orrelation锛? 涓嬮噰鏍?* 绾﹀畾锛? *   cA[k] = 危_{j=0}^{L-1} lo[j] 路 x[2k + j]
 *   cD[k] = 危_{j=0}^{L-1} hi[j] 路 x[2k + j]
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
 * 涓€缁撮€?DWT锛堥噸寤轰竴灞傦級銆? *
 * 涓庡墠鍚戠浉鍏崇害瀹氶厤濂楃殑 **杞疆閲嶅缓** 鍏紡锛? *   x[n] = 危_k lo[n-2k] 路 cA[k] + 危_k hi[n-2k] 路 cD[k]
 *          鍏朵腑浠呭 n-2k 鈭?[0, L-1] 鐨?k 姹傚拰
 *
 * Use analysis filters (lo_d, hi_d) to match the correlation-form forward transform.
 * Boundary uses periodic extension. */
function idwt1d(approx: number[], detail: number[], lo: number[], hi: number[], outLen: number): number[] {
    const L = lo.length;
    const result = new Array<number>(outLen);

    for (let n = 0; n < outLen; n++) {
        let sum = 0;
        for (let j = 0; j < L; j++) {
            // 婊ゆ尝鍣ㄧ储寮?j = n - 2k  鈫? 2k = n - j  鈫? k = (n - j) / 2
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

// 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€ 2D DWT / IDWT 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

/** 杞疆涓€涓?number[][] 鐭╅樀 */
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

/**
 * 浜岀淮灏忔尝鍒嗚В鐨勪竴灞傜粨鏋溿€? */
interface DWT2DLevel {
    /** 杩戜技瀛愬甫 LL锛堢敤浜庡悗缁垎瑙ｆ垨鏈€缁堜繚鐣欙級 */
    LL: number[][];
    /** 姘村钩缁嗚妭瀛愬甫 LH */
    LH: number[][];
    /** 鍨傜洿缁嗚妭瀛愬甫 HL */
    HL: number[][];
    /** 瀵硅缁嗚妭瀛愬甫 HH */
    HH: number[][];
    /** 鍒嗚В鍓嶇殑琛屾暟锛堢敤浜庨€嗗彉鎹㈡仮澶嶅昂瀵革級 */
    origRows: number;
    /** 鍒嗚В鍓嶇殑鍒楁暟 */
    origCols: number;
}

/**
 * 浜岀淮鍓嶅悜 DWT锛堝崟灞傦級銆? *
 * 姝ラ锛? * 1. 瀵规瘡涓€琛屽仛 1D DWT 鈫?[L 琛?| H 琛宂
 * 2. 瀵圭粨鏋滅殑姣忎竴鍒楀仛 1D DWT 鈫?鍥涗釜瀛愬甫 LL, LH, HL, HH
 */
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

/**
 * 浜岀淮閫?DWT锛堝崟灞傦級锛屼粠鍥涗釜瀛愬甫閲嶅缓鍑哄師鐭╅樀銆? */
function idwt2dSingle(level: DWT2DLevel, filter: WaveletFilter): number[][] {
    const { LL, LH, HL, HH, origRows, origCols } = level;
    // Inverse transform uses analysis filters, matching the correlation convention.
    const { lo_d, hi_d } = filter;
    const halfRows = LL.length;
    const halfCols = LL[0]?.length ?? 0;

    // 姝ラ 1锛氭寜鍒楅€嗗彉鎹紝鎭㈠琛屾暟
    // 瀵规瘡涓€鍒椾綅缃紝灏?LL/LH 鍒楀悎骞堕噸寤轰綆棰戣閮ㄥ垎锛孒L/HH 鍒楀悎骞堕噸寤洪珮棰戣閮ㄥ垎
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

    // 姝ラ 2锛氭寜琛岄€嗗彉鎹紝鎭㈠鍒楁暟
    const result: number[][] = new Array(origRows);
    for (let r = 0; r < origRows; r++) {
        result[r] = idwt1d(rowLo[r], rowHi[r], lo_d, hi_d, origCols);
    }

    return result;
}

// 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€ 澶氬眰鍒嗚В / 閲嶅缓 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

interface DWT2DDecomposition {
    /** 鍚勫眰鐨勭粏鑺傚瓙甯?(LH, HL, HH)锛屼粠绗?0 灞傦紙鏈€缁嗭級鍒扮 levels-1 灞傦紙鏈€绮楋級 */
    detailLevels: Array<{ LH: number[][]; HL: number[][]; HH: number[][]; origRows: number; origCols: number }>;
    /** 鏈€缁堢殑浣庨杩戜技 LL */
    approx: number[][];
}

/**
 * 澶氬眰浜岀淮鍓嶅悜 DWT 鍒嗚В銆? * 瀵?LL 瀛愬甫閫掑綊鍒嗚В `levels` 灞傘€? */
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

/**
 * 浠庡灞傚垎瑙ｇ粨鏋滈€嗗悜閲嶅缓瀹屾暣鐭╅樀銆? * 浠庢渶绮楀眰锛堟渶娣卞眰锛夊紑濮嬶紝閫愬眰鍚戜笂 IDWT銆? */
function dwtReconstruct2D(decomp: DWT2DDecomposition, wavelet: DWTWavelet): number[][] {
    const filter = getFilter(wavelet);
    let current = decomp.approx;

    // 浠庢渶娣卞眰鍚戝閫愬眰閲嶅缓
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

// 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€ 闃堝€煎寲 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

/**
 * 纭槇鍊硷細|x| < 位 鐨勭郴鏁扮洿鎺ョ疆闆讹紝鍏朵綑淇濇寔涓嶅彉銆? * 淇濈暀鍘熷绯绘暟骞呭害锛岄噸寤烘洿閿愬埄浣嗗彲鑳芥湁鎸搩銆? */
function hardThreshold(mat: number[][], lambda: number): number[][] {
    return mat.map(row =>
        row.map(v => Math.abs(v) < lambda ? 0 : v),
    );
}

/**
 * 杞槇鍊硷細|x| < 位 缃浂锛屽惁鍒欏悜闆舵敹缂?位銆? * sign(x) 路 max(0, |x| - 位)
 * 閲嶅缓鏇村钩婊戯紝瑙嗚涓婃洿"鏌斿拰"銆? */
function softThreshold(mat: number[][], lambda: number): number[][] {
    return mat.map(row =>
        row.map(v => {
            const abs = Math.abs(v);
            if (abs < lambda) return 0;
            return Math.sign(v) * (abs - lambda);
        }),
    );
}

/**
 * 瀵瑰垎瑙ｇ粨鏋滅殑鎵€鏈夌粏鑺傚瓙甯︽墽琛岄槇鍊煎寲銆? * 娉ㄦ剰锛歀L锛堣繎浼煎瓙甯︼級涓嶅仛闃堝€煎寲锛屽畠淇濆瓨鍥惧儚鐨勪富浣撹兘閲忋€? */
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
        approx: decomp.approx, // LL 瀛愬甫涓嶅仛闃堝€煎寲
        detailLevels: newDetails,
    };
}

// 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€ 鍘嬬缉鐜囪緟鍔╄绠?鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

/**
 * 缁熻鍒嗚В缁撴灉涓潪闆剁郴鏁扮殑涓暟锛岀敤浜庝及绠楀帇缂╃巼銆? */
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

    // LL 杩戜技瀛愬甫
    countMat(decomp.approx);

    // 鍚勫眰缁嗚妭瀛愬甫
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

// 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€ 涓诲叆鍙ｅ嚱鏁?鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

/**
 * DWT 鍥惧儚鍘嬬缉涓诲嚱鏁般€? *
 * 娴佺▼锛? * 1. 灏?ImageData 鎷嗗垎涓?R/G/B 涓変釜 number[][] 鐭╅樀
 * 2. 瀵规瘡涓€氶亾鎵ц澶氬眰 2D DWT 鍒嗚В
 * 3. 瀵圭粏鑺傜郴鏁板仛闃堝€煎寲锛堝帇缂╋級
 * 4. 閫?DWT 閲嶅缓鍚勯€氶亾
 * 5. 鍚堝苟閫氶亾涓?ImageData 杈撳嚭
 *
 * @param imageData  鍘熷鍥惧儚
 * @param params     DWT 鍙傛暟锛堝皬娉㈠熀銆佸眰鏁般€侀槇鍊煎己搴︺€侀槇鍊兼ā寮忥級
 * @returns          鍘嬬缉缁撴灉锛堥噸寤哄浘鍍?+ 鍘嬬缉鐜囩瓑淇℃伅锛? */
export function compressImageByDWT(imageData: ImageData, params: DWTParams): CompressionResult {
    const { wavelet, levels, threshold, thresholdMode } = params;

    // Defensive copy.
    const safe = new ImageData(new Uint8ClampedArray(imageData.data), imageData.width, imageData.height);
    const { r, g, b, width, height } = imageDataToRGBMatrices(safe);

    // 闄愬埗鍒嗚В灞傛暟锛氫笉鑳借秴杩?log2(min(width, height))
    const filterLength = getFilter(wavelet).lo_d.length;
    const maxLevels = computeMaxWaveletLevels(width, height, filterLength);
    const safeLevels = Math.min(levels, maxLevels);

    // 瀵?R/G/B 涓夐€氶亾鍒嗗埆鍋?DWT 鈫?闃堝€煎寲 鈫?IDWT
    const processChannel = (channel: Matrix): { result: Matrix; nonZero: number; total: number } => {
        const decomp = dwtDecompose2D(channel, wavelet, safeLevels);
        const thresholded = applyThreshold(decomp, threshold, thresholdMode);
        const { total, nonZero } = countNonZeroCoefficients(thresholded);
        const reconstructed = dwtReconstruct2D(thresholded, wavelet);

        // 瑁佸壀鍒?[0, 255]
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


