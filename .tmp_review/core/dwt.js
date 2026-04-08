import { imageDataToRGBMatrices, rgbMatricesToImageData } from './image-io';
/**
 * 从分解低通滤波器推导完整的正交/双正交滤波器组。
 * 对于正交小波（Haar, Daubechies 系列）：
 *   hi_d[n] = (-1)^n · lo_d[L-1-n]     (交替反转)
 *   lo_r = reverse(lo_d)
 *   hi_r = reverse(hi_d)
 */
function buildOrthogonalFilters(lo_d) {
    const L = lo_d.length;
    const hi_d = new Array(L);
    for (let n = 0; n < L; n++) {
        hi_d[n] = ((n & 1) === 0 ? 1 : -1) * lo_d[L - 1 - n];
    }
    const lo_r = [...lo_d].reverse();
    const hi_r = [...hi_d].reverse();
    return { lo_d, hi_d, lo_r, hi_r };
}
const WAVELET_FILTERS = {
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
function getFilter(wavelet) {
    return WAVELET_FILTERS[wavelet];
}
/**
 * 对称延拓访问（whole-point symmetric）。
 * 使信号在边界处镜像反射，避免边缘伪影。
 * 例如：对 [a, b, c, d]，索引 -1 → b，索引 4 → c，索引 5 → b。
 */
function symRef(arr, idx) {
    const N = arr.length;
    if (N <= 1)
        return arr[0] ?? 0;
    let i = idx < 0 ? -idx : idx;
    const period = 2 * (N - 1);
    if (period > 0)
        i = i % period;
    if (i >= N)
        i = period - i;
    return arr[i];
}
/**
 * 一维前向 DWT（分解一层）。
 *
 * 使用 **相关（correlation）+ 下采样** 约定：
 *   cA[k] = Σ_{j=0}^{L-1} lo[j] · x[2k + j]
 *   cD[k] = Σ_{j=0}^{L-1} hi[j] · x[2k + j]
 *
 * 边界通过 symRef 对称反射处理。
 * 输出长度：ceil(N / 2)。
 */
function dwt1d(signal, lo, hi) {
    const N = signal.length;
    const L = lo.length;
    const halfLen = Math.ceil(N / 2);
    const approx = new Array(halfLen);
    const detail = new Array(halfLen);
    for (let k = 0; k < halfLen; k++) {
        let sumLo = 0;
        let sumHi = 0;
        for (let j = 0; j < L; j++) {
            const val = symRef(signal, 2 * k + j);
            sumLo += lo[j] * val;
            sumHi += hi[j] * val;
        }
        approx[k] = sumLo;
        detail[k] = sumHi;
    }
    return { approx, detail };
}
/**
 * 一维逆 DWT（重建一层）。
 *
 * 与前向相关约定配套的 **转置重建** 公式：
 *   x[n] = Σ_k lo[n-2k] · cA[k] + Σ_k hi[n-2k] · cD[k]
 *          其中仅对 n-2k ∈ [0, L-1] 的 k 求和
 *
 * 使用分析滤波器 (lo_d, hi_d)，因为正变换使用的是相关而非卷积。
 * 边界通过 symRef 对系数数组做对称反射处理。
 */
function idwt1d(approx, detail, lo, hi, outLen) {
    const L = lo.length;
    const result = new Array(outLen);
    for (let n = 0; n < outLen; n++) {
        let sum = 0;
        for (let j = 0; j < L; j++) {
            // 滤波器索引 j = n - 2k  →  2k = n - j  →  k = (n - j) / 2
            // k 可以为负数，边界外的系数值由 symRef 处理
            const diff = n - j;
            if ((diff & 1) !== 0)
                continue; // 必须是偶数
            const k = diff >> 1;
            sum += lo[j] * symRef(approx, k) + hi[j] * symRef(detail, k);
        }
        result[n] = sum;
    }
    return result;
}
// ──────────────────────────── 2D DWT / IDWT ───────────────────────────────
/** 转置一个 number[][] 矩阵 */
function transposeArr(mat) {
    const rows = mat.length;
    const cols = mat[0]?.length ?? 0;
    const out = Array.from({ length: cols }, () => new Array(rows));
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            out[j][i] = mat[i][j];
        }
    }
    return out;
}
/**
 * 二维前向 DWT（单层）。
 *
 * 步骤：
 * 1. 对每一行做 1D DWT → [L 行 | H 行]
 * 2. 对结果的每一列做 1D DWT → 四个子带 LL, LH, HL, HH
 */
function dwt2dSingle(input, filter) {
    const rows = input.length;
    const cols = input[0]?.length ?? 0;
    const { lo_d, hi_d } = filter;
    const halfCols = Math.ceil(cols / 2);
    // 步骤 1：按行变换
    const rowLo = new Array(rows); // 各行的低频
    const rowHi = new Array(rows); // 各行的高频
    for (let r = 0; r < rows; r++) {
        const { approx, detail } = dwt1d(input[r], lo_d, hi_d);
        rowLo[r] = approx;
        rowHi[r] = detail;
    }
    // 步骤 2：按列变换
    // 对 rowLo 转置后按行做 DWT（等效按列做 DWT）
    const rowLoT = transposeArr(rowLo); // halfCols × rows
    const rowHiT = transposeArr(rowHi); // halfCols × rows
    const halfRows = Math.ceil(rows / 2);
    const LL = new Array(halfRows);
    const LH = new Array(halfRows);
    const HL = new Array(halfRows);
    const HH = new Array(halfRows);
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
 * 二维逆 DWT（单层），从四个子带重建出原矩阵。
 */
function idwt2dSingle(level, filter) {
    const { LL, LH, HL, HH, origRows, origCols } = level;
    // 反变换使用分析滤波器（与相关约定的正变换配套）
    const { lo_d, hi_d } = filter;
    const halfRows = LL.length;
    const halfCols = LL[0]?.length ?? 0;
    // 步骤 1：按列逆变换，恢复行数
    // 对每一列位置，将 LL/LH 列合并重建低频行部分，HL/HH 列合并重建高频行部分
    const rowLo = Array.from({ length: origRows }, () => new Array(halfCols));
    const rowHi = Array.from({ length: origRows }, () => new Array(halfCols));
    for (let c = 0; c < halfCols; c++) {
        const ll_col = new Array(halfRows);
        const lh_col = new Array(halfRows);
        const hl_col = new Array(halfRows);
        const hh_col = new Array(halfRows);
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
    // 步骤 2：按行逆变换，恢复列数
    const result = new Array(origRows);
    for (let r = 0; r < origRows; r++) {
        result[r] = idwt1d(rowLo[r], rowHi[r], lo_d, hi_d, origCols);
    }
    return result;
}
/**
 * 多层二维前向 DWT 分解。
 * 对 LL 子带递归分解 `levels` 层。
 */
function dwtDecompose2D(input, wavelet, levels) {
    const filter = getFilter(wavelet);
    const detailLevels = [];
    let current = input;
    for (let l = 0; l < levels; l++) {
        // 检查当前矩阵是否足以继续分解（至少 2×2）
        if (current.length < 2 || (current[0]?.length ?? 0) < 2)
            break;
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
 * 从多层分解结果逆向重建完整矩阵。
 * 从最粗层（最深层）开始，逐层向上 IDWT。
 */
function dwtReconstruct2D(decomp, wavelet) {
    const filter = getFilter(wavelet);
    let current = decomp.approx;
    // 从最深层向外逐层重建
    for (let l = decomp.detailLevels.length - 1; l >= 0; l--) {
        const dl = decomp.detailLevels[l];
        const level = {
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
// ─────────────────────────────── 阈值化 ──────────────────────────────────
/**
 * 硬阈值：|x| < λ 的系数直接置零，其余保持不变。
 * 保留原始系数幅度，重建更锐利但可能有振铃。
 */
function hardThreshold(mat, lambda) {
    return mat.map(row => row.map(v => Math.abs(v) < lambda ? 0 : v));
}
/**
 * 软阈值：|x| < λ 置零，否则向零收缩 λ。
 * sign(x) · max(0, |x| - λ)
 * 重建更平滑，视觉上更"柔和"。
 */
function softThreshold(mat, lambda) {
    return mat.map(row => row.map(v => {
        const abs = Math.abs(v);
        if (abs < lambda)
            return 0;
        return Math.sign(v) * (abs - lambda);
    }));
}
/**
 * 对分解结果的所有细节子带执行阈值化。
 * 注意：LL（近似子带）不做阈值化，它保存图像的主体能量。
 */
function applyThreshold(decomp, threshold, mode) {
    const threshFn = mode === 'hard' ? hardThreshold : softThreshold;
    const newDetails = decomp.detailLevels.map(dl => ({
        LH: threshFn(dl.LH, threshold),
        HL: threshFn(dl.HL, threshold),
        HH: threshFn(dl.HH, threshold),
        origRows: dl.origRows,
        origCols: dl.origCols,
    }));
    return {
        approx: decomp.approx, // LL 子带不做阈值化
        detailLevels: newDetails,
    };
}
// ───────────────────────── 压缩率辅助计算 ─────────────────────────────────
/**
 * 统计分解结果中非零系数的个数，用于估算压缩率。
 */
function countNonZeroCoefficients(decomp) {
    let total = 0;
    let nonZero = 0;
    const countMat = (mat) => {
        for (const row of mat) {
            for (const v of row) {
                total++;
                if (v !== 0)
                    nonZero++;
            }
        }
    };
    // LL 近似子带
    countMat(decomp.approx);
    // 各层细节子带
    for (const dl of decomp.detailLevels) {
        countMat(dl.LH);
        countMat(dl.HL);
        countMat(dl.HH);
    }
    return { total, nonZero };
}
// ──────────────────────────── 主入口函数 ──────────────────────────────────
/**
 * DWT 图像压缩主函数。
 *
 * 流程：
 * 1. 将 ImageData 拆分为 R/G/B 三个 number[][] 矩阵
 * 2. 对每个通道执行多层 2D DWT 分解
 * 3. 对细节系数做阈值化（压缩）
 * 4. 逆 DWT 重建各通道
 * 5. 合并通道为 ImageData 输出
 *
 * @param imageData  原始图像
 * @param params     DWT 参数（小波基、层数、阈值强度、阈值模式）
 * @returns          压缩结果（重建图像 + 压缩率等信息）
 */
export function compressImageByDWT(imageData, params) {
    const { wavelet, levels, threshold, thresholdMode } = params;
    // 防御式拷贝
    const safe = new ImageData(new Uint8ClampedArray(imageData.data), imageData.width, imageData.height);
    const { r, g, b, width, height } = imageDataToRGBMatrices(safe);
    // 限制分解层数：不能超过 log2(min(width, height))
    const maxLevels = Math.max(1, Math.floor(Math.log2(Math.min(width, height))));
    const safeLevels = Math.min(levels, maxLevels);
    // 对 R/G/B 三通道分别做 DWT → 阈值化 → IDWT
    const processChannel = (channel) => {
        const decomp = dwtDecompose2D(channel, wavelet, safeLevels);
        const thresholded = applyThreshold(decomp, threshold, thresholdMode);
        const { total, nonZero } = countNonZeroCoefficients(thresholded);
        const reconstructed = dwtReconstruct2D(thresholded, wavelet);
        // 裁剪到 [0, 255]
        const clamped = reconstructed.map(row => row.map(v => Math.max(0, Math.min(255, Math.round(v)))));
        return { result: clamped, nonZero, total };
    };
    const rResult = processChannel(r);
    const gResult = processChannel(g);
    const bResult = processChannel(b);
    const totalCoeffs = rResult.total + gResult.total + bResult.total;
    const nonZeroCoeffs = rResult.nonZero + gResult.nonZero + bResult.nonZero;
    const reconstructedImageData = rgbMatricesToImageData(rResult.result, gResult.result, bResult.result, width, height);
    // 压缩率 = 总系数数 / 非零系数数
    const ratio = nonZeroCoeffs > 0 ? totalCoeffs / nonZeroCoeffs : Infinity;
    return {
        imageData: reconstructedImageData,
        rankUsed: safeLevels,
        estimatedCompressionRatio: ratio,
        method: 'dwt',
        notes: `${wavelet.toUpperCase()} · ${safeLevels}层 · λ=${threshold} · ${thresholdMode === 'hard' ? '硬阈值' : '软阈值'} · 非零系数 ${nonZeroCoeffs}/${totalCoeffs}`,
    };
}
