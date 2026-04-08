import type { Matrix, SVDResult } from '../types';
import { cloneMatrix, multiply, transpose, zeros } from './matrix';

const EPS = Number.EPSILON;

/** SVD 计算模式 */
export type SVDMode = 'full' | 'randomized';

export interface SVDOptions {
    copyInput?: boolean;
    /** 'full' = Householder 双对角化 + Wilkinson 迭代；'randomized' = 随机 SVD */
    mode?: SVDMode;
    /** 随机 SVD 的目标秩（仅 mode='randomized' 时有效） */
    rank?: number;
    /** 随机 SVD 的过采样数（默认 10） */
    oversampling?: number;
    /** 随机 SVD 的幂迭代次数（默认 1） */
    powerIterations?: number;
}

function identity(n: number): Matrix {
    const out = zeros(n, n);
    for (let i = 0; i < n; i += 1) out[i][i] = 1;
    return out;
}

function signNonZero(x: number): number {
    return x >= 0 ? 1 : -1;
}

/**
 * 算法 3.2.1（Householder 向量）
 * 输入向量 x，输出反射向量 v 和系数 beta。
 */
function house(xInput: number[]): { v: number[]; beta: number } {
    const n = xInput.length;
    const v = new Array<number>(n).fill(0);
    if (n === 0) return { v, beta: 0 };

    let eta = 0;
    for (let i = 0; i < n; i += 1) {
        eta = Math.max(eta, Math.abs(xInput[i]));
    }
    if (eta === 0) {
        return { v, beta: 0 };
    }

    const x = xInput.map((xi) => xi / eta);

    let sigma = 0;
    for (let i = 1; i < n; i += 1) {
        sigma += x[i] * x[i];
        v[i] = x[i];
    }

    if (sigma === 0) {
        return { v, beta: 0 };
    }

    const alpha = Math.sqrt(x[0] * x[0] + sigma);
    if (x[0] <= 0) {
        v[0] = x[0] - alpha;
    } else {
        v[0] = -sigma / (x[0] + alpha);
    }

    const denom = sigma + v[0] * v[0];
    const beta = (2 * v[0] * v[0]) / denom;
    const scale = v[0];
    for (let i = 0; i < n; i += 1) {
        v[i] /= scale;
    }

    return { v, beta };
}

/** U <- U * H，其中 H = I - beta * v * v^T，作用在列区间 start.. */
function applyHouseholderToRight(orth: Matrix, start: number, v: number[], beta: number): void {
    if (beta === 0) return;
    const rows = orth.length;
    const len = v.length;
    for (let i = 0; i < rows; i += 1) {
        let dot = 0;
        for (let t = 0; t < len; t += 1) {
            dot += orth[i][start + t] * v[t];
        }
        const coeff = beta * dot;
        for (let t = 0; t < len; t += 1) {
            orth[i][start + t] -= coeff * v[t];
        }
    }
}

/**
 * mat <- mat * G^T，其中 G = [c s; -s c]。
 * 作用在列 p, q。
 */
function applyRightRotation(mat: Matrix, p: number, q: number, c: number, s: number): void {
    const rows = mat.length;
    for (let i = 0; i < rows; i += 1) {
        const a = mat[i][p];
        const b = mat[i][q];
        mat[i][p] = c * a + s * b;
        mat[i][q] = -s * a + c * b;
    }
}

/**
 * mat <- G * mat，其中 G = [c s; -s c]。
 * 作用在行 p, q。
 */
function applyLeftRotation(mat: Matrix, p: number, q: number, c: number, s: number): void {
    const cols = mat[0]?.length ?? 0;
    for (let j = 0; j < cols; j += 1) {
        const a = mat[p][j];
        const b = mat[q][j];
        mat[p][j] = c * a + s * b;
        mat[q][j] = -s * a + c * b;
    }
}

/**
 * Givens 参数：返回 (c, s, r)，使 G * [a; b] = [r; 0]。
 */
function givens(a: number, b: number): { c: number; s: number; r: number } {
    if (b === 0) {
        return { c: 1, s: 0, r: a };
    }
    if (a === 0) {
        return { c: 0, s: 1, r: b };
    }
    const r = Math.hypot(a, b);
    return { c: a / r, s: b / r, r };
}

function normInf(mat: Matrix): number {
    let maxRowSum = 0;
    for (let i = 0; i < mat.length; i += 1) {
        let rowSum = 0;
        for (let j = 0; j < mat[i].length; j += 1) {
            rowSum += Math.abs(mat[i][j]);
        }
        maxRowSum = Math.max(maxRowSum, rowSum);
    }
    return maxRowSum;
}

/**
 * 清理双对角外的小噪声：仅保留主对角 (i,i) 与超对角 (i,i+1)。
 */
function cleanBidiagonalNoise(B: Matrix, eps: number): void {
    const n = B.length;
    for (let i = 0; i < n; i += 1) {
        for (let j = 0; j < n; j += 1) {
            if (j !== i && j !== i + 1 && Math.abs(B[i][j]) <= eps) {
                B[i][j] = 0;
            }
        }
    }
}

/**
 * 课本算法 7.6.3 第 (3) 步收敛判定：
 * (i) 小超对角元置零；(ii) 小对角元置零。
 */
function textbookConvergenceSweep(B: Matrix, eps: number): void {
    const n = B.length;
    const diagTol = normInf(B) * eps;

    for (let i = 0; i < n - 1; i += 1) {
        const tol = (Math.abs(B[i][i]) + Math.abs(B[i + 1][i + 1])) * eps;
        if (Math.abs(B[i][i + 1]) <= tol) {
            B[i][i + 1] = 0;
        }
        if (Math.abs(B[i + 1][i]) <= tol) {
            B[i + 1][i] = 0;
        }
    }

    for (let i = 0; i < n; i += 1) {
        if (Math.abs(B[i][i]) <= diagTol) {
            B[i][i] = 0;
        }
    }
}

function isDiagonalized(B: Matrix): boolean {
    const n = B.length;
    for (let i = 0; i < n - 1; i += 1) {
        if (B[i][i + 1] !== 0) return false;
    }
    return true;
}

/**
 * 找到 B22 的活跃子块区间 [l, r]。
 * 若已完全对角化，返回 null。
 */
function findActiveBlock(B: Matrix): { l: number; r: number } | null {
    const n = B.length;

    let r = -1;
    for (let i = n - 2; i >= 0; i -= 1) {
        if (B[i][i + 1] !== 0) {
            r = i + 1;
            break;
        }
    }
    if (r < 0) return null;

    let l = r - 1;
    while (l > 0 && B[l - 1][l] !== 0) {
        l -= 1;
    }

    return { l, r };
}

/**
 * 算法 7.6.3 第 (4)(i) 步：
 * 若 B22 中出现零对角元（最后一个除外），用 Givens 追赶消元。
 */
function deflateZeroDiagonal(B: Matrix, P: Matrix, l: number, r: number, eps: number): boolean {
    const diagTol = normInf(B) * eps;

    for (let i = l; i < r; i += 1) {
        if (Math.abs(B[i][i]) > diagTol) continue;

        for (let j = i + 1; j <= r; j += 1) {
            if (B[i][j] === 0) continue;
            const g = givens(B[j][j], B[i][j]);
            applyLeftRotation(B, j, i, g.c, g.s);
            applyRightRotation(P, j, i, g.c, g.s);
        }

        B[i][i] = 0;
        return true;
    }

    return false;
}

/**
 * 课本算法 7.6.2：对活跃子块 B(l:r,l:r) 执行一次带 Wilkinson 位移的 SVD 迭代。
 */
function wilkinsonSvdStep(B: Matrix, P: Matrix, Q: Matrix, l: number, r: number): void {
    if (r <= l) return;

    const alpha = B[r][r] * B[r][r] + B[r - 1][r] * B[r - 1][r];
    const gammaPrev = r - 2 >= l ? B[r - 2][r - 1] : 0;
    const delta = (B[r - 1][r - 1] * B[r - 1][r - 1] + gammaPrev * gammaPrev - alpha) / 2;
    const beta = B[r - 1][r - 1] * B[r - 1][r];

    const denom = delta + signNonZero(delta) * Math.sqrt(delta * delta + beta * beta);
    const mu = Math.abs(denom) <= EPS ? alpha : alpha - (beta * beta) / denom;

    let y = B[l][l] * B[l][l] - mu;
    let z = B[l][l] * B[l][l + 1];

    for (let k = l; k < r; k += 1) {
        const right = givens(y, z);
        applyRightRotation(B, k, k + 1, right.c, right.s);
        applyRightRotation(Q, k, k + 1, right.c, right.s);

        const left = givens(B[k][k], B[k + 1][k]);
        applyLeftRotation(B, k, k + 1, left.c, left.s);
        applyRightRotation(P, k, k + 1, left.c, left.s);

        if (k < r - 1) {
            y = B[k][k + 1];
            z = B[k][k + 2];
        }
    }
}

/**
 * 课本算法 7.6.3：
 * 在双对角矩阵 B 上迭代，得到 B = P^T * B * Q 的近似对角化结果。
 */
function diagWithWilkinson(B: Matrix): { P: Matrix; Q: Matrix } {
    const n = B.length;
    const P = identity(n);
    const Q = identity(n);

    const eps = Math.sqrt(EPS);
    const maxIter = Math.max(400 * n * n, 4000);

    for (let iter = 0; iter < maxIter; iter += 1) {
        textbookConvergenceSweep(B, eps);
        cleanBidiagonalNoise(B, normInf(B) * eps);

        if (isDiagonalized(B)) {
            return { P, Q };
        }

        const active = findActiveBlock(B);
        if (!active) {
            return { P, Q };
        }

        const { l, r } = active;

        if (deflateZeroDiagonal(B, P, l, r, eps)) {
            cleanBidiagonalNoise(B, normInf(B) * eps);
            continue;
        }

        wilkinsonSvdStep(B, P, Q, l, r);
        cleanBidiagonalNoise(B, normInf(B) * eps);
    }

    throw new Error('SVD 迭代未在最大次数内收敛。');
}

// ---------------- 全 SVD（Householder 双对角化 + Wilkinson 迭代） ----------------

function computeSVDTall(input: Matrix): SVDResult {
    const A = cloneMatrix(input);
    const m = A.length;
    const n = A[0]?.length ?? 0;
    if (m === 0 || n === 0) {
        return { U: [], S: [], Vt: [] };
    }

    // 算法 7.6.1：双对角化，并累计正交矩阵 U、V。
    const Ufull = identity(m);
    const V = identity(n);

    for (let k = 0; k < n; k += 1) {
        const x = new Array<number>(m - k);
        for (let i = k; i < m; i += 1) {
            x[i - k] = A[i][k];
        }
        const leftHouse = house(x);

        if (leftHouse.beta !== 0) {
            for (let j = k; j < n; j += 1) {
                let dot = 0;
                for (let i = 0; i < leftHouse.v.length; i += 1) {
                    dot += leftHouse.v[i] * A[k + i][j];
                }
                const coeff = leftHouse.beta * dot;
                for (let i = 0; i < leftHouse.v.length; i += 1) {
                    A[k + i][j] -= coeff * leftHouse.v[i];
                }
            }

            applyHouseholderToRight(Ufull, k, leftHouse.v, leftHouse.beta);
        }

        if (k < n - 1) {
            const xRow = new Array<number>(n - (k + 1));
            for (let j = k + 1; j < n; j += 1) {
                xRow[j - (k + 1)] = A[k][j];
            }
            const rightHouse = house(xRow);

            if (rightHouse.beta !== 0) {
                for (let i = k; i < m; i += 1) {
                    let dot = 0;
                    for (let t = 0; t < rightHouse.v.length; t += 1) {
                        dot += A[i][k + 1 + t] * rightHouse.v[t];
                    }
                    const coeff = rightHouse.beta * dot;
                    for (let t = 0; t < rightHouse.v.length; t += 1) {
                        A[i][k + 1 + t] -= coeff * rightHouse.v[t];
                    }
                }

                applyHouseholderToRight(V, k + 1, rightHouse.v, rightHouse.beta);
            }
        }
    }

    // 提取 n x n 上双对角块 B。
    const B = zeros(n, n);
    for (let i = 0; i < n; i += 1) {
        B[i][i] = A[i][i];
        if (i < n - 1) {
            B[i][i + 1] = A[i][i + 1];
        }
    }

    const { P, Q } = diagWithWilkinson(B);

    // U = U * diag(P, I_{m-n})，仅保留前 n 列。
    const Uleft = zeros(m, n);
    for (let i = 0; i < m; i += 1) {
        for (let j = 0; j < n; j += 1) {
            Uleft[i][j] = Ufull[i][j];
        }
    }
    let U = multiply(Uleft, P);
    let Vmat = multiply(V, Q);

    const S = new Array<number>(n);
    for (let i = 0; i < n; i += 1) {
        let sigma = B[i][i];
        if (sigma < 0) {
            sigma = -sigma;
            for (let r = 0; r < U.length; r += 1) {
                U[r][i] = -U[r][i];
            }
        }
        S[i] = sigma;
    }

    // 奇异值按降序排列，并同步重排 U/V。
    const order = S.map((sv, idx) => ({ sv, idx })).sort((a, b) => b.sv - a.sv).map((e) => e.idx);

    const Usorted = U.map((row) => order.map((idx) => row[idx]));
    const Ssorted = order.map((idx) => S[idx]);
    Vmat = Vmat.map((row) => order.map((idx) => row[idx]));

    return {
        U: Usorted,
        S: Ssorted,
        Vt: transpose(Vmat),
    };
}

// ---------------- 随机 SVD（用于可选快速近似） ----------------

/** Box-Muller 变换生成标准正态随机数 */
function randn(): number {
    let u1 = 0;
    let u2 = 0;
    while (u1 === 0) u1 = Math.random();
    while (u2 === 0) u2 = Math.random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/** 生成 m x n 的高斯随机矩阵 */
function gaussianRandom(m: number, n: number): Matrix {
    const out = zeros(m, n);
    for (let i = 0; i < m; i += 1) {
        for (let j = 0; j < n; j += 1) {
            out[i][j] = randn();
        }
    }
    return out;
}

/**
 * Householder 版 thin QR：A = Q * R
 * A: m x n (m >= n)，返回 Q: m x n, R: n x n
 */
function thinQR(Ainput: Matrix): { Q: Matrix; R: Matrix } {
    const m = Ainput.length;
    const n = Ainput[0]?.length ?? 0;
    const R = cloneMatrix(Ainput);
    const Qfull = identity(m);

    for (let k = 0; k < n; k += 1) {
        const x = new Array<number>(m - k);
        for (let i = k; i < m; i += 1) x[i - k] = R[i][k];
        const { v, beta } = house(x);
        if (beta === 0) continue;

        for (let j = k; j < n; j += 1) {
            let dot = 0;
            for (let i = 0; i < v.length; i += 1) {
                dot += v[i] * R[k + i][j];
            }
            const coeff = beta * dot;
            for (let i = 0; i < v.length; i += 1) {
                R[k + i][j] -= coeff * v[i];
            }
        }

        applyHouseholderToRight(Qfull, k, v, beta);
    }

    const Q = zeros(m, n);
    for (let i = 0; i < m; i += 1) {
        for (let j = 0; j < n; j += 1) {
            Q[i][j] = Qfull[i][j];
        }
    }

    const Rthin = zeros(n, n);
    for (let i = 0; i < n; i += 1) {
        for (let j = i; j < n; j += 1) {
            Rthin[i][j] = R[i][j];
        }
    }

    return { Q, R: Rthin };
}

/** 高效计算 A^T * B，避免显式转置 */
function multiplyAtB(A: Matrix, B: Matrix): Matrix {
    const m = A.length;
    const nA = A[0]?.length ?? 0;
    const nB = B[0]?.length ?? 0;
    const out = zeros(nA, nB);
    for (let i = 0; i < nA; i += 1) {
        for (let k = 0; k < m; k += 1) {
            const aki = A[k][i];
            if (aki === 0) continue;
            for (let j = 0; j < nB; j += 1) {
                out[i][j] += aki * B[k][j];
            }
        }
    }
    return out;
}

/**
 * 随机 SVD（Halko-Martinsson-Tropp, 2011）
 * 返回矩阵 A 的秩-k 近似分解。
 */
function computeRandomizedSVD(
    input: Matrix,
    targetRank: number,
    oversampling: number = 10,
    powerIter: number = 1,
): SVDResult {
    const A = cloneMatrix(input);
    const m = A.length;
    const n = A[0]?.length ?? 0;

    if (m === 0 || n === 0) {
        return { U: [], S: [], Vt: [] };
    }

    const k = Math.min(targetRank, Math.min(m, n));
    const p = Math.min(oversampling, Math.min(m, n) - k);
    const l = k + p;

    if (l >= Math.min(m, n)) {
        return computeSVDFull(input);
    }

    const omega = gaussianRandom(n, l);
    let Y = multiply(A, omega);

    for (let q = 0; q < powerIter; q += 1) {
        const qr1 = thinQR(Y);
        const Z = multiplyAtB(A, qr1.Q);
        const qr2 = thinQR(Z);
        Y = multiply(A, qr2.Q);
    }

    const { Q } = thinQR(Y);
    const B = multiplyAtB(Q, A);
    const svdB = computeSVDFull(B);

    const Ub = svdB.U;
    const Ufinal = multiply(Q, Ub);

    const Utrunc = Ufinal.map((row) => row.slice(0, k));
    const Strunc = svdB.S.slice(0, k);
    const VtTrunc = svdB.Vt.slice(0, k);

    return {
        U: Utrunc,
        S: Strunc,
        Vt: VtTrunc,
    };
}

// ---------------- 对外入口 ----------------

/**
 * 全 SVD（内部函数，自动处理 m < n 的情况）
 */
function computeSVDFull(input: Matrix): SVDResult {
    const A = cloneMatrix(input);
    const m = A.length;
    const n = A[0]?.length ?? 0;

    if (m === 0 || n === 0) {
        return { U: [], S: [], Vt: [] };
    }

    if (m < n) {
        const transposed = transpose(A);
        const svdAt = computeSVDTall(transposed);
        return {
            U: transpose(svdAt.Vt),
            S: svdAt.S,
            Vt: transpose(svdAt.U),
        };
    }

    return computeSVDTall(A);
}

/**
 * 作业要求：SVD 必须手动实现，不调用现成库。
 * 支持两种模式：
 * - full：课本流程（Householder 双对角化 + Wilkinson 迭代）
 * - randomized：随机 SVD 近似
 */
export function computeSVD(input: Matrix, options?: SVDOptions): SVDResult {
    const shouldCopy = options?.copyInput ?? true;
    const A = shouldCopy ? cloneMatrix(input) : input;
    const mode = options?.mode ?? 'full';

    if (mode === 'randomized') {
        const m = A.length;
        const n = A[0]?.length ?? 0;
        const rank = options?.rank ?? Math.max(1, Math.min(30, Math.min(m, n)));
        const oversampling = options?.oversampling ?? 10;
        const powerIter = options?.powerIterations ?? 1;
        return computeRandomizedSVD(A, rank, oversampling, powerIter);
    }

    return computeSVDFull(A);
}
