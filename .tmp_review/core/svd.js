import { cloneMatrix, multiply, transpose, zeros } from './matrix';
const EPS = Number.EPSILON;
function identity(n) {
    const out = zeros(n, n);
    for (let i = 0; i < n; i += 1)
        out[i][i] = 1;
    return out;
}
function signNonZero(x) {
    return x >= 0 ? 1 : -1;
}
/**
 * 算法 3.2.1（house）
 * 输入向量 x，输出 Householder 向量 v 与系数 beta。
 */
function house(xInput) {
    const n = xInput.length;
    const v = new Array(n).fill(0);
    if (n === 0)
        return { v, beta: 0 };
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
    }
    else {
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
/** U <- U * H，其中 H = I - beta * v * v^T 作用在列区间 start.. */
function applyHouseholderToRight(orth, start, v, beta) {
    if (beta === 0)
        return;
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
 * mat <- mat * G^T，其中 G = [c s; -s c]，G^T = [c -s; s c]
 * 作用在列 p, q。
 *
 * givens(a, b) 返回的 (c, s) 满足 G * [a; b] = [r; 0]。
 * 对应右乘 G^T 时：[a, b] * G^T = [a*c+b*s, -a*s+b*c]，
 * 当 (a,b) = givens 的输入时第二分量被消零。
 */
function applyRightRotation(mat, p, q, c, s) {
    const rows = mat.length;
    for (let i = 0; i < rows; i += 1) {
        const a = mat[i][p];
        const b = mat[i][q];
        mat[i][p] = c * a + s * b;
        mat[i][q] = -s * a + c * b;
    }
}
/**
 * mat <- G * mat，其中 G = [c s; -s c]
 * 作用在行 p, q。
 *
 * G * [a; b] = [c*a + s*b; -s*a + c*b]，
 * 当 (a,b) = givens 的输入时第二分量被消零。
 */
function applyLeftRotation(mat, p, q, c, s) {
    const cols = mat[0]?.length ?? 0;
    for (let j = 0; j < cols; j += 1) {
        const a = mat[p][j];
        const b = mat[q][j];
        mat[p][j] = c * a + s * b;
        mat[q][j] = -s * a + c * b;
    }
}
/**
 * 计算 Givens 旋转参数。
 * 返回 (c, s, r) 使得 G * [a; b] = [r; 0]，其中 G = [c s; -s c]。
 */
function givens(a, b) {
    if (b === 0) {
        return { c: 1, s: 0, r: a };
    }
    if (a === 0) {
        return { c: 0, s: 1, r: b };
    }
    const r = Math.hypot(a, b);
    return { c: a / r, s: b / r, r };
}
function normInf(mat) {
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
 * 清理上双对角矩阵中的数值噪声：
 * 仅保留主对角线 (i,i) 和超对角线 (i,i+1)，其余位置若小于 eps 则置零。
 */
function cleanBidiagonalNoise(B, eps) {
    const n = B.length;
    for (let i = 0; i < n; i += 1) {
        for (let j = 0; j < n; j += 1) {
            if (j !== i && j !== i + 1) {
                if (Math.abs(B[i][j]) <= eps)
                    B[i][j] = 0;
            }
        }
    }
}
/**
 * 零对角元素 deflation：
 * 当活跃区间 [l, r] 中某个对角元素 B[i][i] ≈ 0 时，
 * 使用 Givens 旋转消除该行的超对角元素 B[i][i+1]，
 * 从而将问题分解为更小的子问题。
 * 返回 true 表示执行了 deflation（应跳过本轮 Wilkinson step）。
 */
function deflateZeroDiagonal(B, P, l, r, eps) {
    for (let i = l; i < r; i += 1) {
        if (Math.abs(B[i][i]) <= eps) {
            // B[i][i] ≈ 0，用左旋转依次消除 B[i][i+1], B[i][i+2], ..., B[i][r]
            for (let j = i + 1; j <= r; j += 1) {
                if (B[i][j] === 0)
                    continue;
                const g = givens(B[j][j], B[i][j]);
                // 左旋转作用在行 j, i（注意顺序：零化 B[i][j]）
                applyLeftRotation(B, j, i, g.c, g.s);
                applyRightRotation(P, j, i, g.c, g.s);
            }
            B[i][i] = 0;
            return true;
        }
    }
    return false;
}
/**
 * 算法 7.6.2 的 Wilkinson 位移思想，作用于 B(l:r,l:r)。
 */
function wilkinsonSvdStep(B, P, Q, l, r) {
    if (r <= l)
        return;
    const alpha = B[r][r] * B[r][r] + B[r - 1][r] * B[r - 1][r];
    const gammaPrev = r - 2 >= l ? B[r - 2][r - 1] : 0;
    const delta = (B[r - 1][r - 1] * B[r - 1][r - 1] + gammaPrev * gammaPrev - alpha) / 2;
    const beta = B[r - 1][r - 1] * B[r - 1][r];
    const denom = delta + signNonZero(delta) * Math.sqrt(delta * delta + beta * beta);
    const mu = alpha - (Math.abs(denom) < EPS ? 0 : (beta * beta) / denom);
    let y = B[l][l] * B[l][l] - mu;
    let z = B[l][l] * B[l][l + 1];
    for (let k = l; k < r; k += 1) {
        // 右旋转消零 [y, z] 中的 z（作用在列 k, k+1）
        const right = givens(y, z);
        applyRightRotation(B, k, k + 1, right.c, right.s);
        applyRightRotation(Q, k, k + 1, right.c, right.s);
        // 左旋转消零由右旋转产生的下对角元素 B[k+1][k]
        const left = givens(B[k][k], B[k + 1][k]);
        applyLeftRotation(B, k, k + 1, left.c, left.s);
        applyRightRotation(P, k, k + 1, left.c, left.s);
        if (k < r - 1) {
            y = B[k][k + 1];
            z = B[k][k + 2];
        }
    }
}
function allSuperDiagonalSmall(B, eps) {
    const n = B.length;
    for (let i = 0; i < n - 1; i += 1) {
        const tol = eps * (Math.abs(B[i][i]) + Math.abs(B[i + 1][i + 1]) + 1);
        if (Math.abs(B[i][i + 1]) > tol)
            return false;
    }
    return true;
}
function diagWithWilkinson(B) {
    const n = B.length;
    const P = identity(n);
    const Q = identity(n);
    const scale = Math.max(1, normInf(B));
    const eps = Math.sqrt(EPS) * scale;
    const maxIter = Math.max(80 * n * n, 2000);
    for (let iter = 0; iter < maxIter; iter += 1) {
        // 收敛检查：将足够小的超对角/下对角置零
        for (let i = 0; i < n - 1; i += 1) {
            const tol = eps * (Math.abs(B[i][i]) + Math.abs(B[i + 1][i + 1]) + 1);
            if (Math.abs(B[i][i + 1]) <= tol) {
                B[i][i + 1] = 0;
            }
            if (Math.abs(B[i + 1][i]) <= tol) {
                B[i + 1][i] = 0;
            }
        }
        if (allSuperDiagonalSmall(B, eps)) {
            break;
        }
        // 找活跃子矩阵 [l, r]：从底部找第一个非零超对角
        let r = -1;
        for (let i = n - 2; i >= 0; i -= 1) {
            if (B[i][i + 1] !== 0) {
                r = i + 1;
                break;
            }
        }
        if (r === -1)
            break;
        let l = r - 1;
        while (l > 0 && B[l - 1][l] !== 0) {
            l -= 1;
        }
        // 零对角 deflation（Golub-Van Loan §8.6.2）
        if (deflateZeroDiagonal(B, P, l, r, eps)) {
            cleanBidiagonalNoise(B, eps);
            continue;
        }
        wilkinsonSvdStep(B, P, Q, l, r);
        cleanBidiagonalNoise(B, eps);
    }
    if (!allSuperDiagonalSmall(B, eps * 10)) {
        throw new Error('SVD 迭代未在限定步数内收敛，请尝试更小图像或先降采样。');
    }
    return { P, Q };
}
// ─── 全 SVD（Householder 双对角化 + Wilkinson 迭代） ───────────────────────────
function computeSVDTall(input) {
    const A = cloneMatrix(input);
    const m = A.length;
    const n = A[0]?.length ?? 0;
    if (m === 0 || n === 0) {
        return { U: [], S: [], Vt: [] };
    }
    // 算法 7.6.1：Householder 二对角化，累积 U、V。
    const Ufull = identity(m);
    const V = identity(n);
    for (let k = 0; k < n; k += 1) {
        const x = new Array(m - k);
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
            const xRow = new Array(n - (k + 1));
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
    // 从二对角部分提取 B（n x n）
    const B = zeros(n, n);
    for (let i = 0; i < n; i += 1) {
        B[i][i] = A[i][i];
        if (i < n - 1) {
            B[i][i + 1] = A[i][i + 1];
        }
    }
    // 算法 7.6.2 + 7.6.3：Wilkinson 位移迭代对角化
    const { P, Q } = diagWithWilkinson(B);
    // U = U * diag(P, I_{m-n})，仅取前 n 列
    const Uleft = zeros(m, n);
    for (let i = 0; i < m; i += 1) {
        for (let j = 0; j < n; j += 1) {
            Uleft[i][j] = Ufull[i][j];
        }
    }
    let U = multiply(Uleft, P);
    let Vmat = multiply(V, Q);
    const S = new Array(n);
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
    // 按奇异值从大到小排序（同步重排 U, V^T）
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
// ─── 随机 SVD（Halko-Martinsson-Tropp, 2011）──────────────────────────────────
/** Box-Muller 变换生成标准正态随机数 */
function randn() {
    let u1 = 0;
    let u2 = 0;
    while (u1 === 0)
        u1 = Math.random();
    while (u2 === 0)
        u2 = Math.random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}
/** 生成 m x n 的高斯随机矩阵 */
function gaussianRandom(m, n) {
    const out = zeros(m, n);
    for (let i = 0; i < m; i += 1) {
        for (let j = 0; j < n; j += 1) {
            out[i][j] = randn();
        }
    }
    return out;
}
/**
 * Householder 薄 QR 分解：A = Q * R
 * A: m x n (m >= n)，返回 Q: m x n, R: n x n
 */
function thinQR(Ainput) {
    const m = Ainput.length;
    const n = Ainput[0]?.length ?? 0;
    const R = cloneMatrix(Ainput);
    const Qfull = identity(m);
    for (let k = 0; k < n; k += 1) {
        const x = new Array(m - k);
        for (let i = k; i < m; i += 1)
            x[i - k] = R[i][k];
        const { v, beta } = house(x);
        if (beta === 0)
            continue;
        // R(k:m, k:n) <- (I - beta*v*v^T) * R(k:m, k:n)
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
        // 累积 Q = Q * H_k
        applyHouseholderToRight(Qfull, k, v, beta);
    }
    // 提取 thin Q (m x n) 和 upper-triangular R (n x n)
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
/** mat^T * mat2 的高效计算，避免显式转置 */
function multiplyAtB(A, B) {
    const m = A.length;
    const nA = A[0]?.length ?? 0;
    const nB = B[0]?.length ?? 0;
    const out = zeros(nA, nB);
    for (let i = 0; i < nA; i += 1) {
        for (let k = 0; k < m; k += 1) {
            const aki = A[k][i];
            if (aki === 0)
                continue;
            for (let j = 0; j < nB; j += 1) {
                out[i][j] += aki * B[k][j];
            }
        }
    }
    return out;
}
/**
 * 随机 SVD：
 * 给定 m x n 矩阵 A 和目标秩 k，返回近似的秩-k SVD。
 *
 * 算法（Halko-Martinsson-Tropp, 2011）：
 * 1. Ω = randn(n, k+p)
 * 2. Y = A * Ω                 (m x (k+p))
 * 3. （可选）幂迭代提升精度
 * 4. Q, _ = QR(Y)              (m x (k+p))
 * 5. B = Q^T * A               ((k+p) x n)
 * 6. SVD(B) = Ub * S * Vt
 * 7. U = Q * Ub, 截断到前 k 个
 */
function computeRandomizedSVD(input, targetRank, oversampling = 10, powerIter = 1) {
    const A = cloneMatrix(input);
    const m = A.length;
    const n = A[0]?.length ?? 0;
    if (m === 0 || n === 0) {
        return { U: [], S: [], Vt: [] };
    }
    const k = Math.min(targetRank, Math.min(m, n));
    const p = Math.min(oversampling, Math.min(m, n) - k);
    const l = k + p;
    // 若目标秩 >= min(m,n)，直接使用全 SVD
    if (l >= Math.min(m, n)) {
        return computeSVDFull(input);
    }
    // Step 1: 随机采样矩阵
    const omega = gaussianRandom(n, l);
    // Step 2: Y = A * Omega
    let Y = multiply(A, omega);
    // Step 3: 幂迭代（提高对衰减慢的奇异值的捕捉精度）
    for (let q = 0; q < powerIter; q += 1) {
        // 正交化 Y
        const qr1 = thinQR(Y);
        // Z = A^T * Q1
        const Z = multiplyAtB(A, qr1.Q);
        // 正交化 Z
        const qr2 = thinQR(Z);
        // Y = A * Q2
        Y = multiply(A, qr2.Q);
    }
    // Step 4: QR(Y) → Q
    const { Q } = thinQR(Y);
    // Step 5: B = Q^T * A
    const B = multiplyAtB(Q, A);
    // Step 6: 对小矩阵 B 做精确 SVD
    const svdB = computeSVDFull(B);
    // Step 7: U = Q * Ub，截断到前 k 个
    const Ub = svdB.U;
    const Ufinal = multiply(Q, Ub);
    // 截断
    const Utrunc = Ufinal.map((row) => row.slice(0, k));
    const Strunc = svdB.S.slice(0, k);
    const VtFull = svdB.Vt;
    const VtTrunc = VtFull.slice(0, k);
    return {
        U: Utrunc,
        S: Strunc,
        Vt: VtTrunc,
    };
}
// ─── 对外接口 ─────────────────────────────────────────────────────────────────
/**
 * 全 SVD 分解（内部函数，自动处理 m < n 的情况）
 */
function computeSVDFull(input) {
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
 * 作业要求：SVD 必须手动实现，不可调用现成库函数。
 *
 * 支持两种模式：
 * - 'full'（默认）：Householder 双对角化 + Wilkinson 位移迭代，精确全分解
 * - 'randomized'：Halko-Martinsson-Tropp 随机 SVD，适合大矩阵低秩近似
 */
export function computeSVD(input, options) {
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
