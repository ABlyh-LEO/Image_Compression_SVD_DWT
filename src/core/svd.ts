import type { Matrix, SVDResult } from '../types';
import { cloneMatrix, multiply, transpose, zeros } from './matrix';

const EPS = Number.EPSILON;

/** SVD 璁＄畻妯″紡 */
export type SVDMode = 'full' | 'randomized';

export interface SVDOptions {
    copyInput?: boolean;
    /** 'full' = Householder 鍙屽瑙掑寲 + Wilkinson 杩唬锛?randomized' = Halko-Martinsson-Tropp 闅忔満 SVD */
    mode?: SVDMode;
    /** 闅忔満 SVD 鐨勭洰鏍囩З锛堜粎 mode='randomized' 鏃舵湁鏁堬級 */
    rank?: number;
    /** 闅忔満 SVD 杩囬噰鏍锋暟锛堥粯璁?10锛?*/
    oversampling?: number;
    /** 闅忔満 SVD 骞傝凯浠ｆ鏁帮紙榛樿 1锛屾彁楂樿繎浼肩簿搴︼級 */
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
 * 绠楁硶 3.2.1锛坔ouse锛? * 杈撳叆鍚戦噺 x锛岃緭鍑?Householder 鍚戦噺 v 涓庣郴鏁?beta銆? */
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

/** U <- U * H锛屽叾涓?H = I - beta * v * v^T 浣滅敤鍦ㄥ垪鍖洪棿 start.. */
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
 * mat <- mat * G^T锛屽叾涓?G = [c s; -s c]锛孏^T = [c -s; s c]
 * 浣滅敤鍦ㄥ垪 p, q銆? *
 * givens(a, b) 杩斿洖鐨?(c, s) 婊¤冻 G * [a; b] = [r; 0]銆? * 瀵瑰簲鍙充箻 G^T 鏃讹細[a, b] * G^T = [a*c+b*s, -a*s+b*c]锛? * 褰?(a,b) = givens 鐨勮緭鍏ユ椂绗簩鍒嗛噺琚秷闆躲€? */
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
 * mat <- G * mat锛屽叾涓?G = [c s; -s c]
 * 浣滅敤鍦ㄨ p, q銆? *
 * G * [a; b] = [c*a + s*b; -s*a + c*b]锛? * 褰?(a,b) = givens 鐨勮緭鍏ユ椂绗簩鍒嗛噺琚秷闆躲€? */
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
 * 璁＄畻 Givens 鏃嬭浆鍙傛暟銆? * 杩斿洖 (c, s, r) 浣垮緱 G * [a; b] = [r; 0]锛屽叾涓?G = [c s; -s c]銆? */
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
 * 娓呯悊涓婂弻瀵硅鐭╅樀涓殑鏁板€煎櫔澹帮細
 * 浠呬繚鐣欎富瀵硅绾?(i,i) 鍜岃秴瀵硅绾?(i,i+1)锛屽叾浣欎綅缃嫢灏忎簬 eps 鍒欑疆闆躲€? */
function cleanBidiagonalNoise(B: Matrix, eps: number): void {
    const n = B.length;
    for (let i = 0; i < n; i += 1) {
        for (let j = 0; j < n; j += 1) {
            if (j !== i && j !== i + 1) {
                if (Math.abs(B[i][j]) <= eps) B[i][j] = 0;
            }
        }
    }
}

/**
 * 闆跺瑙掑厓绱?deflation锛? * 褰撴椿璺冨尯闂?[l, r] 涓煇涓瑙掑厓绱?B[i][i] 鈮?0 鏃讹紝
 * 浣跨敤 Givens 鏃嬭浆娑堥櫎璇ヨ鐨勮秴瀵硅鍏冪礌 B[i][i+1]锛? * 浠庤€屽皢闂鍒嗚В涓烘洿灏忕殑瀛愰棶棰樸€? * 杩斿洖 true 琛ㄧず鎵ц浜?deflation锛堝簲璺宠繃鏈疆 Wilkinson step锛夈€? */
function deflateZeroDiagonal(B: Matrix, P: Matrix, l: number, r: number, eps: number): boolean {
    for (let i = l; i < r; i += 1) {
        if (Math.abs(B[i][i]) <= eps) {
            // B[i][i] 鈮?0锛岀敤宸︽棆杞緷娆℃秷闄?B[i][i+1], B[i][i+2], ..., B[i][r]
            for (let j = i + 1; j <= r; j += 1) {
                if (B[i][j] === 0) continue;
                const g = givens(B[j][j], B[i][j]);
                // 宸︽棆杞綔鐢ㄥ湪琛?j, i锛堟敞鎰忛『搴忥細闆跺寲 B[i][j]锛?                applyLeftRotation(B, j, i, g.c, g.s);
                applyRightRotation(P, j, i, g.c, g.s);
            }
            B[i][i] = 0;
            return true;
        }
    }
    return false;
}

/**
 * 绠楁硶 7.6.2 鐨?Wilkinson 浣嶇Щ鎬濇兂锛屼綔鐢ㄤ簬 B(l:r,l:r)銆? */
function wilkinsonSvdStep(B: Matrix, P: Matrix, Q: Matrix, l: number, r: number): void {
    if (r <= l) return;

    const alpha = B[r][r] * B[r][r] + B[r - 1][r] * B[r - 1][r];
    const gammaPrev = r - 2 >= l ? B[r - 2][r - 1] : 0;
    const delta = (B[r - 1][r - 1] * B[r - 1][r - 1] + gammaPrev * gammaPrev - alpha) / 2;
    const beta = B[r - 1][r - 1] * B[r - 1][r];
    const denom = delta + signNonZero(delta) * Math.sqrt(delta * delta + beta * beta);
    const mu = alpha - (Math.abs(denom) < EPS ? 0 : (beta * beta) / denom);

    let y = B[l][l] * B[l][l] - mu;
    let z = B[l][l] * B[l][l + 1];

    for (let k = l; k < r; k += 1) {
        // Right rotation to annihilate z in [y, z].
        const right = givens(y, z);
        applyRightRotation(B, k, k + 1, right.c, right.s);
        applyRightRotation(Q, k, k + 1, right.c, right.s);

        // 宸︽棆杞秷闆剁敱鍙虫棆杞骇鐢熺殑涓嬪瑙掑厓绱?B[k+1][k]
        const left = givens(B[k][k], B[k + 1][k]);
        applyLeftRotation(B, k, k + 1, left.c, left.s);
        applyRightRotation(P, k, k + 1, left.c, left.s);

        if (k < r - 1) {
            y = B[k][k + 1];
            z = B[k][k + 2];
        }
    }
}

function allSuperDiagonalSmall(B: Matrix, eps: number): boolean {
    const n = B.length;
    for (let i = 0; i < n - 1; i += 1) {
        const tol = eps * (Math.abs(B[i][i]) + Math.abs(B[i + 1][i + 1]) + 1);
        if (Math.abs(B[i][i + 1]) > tol) return false;
    }
    return true;
}

function runWilkinsonPass(
    B: Matrix,
    P: Matrix,
    Q: Matrix,
    eps: number,
    maxIter: number,
): boolean {
    const n = B.length;

    for (let iter = 0; iter < maxIter; iter += 1) {
        // Convergence check: zero-out tiny off-diagonal noise.
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

        // 鎵炬椿璺冨瓙鐭╅樀 [l, r]锛氫粠搴曢儴鎵剧涓€涓潪闆惰秴瀵硅
        let r = -1;
        for (let i = n - 2; i >= 0; i -= 1) {
            if (B[i][i + 1] !== 0) {
                r = i + 1;
                break;
            }
        }
        if (r === -1) break;

        let l = r - 1;
        while (l > 0 && B[l - 1][l] !== 0) {
            l -= 1;
        }

        // Zero-diagonal deflation (Golub-Van Loan 8.6.2).
        if (deflateZeroDiagonal(B, P, l, r, eps)) {
            cleanBidiagonalNoise(B, eps);
            continue;
        }

        wilkinsonSvdStep(B, P, Q, l, r);
        cleanBidiagonalNoise(B, eps);
    }

    return allSuperDiagonalSmall(B, eps);
}

function diagWithWilkinson(B: Matrix): { P: Matrix; Q: Matrix } {
    const n = B.length;
    const P = identity(n);
    const Q = identity(n);

    const scale = Math.max(1, normInf(B));
    const baseEps = Math.sqrt(EPS) * scale;
    const baseIter = Math.max(80 * n * n, 2000);

    const attempts = [
        { epsScale: 1, iterScale: 1 },
        { epsScale: 4, iterScale: 0.45 },
        { epsScale: 16, iterScale: 0.3 },
    ];

    for (let attempt = 0; attempt < attempts.length; attempt += 1) {
        const { epsScale, iterScale } = attempts[attempt];
        const eps = baseEps * epsScale;
        const maxIter = Math.max(200, Math.floor(baseIter * iterScale));

        if (runWilkinsonPass(B, P, Q, eps, maxIter)) {
            if (attempt === 0) {
                return { P, Q };
            }

            cleanBidiagonalNoise(B, eps * 0.5);
            if (allSuperDiagonalSmall(B, eps)) {
                return { P, Q };
            }
        }

        cleanBidiagonalNoise(B, eps * 2);
    }

    const fallbackEps = baseEps * 64;
    cleanBidiagonalNoise(B, fallbackEps);
    if (allSuperDiagonalSmall(B, fallbackEps)) {
        return { P, Q };
    }

    return { P, Q };
}

// 鈹€鈹€鈹€ 鍏?SVD锛圚ouseholder 鍙屽瑙掑寲 + Wilkinson 杩唬锛?鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

function computeSVDTall(input: Matrix): SVDResult {
    const A = cloneMatrix(input);
    const m = A.length;
    const n = A[0]?.length ?? 0;
    if (m === 0 || n === 0) {
        return { U: [], S: [], Vt: [] };
    }

    // Algorithm 7.6.1: Householder bidiagonalization with accumulated U, V.
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

    // Extract bidiagonal block B (n x n).
    const B = zeros(n, n);
    for (let i = 0; i < n; i += 1) {
        B[i][i] = A[i][i];
        if (i < n - 1) {
            B[i][i + 1] = A[i][i + 1];
        }
    }

    // Algorithm 7.6.2/7.6.3: Wilkinson-shift diagonalization.
    const { P, Q } = diagWithWilkinson(B);

    // U = U * diag(P, I_{m-n}), only keep first n columns.
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

    // Sort singular values descending and reorder U/Vt accordingly.
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

// 鈹€鈹€鈹€ 闅忔満 SVD锛圚alko-Martinsson-Tropp, 2011锛夆攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

/** Box-Muller 鍙樻崲鐢熸垚鏍囧噯姝ｆ€侀殢鏈烘暟 */
function randn(): number {
    let u1 = 0;
    let u2 = 0;
    while (u1 === 0) u1 = Math.random();
    while (u2 === 0) u2 = Math.random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/** 鐢熸垚 m x n 鐨勯珮鏂殢鏈虹煩闃?*/
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
 * Householder 钖?QR 鍒嗚В锛欰 = Q * R
 * A: m x n (m >= n)锛岃繑鍥?Q: m x n, R: n x n
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

        // 绱Н Q = Q * H_k
        applyHouseholderToRight(Qfull, k, v, beta);
    }

    // 鎻愬彇 thin Q (m x n) 鍜?upper-triangular R (n x n)
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

/** mat^T * mat2 鐨勯珮鏁堣绠楋紝閬垮厤鏄惧紡杞疆 */
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
 * 闅忔満 SVD锛? * 缁欏畾 m x n 鐭╅樀 A 鍜岀洰鏍囩З k锛岃繑鍥炶繎浼肩殑绉?k SVD銆? *
 * 绠楁硶锛圚alko-Martinsson-Tropp, 2011锛夛細
 * 1. 惟 = randn(n, k+p)
 * 2. Y = A * 惟                 (m x (k+p))
 * 3. 锛堝彲閫夛級骞傝凯浠ｆ彁鍗囩簿搴? * 4. Q, _ = QR(Y)              (m x (k+p))
 * 5. B = Q^T * A               ((k+p) x n)
 * 6. SVD(B) = Ub * S * Vt
 * 7. U = Q * Ub, 鎴柇鍒板墠 k 涓? */
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

    // 鑻ョ洰鏍囩З >= min(m,n)锛岀洿鎺ヤ娇鐢ㄥ叏 SVD
    if (l >= Math.min(m, n)) {
        return computeSVDFull(input);
    }

    // Step 1: 闅忔満閲囨牱鐭╅樀
    const omega = gaussianRandom(n, l);

    // Step 2: Y = A * Omega
    let Y = multiply(A, omega);

    // Step 3: power iterations to improve approximation.
    for (let q = 0; q < powerIter; q += 1) {
        // 姝ｄ氦鍖?Y
        const qr1 = thinQR(Y);
        // Z = A^T * Q1
        const Z = multiplyAtB(A, qr1.Q);
        // 姝ｄ氦鍖?Z
        const qr2 = thinQR(Z);
        // Y = A * Q2
        Y = multiply(A, qr2.Q);
    }

    // Step 4: QR(Y) 鈫?Q
    const { Q } = thinQR(Y);

    // Step 5: B = Q^T * A
    const B = multiplyAtB(Q, A);

    // Step 6: 瀵瑰皬鐭╅樀 B 鍋氱簿纭?SVD
    const svdB = computeSVDFull(B);

    // Step 7: U = Q * Ub, then truncate to top-k.
    const Ub = svdB.U;
    const Ufinal = multiply(Q, Ub);

    // 鎴柇
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

// 鈹€鈹€鈹€ 瀵瑰鎺ュ彛 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

/**
 * 鍏?SVD 鍒嗚В锛堝唴閮ㄥ嚱鏁帮紝鑷姩澶勭悊 m < n 鐨勬儏鍐碉級
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
 * 浣滀笟瑕佹眰锛歋VD 蹇呴』鎵嬪姩瀹炵幇锛屼笉鍙皟鐢ㄧ幇鎴愬簱鍑芥暟銆? *
 * 鏀寔涓ょ妯″紡锛? * - 'full'锛堥粯璁わ級锛欻ouseholder 鍙屽瑙掑寲 + Wilkinson 浣嶇Щ杩唬锛岀簿纭叏鍒嗚В
 * - 'randomized'锛欻alko-Martinsson-Tropp 闅忔満 SVD锛岄€傚悎澶х煩闃典綆绉╄繎浼? */
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

