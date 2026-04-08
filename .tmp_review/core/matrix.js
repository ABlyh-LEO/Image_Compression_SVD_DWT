export function cloneMatrix(input) {
    return input.map((row) => [...row]);
}
export function zeros(rows, cols) {
    return Array.from({ length: rows }, () => Array.from({ length: cols }, () => 0));
}
export function transpose(input) {
    const rows = input.length;
    const cols = input[0]?.length ?? 0;
    const out = zeros(cols, rows);
    for (let i = 0; i < rows; i += 1) {
        for (let j = 0; j < cols; j += 1) {
            out[j][i] = input[i][j];
        }
    }
    return out;
}
export function multiply(A, B) {
    const rowsA = A.length;
    const colsA = A[0]?.length ?? 0;
    const rowsB = B.length;
    const colsB = B[0]?.length ?? 0;
    if (colsA !== rowsB) {
        throw new Error(`矩阵乘法维度不匹配: A(${rowsA}x${colsA}) vs B(${rowsB}x${colsB})`);
    }
    const out = zeros(rowsA, colsB);
    for (let i = 0; i < rowsA; i += 1) {
        for (let k = 0; k < colsA; k += 1) {
            const aik = A[i][k];
            for (let j = 0; j < colsB; j += 1) {
                out[i][j] += aik * B[k][j];
            }
        }
    }
    return out;
}
export function diagonal(values) {
    const out = zeros(values.length, values.length);
    values.forEach((v, i) => {
        out[i][i] = v;
    });
    return out;
}
export function clampMatrix(input, min = 0, max = 255) {
    return input.map((row) => row.map((v) => Math.max(min, Math.min(max, v))));
}
