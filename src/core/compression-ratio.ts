const RGB_CHANNELS = 3;
const SOURCE_BITS_PER_CHANNEL = 8;
const COEFFICIENT_BITS = 32;

function safePositiveInt(value: number, fallback = 1): number {
    if (!Number.isFinite(value)) return fallback;
    return Math.max(1, Math.floor(value));
}

export function clampRankByShape(rank: number, width: number, height: number): number {
    const maxRank = Math.max(1, Math.min(safePositiveInt(width), safePositiveInt(height)));
    const raw = safePositiveInt(rank);
    return Math.min(raw, maxRank);
}

export function estimateOriginalImageBytes(width: number, height: number): number {
    const w = safePositiveInt(width);
    const h = safePositiveInt(height);
    return (w * h * RGB_CHANNELS * SOURCE_BITS_PER_CHANNEL) / 8;
}

export function estimateSVDCompressedBytes(width: number, height: number, rank: number): number {
    const w = safePositiveInt(width);
    const h = safePositiveInt(height);
    const k = clampRankByShape(rank, w, h);
    // Per channel: U(m*k) + S(k) + Vt(k*n) scalars.
    const scalarCount = RGB_CHANNELS * k * (h + w + 1);
    return (scalarCount * COEFFICIENT_BITS) / 8;
}

export function estimateDWTCompressedBytes(nonZeroCoefficients: number, totalCoefficients: number): number {
    const nnz = Math.max(0, Math.floor(nonZeroCoefficients));
    if (nnz === 0) return 0;
    const total = Math.max(2, Math.floor(totalCoefficients));
    // Sparse representation: each retained coefficient stores a value and an index.
    const indexBits = Math.ceil(Math.log2(total));
    const bitsPerEntry = COEFFICIENT_BITS + indexBits;
    return (nnz * bitsPerEntry) / 8;
}

export function compressionRatioFromBytes(originalBytes: number, compressedBytes: number): number {
    if (!Number.isFinite(originalBytes) || originalBytes <= 0) return 0;
    if (!Number.isFinite(compressedBytes) || compressedBytes <= 0) return Infinity;
    return originalBytes / compressedBytes;
}

export function estimateSVDCompressionRatio(width: number, height: number, rank: number): number {
    const originalBytes = estimateOriginalImageBytes(width, height);
    const compressedBytes = estimateSVDCompressedBytes(width, height, rank);
    return compressionRatioFromBytes(originalBytes, compressedBytes);
}

export function estimateDWTCompressionRatio(
    width: number,
    height: number,
    nonZeroCoefficients: number,
    totalCoefficients: number,
): number {
    const originalBytes = estimateOriginalImageBytes(width, height);
    const compressedBytes = estimateDWTCompressedBytes(nonZeroCoefficients, totalCoefficients);
    return compressionRatioFromBytes(originalBytes, compressedBytes);
}
