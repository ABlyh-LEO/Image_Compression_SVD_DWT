function toGray(data, width, height) {
    const gray = new Float64Array(width * height);
    for (let i = 0; i < width * height; i += 1) {
        const idx = i * 4;
        const r = data[idx];
        const g = data[idx + 1];
        const b = data[idx + 2];
        gray[i] = 0.299 * r + 0.587 * g + 0.114 * b;
    }
    return gray;
}
function mean(arr) {
    let sum = 0;
    for (let i = 0; i < arr.length; i += 1)
        sum += arr[i];
    return sum / arr.length;
}
export function evaluateQuality(original, reconstructed) {
    if (original.width !== reconstructed.width || original.height !== reconstructed.height) {
        throw new Error('质量评估失败：两张图像尺寸不一致');
    }
    const totalPixels = original.width * original.height;
    const o = toGray(original.data, original.width, original.height);
    const r = toGray(reconstructed.data, reconstructed.width, reconstructed.height);
    let mse = 0;
    let mae = 0;
    for (let i = 0; i < totalPixels; i += 1) {
        const diff = o[i] - r[i];
        mse += diff * diff;
        mae += Math.abs(diff);
    }
    mse /= totalPixels;
    mae /= totalPixels;
    const rmse = Math.sqrt(mse);
    const psnr = mse === 0 ? Infinity : 10 * Math.log10((255 * 255) / mse);
    // 全局 SSIM（简化版本）
    const muX = mean(o);
    const muY = mean(r);
    let sigmaX2 = 0;
    let sigmaY2 = 0;
    let sigmaXY = 0;
    for (let i = 0; i < totalPixels; i += 1) {
        const dx = o[i] - muX;
        const dy = r[i] - muY;
        sigmaX2 += dx * dx;
        sigmaY2 += dy * dy;
        sigmaXY += dx * dy;
    }
    sigmaX2 /= totalPixels;
    sigmaY2 /= totalPixels;
    sigmaXY /= totalPixels;
    const L = 255;
    const c1 = (0.01 * L) ** 2;
    const c2 = (0.03 * L) ** 2;
    const ssimNumerator = (2 * muX * muY + c1) * (2 * sigmaXY + c2);
    const ssimDenominator = (muX * muX + muY * muY + c1) * (sigmaX2 + sigmaY2 + c2);
    const ssim = ssimDenominator === 0 ? 1 : ssimNumerator / ssimDenominator;
    // NCC (Normalized Cross-Correlation)
    let numerator = 0;
    let denomX = 0;
    let denomY = 0;
    for (let i = 0; i < totalPixels; i += 1) {
        const dx = o[i] - muX;
        const dy = r[i] - muY;
        numerator += dx * dy;
        denomX += dx * dx;
        denomY += dy * dy;
    }
    const ncc = denomX === 0 || denomY === 0 ? 1 : numerator / Math.sqrt(denomX * denomY);
    return {
        mse,
        rmse,
        mae,
        psnr,
        ssim,
        ncc,
    };
}
