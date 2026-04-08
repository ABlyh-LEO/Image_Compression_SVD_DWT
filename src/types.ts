export type Matrix = number[][];

export type CompressionMethod = 'svd' | 'dwt';

export type DWTWavelet = 'haar' | 'db2' | 'db4';

export type DWTThresholdMode = 'hard' | 'soft';

export interface DWTParams {
    wavelet: DWTWavelet;
    levels: number;
    threshold: number;
    thresholdMode: DWTThresholdMode;
}

export interface RGBChannels {
    r: Matrix;
    g: Matrix;
    b: Matrix;
    width: number;
    height: number;
}

export interface SVDResult {
    U: Matrix;
    S: number[];
    Vt: Matrix;
}

export interface CompressionResult {
    imageData: ImageData;
    rankUsed: number;
    estimatedCompressionRatio: number;
    method?: CompressionMethod;
    notes?: string;
}

export interface QualityMetrics {
    mse: number;
    rmse: number;
    mae: number;
    psnr: number;
    ssim: number;
    ncc: number;
}
