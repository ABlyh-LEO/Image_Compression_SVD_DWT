/// <reference lib="webworker" />

import type { Matrix } from '../types';
import { computeSVD, type SVDMode } from '../core/svd';

interface PrepareRequest {
    id: number;
    type: 'prepare';
    channel: Uint8ClampedArray;
    width: number;
    height: number;
    svdMode?: SVDMode;
    rank?: number;
}

interface ReconstructRequest {
    id: number;
    type: 'reconstruct';
    k: number;
}

interface DisposeRequest {
    id: number;
    type: 'dispose';
}

type ChannelWorkerRequest = PrepareRequest | ReconstructRequest | DisposeRequest;

interface ChannelWorkerSuccess {
    id: number;
    ok: true;
    channel?: Uint8ClampedArray;
}

interface ChannelWorkerFailure {
    id: number;
    ok: false;
    error: string;
}

type ChannelWorkerResponse = ChannelWorkerSuccess | ChannelWorkerFailure;

interface CachedSVD {
    S: number[];
    UTypedRows: Float64Array[];
    VtTypedRows: Float64Array[];
    width: number;
    height: number;
}

let cache: CachedSVD | null = null;

function toMatrix(channel: Uint8ClampedArray, width: number, height: number): Matrix {
    const buffer = new Float64Array(width * height);
    const rows: Matrix = new Array(height);

    for (let y = 0; y < height; y += 1) {
        const start = y * width;
        const row = buffer.subarray(start, start + width);
        rows[y] = row as unknown as number[];
        for (let x = 0; x < width; x += 1) {
            row[x] = channel[start + x];
        }
    }
    return rows;
}

function reconstructToUint8(U: Float64Array[], S: number[], Vt: Float64Array[], k: number): Uint8ClampedArray {
    const rows = U.length;
    const cols = Vt[0]?.length ?? 0;
    const rank = Math.max(1, Math.min(k, S.length));
    const out = new Uint8ClampedArray(rows * cols);

    const rowAcc = new Float64Array(cols);

    for (let i = 0; i < rows; i += 1) {
        rowAcc.fill(0);
        const uRow = U[i];

        for (let t = 0; t < rank; t += 1) {
            const scaled = uRow[t] * S[t];
            if (scaled === 0) continue;
            const vtRow = Vt[t];
            for (let j = 0; j < cols; j += 1) {
                rowAcc[j] += scaled * vtRow[j];
            }
        }

        const rowStart = i * cols;
        for (let j = 0; j < cols; j += 1) {
            const v = Math.round(rowAcc[j]);
            out[rowStart + j] = v < 0 ? 0 : v > 255 ? 255 : v;
        }
    }

    return out;
}

self.onmessage = (event: MessageEvent<ChannelWorkerRequest>) => {
    const request = event.data;
    const { id } = request;

    try {
        if (request.type === 'prepare') {
            const svdMode: SVDMode = request.svdMode ?? 'full';
            const mat = toMatrix(request.channel, request.width, request.height);
            const { U, S, Vt } = computeSVD(mat, {
                copyInput: false,
                mode: svdMode,
                rank: request.rank,
            });
            const UTypedRows = U.map((row) => Float64Array.from(row));
            const VtTypedRows = Vt.map((row) => Float64Array.from(row));
            cache = {
                S,
                UTypedRows,
                VtTypedRows,
                width: request.width,
                height: request.height,
            };
            const response: ChannelWorkerResponse = { id, ok: true };
            self.postMessage(response);
            return;
        }

        if (request.type === 'reconstruct') {
            if (!cache) {
                throw new Error('SVD 缓存未初始化，请先发送 prepare 请求');
            }
            const out = reconstructToUint8(cache.UTypedRows, cache.S, cache.VtTypedRows, request.k);
            const response: ChannelWorkerResponse = { id, ok: true, channel: out };
            self.postMessage(response, [out.buffer]);
            return;
        }

        cache = null;
        const response: ChannelWorkerResponse = { id, ok: true };
        self.postMessage(response);
    } catch (error) {
        const response: ChannelWorkerResponse = {
            id,
            ok: false,
            error: error instanceof Error ? error.message : 'SVD Channel Worker 执行失败',
        };
        self.postMessage(response);
    }
};
