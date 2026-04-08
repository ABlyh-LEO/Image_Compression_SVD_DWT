/// <reference lib="webworker" />

import { compressImageBySVD } from '../core/compress';

interface SVDWorkerRequest {
    id: number;
    imageData: ImageData;
    k: number;
}

interface SVDWorkerSuccess {
    id: number;
    ok: true;
    result: {
        imageData: ImageData;
        rankUsed: number;
        estimatedCompressionRatio: number;
    };
}

interface SVDWorkerFailure {
    id: number;
    ok: false;
    error: string;
}

type SVDWorkerResponse = SVDWorkerSuccess | SVDWorkerFailure;

self.onmessage = (event: MessageEvent<SVDWorkerRequest>) => {
    const { id, imageData, k } = event.data;

    try {
        const result = compressImageBySVD(imageData, k);
        const response: SVDWorkerResponse = { id, ok: true, result };
        self.postMessage(response);
    } catch (error) {
        const response: SVDWorkerResponse = {
            id,
            ok: false,
            error: error instanceof Error ? error.message : 'SVD Worker 执行失败',
        };
        self.postMessage(response);
    }
};
