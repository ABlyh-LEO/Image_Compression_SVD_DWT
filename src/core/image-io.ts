import type { Matrix, RGBChannels } from '../types';
import { zeros } from './matrix';

function getCanvasContext(canvas: HTMLCanvasElement): CanvasRenderingContext2D {
    const ctx = canvas.getContext('2d');
    if (!ctx) {
        throw new Error('Cannot get 2D canvas context');
    }
    return ctx;
}

export async function fileToImageElement(file: File): Promise<HTMLImageElement> {
    const objectUrl = URL.createObjectURL(file);
    const img = new Image();
    img.decoding = 'async';
    img.src = objectUrl;

    try {
        if (typeof img.decode === 'function') {
            await img.decode();
        } else {
            await new Promise<void>((resolve, reject) => {
                img.onload = () => resolve();
                img.onerror = () => reject(new Error('Image decode failed'));
            });
        }
        return img;
    } catch (error) {
        URL.revokeObjectURL(objectUrl);
        throw error;
    }
}

export function drawImageToCanvas(image: HTMLImageElement, canvas: HTMLCanvasElement): void {
    const objectUrl = image.src.startsWith('blob:') ? image.src : '';
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    const ctx = getCanvasContext(canvas);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0);

    if (objectUrl) {
        URL.revokeObjectURL(objectUrl);
    }
}

export function getImageDataFromCanvas(canvas: HTMLCanvasElement): ImageData {
    const ctx = getCanvasContext(canvas);
    return ctx.getImageData(0, 0, canvas.width, canvas.height);
}

export function imageDataToRGBMatrices(imageData: ImageData): RGBChannels {
    const { width, height, data } = imageData;
    const r = zeros(height, width);
    const g = zeros(height, width);
    const b = zeros(height, width);

    for (let y = 0; y < height; y += 1) {
        for (let x = 0; x < width; x += 1) {
            const idx = (y * width + x) * 4;
            r[y][x] = data[idx];
            g[y][x] = data[idx + 1];
            b[y][x] = data[idx + 2];
        }
    }

    return { r, g, b, width, height };
}

export function rgbMatricesToImageData(
    r: Matrix,
    g: Matrix,
    b: Matrix,
    width: number,
    height: number,
): ImageData {
    const out = new ImageData(width, height);
    for (let y = 0; y < height; y += 1) {
        for (let x = 0; x < width; x += 1) {
            const idx = (y * width + x) * 4;
            out.data[idx] = Math.max(0, Math.min(255, Math.round(r[y][x])));
            out.data[idx + 1] = Math.max(0, Math.min(255, Math.round(g[y][x])));
            out.data[idx + 2] = Math.max(0, Math.min(255, Math.round(b[y][x])));
            out.data[idx + 3] = 255;
        }
    }
    return out;
}

export function putImageDataToCanvas(imageData: ImageData, canvas: HTMLCanvasElement): void {
    canvas.width = imageData.width;
    canvas.height = imageData.height;
    const ctx = getCanvasContext(canvas);
    ctx.putImageData(imageData, 0, 0);
}
