import { zeros } from './matrix';
function getCanvasContext(canvas) {
    const ctx = canvas.getContext('2d');
    if (!ctx) {
        throw new Error('无法获取 2D canvas 上下文');
    }
    return ctx;
}
export async function fileToImageElement(file) {
    const objectUrl = URL.createObjectURL(file);
    const img = new Image();
    img.decoding = 'async';
    img.src = objectUrl;
    try {
        if (typeof img.decode === 'function') {
            await img.decode();
        }
        else {
            await new Promise((resolve, reject) => {
                img.onload = () => resolve();
                img.onerror = () => reject(new Error('图片解码失败'));
            });
        }
        return img;
    }
    catch (error) {
        URL.revokeObjectURL(objectUrl);
        throw error;
    }
}
export function drawImageToCanvas(image, canvas) {
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
export function getImageDataFromCanvas(canvas) {
    const ctx = getCanvasContext(canvas);
    return ctx.getImageData(0, 0, canvas.width, canvas.height);
}
export function imageDataToRGBMatrices(imageData) {
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
export function rgbMatricesToImageData(r, g, b, width, height) {
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
export function putImageDataToCanvas(imageData, canvas) {
    canvas.width = imageData.width;
    canvas.height = imageData.height;
    const ctx = getCanvasContext(canvas);
    ctx.putImageData(imageData, 0, 0);
}
