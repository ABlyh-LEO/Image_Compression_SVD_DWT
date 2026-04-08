export function clearCanvas(canvas: HTMLCanvasElement): void {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

export function fitCanvasPreview(canvas: HTMLCanvasElement, maxWidth = 560, maxHeight = 420): void {
    if (canvas.width <= 0 || canvas.height <= 0) {
        canvas.style.width = '';
        canvas.style.height = '';
        return;
    }

    const widthScale = maxWidth / canvas.width;
    const heightScale = maxHeight / canvas.height;
    const scale = Math.min(widthScale, heightScale, 1);

    canvas.style.width = `${Math.round(canvas.width * scale)}px`;
    canvas.style.height = `${Math.round(canvas.height * scale)}px`;
}
