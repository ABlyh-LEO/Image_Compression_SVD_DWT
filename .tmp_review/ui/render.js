export function clearCanvas(canvas) {
    const ctx = canvas.getContext('2d');
    if (!ctx)
        return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}
export function fitCanvasPreview(canvas, maxWidth = 560) {
    if (canvas.width <= maxWidth) {
        canvas.style.width = `${canvas.width}px`;
        return;
    }
    canvas.style.width = `${maxWidth}px`;
}
