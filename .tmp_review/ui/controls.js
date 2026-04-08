export function bindRankDisplay(rankSlider, rankLabel) {
    const sync = () => {
        rankLabel.textContent = rankSlider.value;
    };
    sync();
    rankSlider.addEventListener('input', sync);
}
function fmt(num, digits = 4) {
    if (!Number.isFinite(num))
        return '∞';
    return num.toFixed(digits);
}
export function renderMetrics(metricsPanel, metrics, ratio) {
    metricsPanel.innerHTML = `
    <div class="metric-card"><span>MSE</span><strong>${fmt(metrics.mse)}</strong></div>
    <div class="metric-card"><span>RMSE</span><strong>${fmt(metrics.rmse)}</strong></div>
    <div class="metric-card"><span>MAE</span><strong>${fmt(metrics.mae)}</strong></div>
    <div class="metric-card"><span>PSNR (dB)</span><strong>${fmt(metrics.psnr, 2)}</strong></div>
    <div class="metric-card"><span>SSIM</span><strong>${fmt(metrics.ssim, 4)}</strong></div>
    <div class="metric-card"><span>NCC</span><strong>${fmt(metrics.ncc, 4)}</strong></div>
    <div class="metric-card"><span>估计压缩比</span><strong>${fmt(ratio, 2)}x</strong></div>
  `;
}
export function setStatus(statusText, message, mode = 'normal') {
    statusText.textContent = message;
    statusText.classList.remove('ok', 'error');
    if (mode !== 'normal') {
        statusText.classList.add(mode);
    }
}
