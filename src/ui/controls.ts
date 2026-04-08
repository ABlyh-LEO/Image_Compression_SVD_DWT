import type { QualityMetrics } from '../types';

export interface UIRefs {
    fileInput: HTMLInputElement;
    rankSlider: HTMLInputElement;
    rankLabel: HTMLElement;
    compressButton: HTMLButtonElement;
    statusText: HTMLElement;
    metricsPanel: HTMLElement;
}

export function bindRankDisplay(rankSlider: HTMLInputElement, rankLabel: HTMLElement): void {
    const sync = () => {
        rankLabel.textContent = rankSlider.value;
    };
    sync();
    rankSlider.addEventListener('input', sync);
}

function fmt(num: number, digits = 4): string {
    if (!Number.isFinite(num)) return num > 0 ? 'Inf' : '-Inf';
    return num.toFixed(digits);
}

export function renderMetricsPlaceholder(metricsPanel: HTMLElement): void {
    metricsPanel.innerHTML = '<div class="metrics-empty">压缩完成后在这里显示指标。</div>';
}

export function renderMetrics(metricsPanel: HTMLElement, metrics: QualityMetrics, ratio: number): void {
    const cards = [
        { label: 'MSE', value: fmt(metrics.mse) },
        { label: 'RMSE', value: fmt(metrics.rmse) },
        { label: 'MAE', value: fmt(metrics.mae) },
        { label: 'PSNR (dB)', value: fmt(metrics.psnr, 2), highlight: true },
        { label: 'SSIM', value: fmt(metrics.ssim, 4), highlight: true },
        { label: 'NCC', value: fmt(metrics.ncc, 4) },
        { label: '估算压缩率', value: `${fmt(ratio, 2)}x`, highlight: true },
    ];

    metricsPanel.innerHTML = cards
        .map((card) => `
    <div class="metric-card${card.highlight ? ' metric-highlight' : ''}">
      <span>${card.label}</span>
      <strong>${card.value}</strong>
    </div>
  `)
        .join('');
}

export function setStatus(statusText: HTMLElement, message: string, mode: 'normal' | 'error' | 'ok' = 'normal'): void {
    statusText.textContent = message;
    statusText.classList.remove('ok', 'error');
    if (mode !== 'normal') {
        statusText.classList.add(mode);
    }
}
