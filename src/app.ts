import { compressImageBySVDNonBlocking } from './core/compress';
import type { SVDMode } from './core/svd';
import { compressImageByDWT, suggestDWTThresholdForTargetRatio } from './core/dwt';
import SVDChannelWorkerConstructor from './workers/svd-channel.worker.ts?worker&inline';
import {
  drawImageToCanvas,
  fileToImageElement,
  getImageDataFromCanvas,
  putImageDataToCanvas,
} from './core/image-io';
import { clampRankByShape, estimateSVDCompressionRatio } from './core/compression-ratio';
import { evaluateQuality } from './core/metrics';
import type { CompressionMethod, CompressionResult, DWTParams } from './types';
import { renderMetrics, renderMetricsPlaceholder, setStatus } from './ui/controls';
import { fitCanvasPreview } from './ui/render';

interface ChannelWorkerRequest {
  id: number;
  type: 'prepare' | 'reconstruct' | 'dispose';
  channel?: Uint8ClampedArray;
  width?: number;
  height?: number;
  k?: number;
  svdMode?: SVDMode;
  rank?: number;
}

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

interface SVDWorkerClient {
  prepare: (
    imageData: ImageData,
    onProgress?: (progress01: number) => void,
    svdMode?: SVDMode,
    rank?: number,
  ) => Promise<void>;
  reconstruct: (k: number, onProgress?: (progress01: number) => void) => Promise<CompressionResult>;
  hasPrepared: () => boolean;
  invalidate: () => void;
  dispose: () => void;
}

function createSVDWorkerClient(): SVDWorkerClient {
  const cpuCores = Math.max(1, navigator.hardwareConcurrency ?? 4);
  // 独立重任务最多只有 RGB 三通道，这里按设备核心数自适应分配 worker 数量。
  const workerCount = Math.max(1, Math.min(3, cpuCores));
  const workers = Array.from({ length: workerCount }, () => new SVDChannelWorkerConstructor());
  let seq = 1;

  const runTask = (
    worker: Worker,
    payload: Omit<ChannelWorkerRequest, 'id'>,
    transferList?: Transferable[],
  ): Promise<ChannelWorkerSuccess> => {
    const id = seq;
    seq += 1;

    return new Promise<ChannelWorkerSuccess>((resolve, reject) => {
      const onMessage = (event: MessageEvent<ChannelWorkerResponse>) => {
        const response = event.data;
        if (response.id !== id) return;
        worker.removeEventListener('message', onMessage as EventListener);
        worker.removeEventListener('error', onError);

        if (response.ok) {
          resolve(response);
        } else {
          reject(new Error(response.error));
        }
      };

      const onError = (event: ErrorEvent) => {
        worker.removeEventListener('message', onMessage as EventListener);
        worker.removeEventListener('error', onError);
        reject(new Error(event.message || 'SVD Worker 运行错误'));
      };

      worker.addEventListener('message', onMessage as EventListener);
      worker.addEventListener('error', onError);

      const request: ChannelWorkerRequest = { id, ...payload };
      worker.postMessage(request, transferList ?? []);
    });
  };

  const splitChannels = (imageData: ImageData) => {
    const { width, height, data } = imageData;
    const size = width * height;
    const r = new Uint8ClampedArray(size);
    const g = new Uint8ClampedArray(size);
    const b = new Uint8ClampedArray(size);

    for (let i = 0; i < size; i += 1) {
      const idx = i * 4;
      r[i] = data[idx];
      g[i] = data[idx + 1];
      b[i] = data[idx + 2];
    }

    return { r, g, b, width, height };
  };

  const mergeChannels = (
    width: number,
    height: number,
    r: Uint8ClampedArray,
    g: Uint8ClampedArray,
    b: Uint8ClampedArray,
  ): ImageData => {
    const out = new ImageData(width, height);
    const size = width * height;
    for (let i = 0; i < size; i += 1) {
      const idx = i * 4;
      out.data[idx] = r[i];
      out.data[idx + 1] = g[i];
      out.data[idx + 2] = b[i];
      out.data[idx + 3] = 255;
    }
    return out;
  };

  let prepared = false;
  let imageWidth = 0;
  let imageHeight = 0;
  const perChannelKCache = new Map<number, [Uint8ClampedArray, Uint8ClampedArray, Uint8ClampedArray]>();

  const clampRank = (k: number) => clampRankByShape(k, imageWidth, imageHeight);

  const computeRatio = (k: number) => estimateSVDCompressionRatio(imageWidth, imageHeight, k);

  return {
    async prepare(
      imageData: ImageData,
      onProgress?: (progress01: number) => void,
      svdMode: SVDMode = 'full',
      rank?: number,
    ): Promise<void> {
      const safeCopy = new ImageData(new Uint8ClampedArray(imageData.data), imageData.width, imageData.height);
      const { r, g, b, width, height } = splitChannels(safeCopy);
      imageWidth = width;
      imageHeight = height;
      perChannelKCache.clear();
      prepared = false;

      const channels = [r, g, b];
      let done = 0;

      await Promise.all(
        channels.map(async (channel, idx) => {
          const worker = workers[idx % workers.length];
          await runTask(
            worker,
            {
              type: 'prepare',
              channel,
              width,
              height,
              svdMode,
              rank,
            },
            [channel.buffer],
          );
          done += 1;
          onProgress?.(done / 3);
        }),
      );

      prepared = true;
    },

    async reconstruct(k: number, onProgress?: (progress01: number) => void): Promise<CompressionResult> {
      if (!prepared) {
        throw new Error('SVD 尚未预计算，请先执行 prepare');
      }

      const kClamped = clampRank(k);
      const cached = perChannelKCache.get(kClamped);
      if (cached) {
        const image = mergeChannels(imageWidth, imageHeight, cached[0], cached[1], cached[2]);
        return {
          imageData: image,
          rankUsed: kClamped,
          estimatedCompressionRatio: computeRatio(kClamped),
        };
      }

      const outputs = new Array<Uint8ClampedArray>(3);
      let done = 0;

      await Promise.all(
        [0, 1, 2].map(async (idx) => {
          const worker = workers[idx % workers.length];
          const resp = await runTask(worker, {
            type: 'reconstruct',
            k: kClamped,
          });
          if (!resp.channel) {
            throw new Error('Worker 未返回重建通道数据');
          }
          outputs[idx] = resp.channel;
          done += 1;
          onProgress?.(done / 3);
        }),
      );

      perChannelKCache.set(kClamped, [outputs[0], outputs[1], outputs[2]]);

      const image = mergeChannels(imageWidth, imageHeight, outputs[0], outputs[1], outputs[2]);
      return {
        imageData: image,
        rankUsed: kClamped,
        estimatedCompressionRatio: computeRatio(kClamped),
      };
    },

    hasPrepared(): boolean {
      return prepared;
    },

    invalidate(): void {
      prepared = false;
      perChannelKCache.clear();
      workers.forEach((worker) => {
        void runTask(worker, { type: 'dispose' });
      });
    },

    dispose(): void {
      workers.forEach((w) => w.terminate());
    },
  };
}

function nextFrame(): Promise<void> {
  return new Promise((resolve) => {
    requestAnimationFrame(() => resolve());
  });
}

function template(): string {
  return `
    <main class="shell">
      <section class="hero glass">
        <div>
          <span class="eyebrow">Image Compression Lab</span>
          <h1>SVD 图像压缩实验台</h1>
          <p>纯前端实现 SVD 与 DWT 图像压缩，对比压缩率与重建质量指标。</p>
        </div>
      </section>

      <section class="panel glass controls">
        <div class="section-heading">
          <h2>参数设置</h2>
        </div>

        <div class="controls-top">
          <label class="file-upload control-card control-card-upload">
            <input id="fileInput" type="file" accept="image/*" />
            <span class="control-kicker">图片</span>
            <strong id="fileNameText">选择图片</strong>
            <small id="fileMetaText">支持常见 image/* 格式</small>
          </label>

          <div class="control-card algo-group">
            <label for="methodSelect">压缩方法</label>
            <select id="methodSelect">
              <option value="svd">SVD 压缩</option>
              <option value="dwt">DWT 压缩</option>
            </select>
          </div>

          <div class="control-card ratio-tool">
            <label for="targetRatioInput">目标压缩率 (x)</label>
            <div class="ratio-tool-row">
              <input id="targetRatioInput" type="number" min="0.1" step="0.1" value="10" />
              <button id="solveSvdRankButton" type="button" class="aux-btn">求解 SVD k</button>
              <button id="solveDwtThresholdButton" type="button" class="aux-btn">求解 DWT λ</button>
            </div>
          </div>
        </div>

        <div class="param-panels">
          <div id="svdParams" class="param-panel">
            <div class="param-panel-head">
              <h4>SVD 参数</h4>
            </div>
            <div class="param-grid">
              <div class="algo-group">
                <label for="svdModeSelect">SVD 模式</label>
                <select id="svdModeSelect">
                  <option value="full">全量分解（精确）</option>
                  <option value="randomized">随机 SVD（快速近似）</option>
                </select>
              </div>
              <div class="rank-group">
                <label for="rankSlider">保留 rank k: <strong id="rankLabel">30</strong></label>
                <div class="rank-control-row">
                  <input id="rankSlider" type="range" min="1" max="200" value="30" />
                  <input id="rankInput" type="number" min="1" max="200" value="30" step="1" />
                </div>
              </div>
            </div>
            <p id="svdModeHint" class="todo-hint" style="display:none">
              随机 SVD 模式下，调整 rank 后需要点击“开始压缩”才会重新计算。
            </p>
          </div>

          <div id="dwtParams" class="param-panel hidden">
            <div class="param-panel-head">
              <h4>DWT 参数</h4>
            </div>
            <div class="dwt-grid">
              <div class="dwt-item">
                <label for="waveletSelect">小波基</label>
                <select id="waveletSelect">
                  <option value="haar">Haar</option>
                  <option value="db2">Daubechies-2</option>
                  <option value="db4">Daubechies-4</option>
                </select>
                <span class="hint">Haar 最快，DB4 更平滑</span>
              </div>

              <div class="dwt-item">
                <label for="dwtLevelSlider">分解层数: <strong id="dwtLevelLabel">2</strong></label>
                <input id="dwtLevelSlider" type="range" min="1" max="6" value="2" />
                <span class="hint">层数越多，压缩更强</span>
              </div>

              <div class="dwt-item">
                <label for="dwtThresholdSlider">阈值强度 λ: <strong id="dwtThresholdLabel">20</strong></label>
                <input id="dwtThresholdSlider" type="range" min="0" max="100" value="20" />
                <span class="hint">阈值越大，压缩率越高</span>
              </div>

              <div class="dwt-item">
                <label for="thresholdModeSelect">阈值策略</label>
                <select id="thresholdModeSelect">
                  <option value="hard">硬阈值</option>
                  <option value="soft">软阈值</option>
                </select>
                <span class="hint">硬阈值更锐，软阈值更平滑</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section class="panel glass stage-panel">
        <div class="section-heading">
          <h2>图像预览</h2>
        </div>

        <div class="action-row stage-toolbar">
          <div class="button-row">
            <button id="compressButton" disabled>开始压缩</button>
            <button id="stopButton" class="stop-btn" style="display:none">停止计算</button>
          </div>
          <p id="statusText" class="status">请先上传一张图片。</p>
        </div>

        <div class="canvas-grid">
          <article id="originalStage" class="canvas-card is-empty">
            <div class="canvas-card-head">
              <h3>原图</h3>
            </div>
            <div class="canvas-frame">
              <canvas id="originalCanvas"></canvas>
              <p class="canvas-empty-tip">上传图片后显示原图</p>
            </div>
          </article>
          <article id="compressedStage" class="canvas-card is-empty">
            <div class="canvas-card-head">
              <h3>压缩重建图</h3>
            </div>
            <div class="canvas-frame">
              <canvas id="compressedCanvas"></canvas>
              <p class="canvas-empty-tip">压缩完成后显示结果</p>
            </div>
          </article>
        </div>
      </section>

      <section class="panel glass metrics">
        <div class="section-heading">
          <h2>压缩效果指标</h2>
        </div>
        <div id="metricsPanel" class="metric-grid">
          <div class="metrics-empty">压缩完成后在这里显示指标。</div>
        </div>
      </section>
    </main>
  `;
}

export function initApp(root: HTMLDivElement): void {
  root.innerHTML = template();

  // 优先尝试 Worker；若浏览器策略限制则自动降级到主线程。
  let svdWorker: SVDWorkerClient | null = null;
  let workerDisabledReason = '';
  try {
    svdWorker = createSVDWorkerClient();
  } catch (error) {
    workerDisabledReason = error instanceof Error ? error.message : 'Worker 初始化失败';
  }

  const fileInput = document.querySelector<HTMLInputElement>('#fileInput');
  const uploadCard = document.querySelector<HTMLElement>('.control-card-upload');
  const fileNameText = document.querySelector<HTMLElement>('#fileNameText');
  const fileMetaText = document.querySelector<HTMLElement>('#fileMetaText');
  const methodSelect = document.querySelector<HTMLSelectElement>('#methodSelect');
  const svdParams = document.querySelector<HTMLDivElement>('#svdParams');
  const svdModeSelect = document.querySelector<HTMLSelectElement>('#svdModeSelect');
  const svdModeHint = document.querySelector<HTMLElement>('#svdModeHint');
  const dwtParams = document.querySelector<HTMLDivElement>('#dwtParams');
  const rankSlider = document.querySelector<HTMLInputElement>('#rankSlider');
  const rankInput = document.querySelector<HTMLInputElement>('#rankInput');
  const rankLabel = document.querySelector<HTMLElement>('#rankLabel');
  const waveletSelect = document.querySelector<HTMLSelectElement>('#waveletSelect');
  const dwtLevelSlider = document.querySelector<HTMLInputElement>('#dwtLevelSlider');
  const dwtLevelLabel = document.querySelector<HTMLElement>('#dwtLevelLabel');
  const dwtThresholdSlider = document.querySelector<HTMLInputElement>('#dwtThresholdSlider');
  const dwtThresholdLabel = document.querySelector<HTMLElement>('#dwtThresholdLabel');
  const thresholdModeSelect = document.querySelector<HTMLSelectElement>('#thresholdModeSelect');
  const targetRatioInput = document.querySelector<HTMLInputElement>('#targetRatioInput');
  const solveSvdRankButton = document.querySelector<HTMLButtonElement>('#solveSvdRankButton');
  const solveDwtThresholdButton = document.querySelector<HTMLButtonElement>('#solveDwtThresholdButton');
  const compressButton = document.querySelector<HTMLButtonElement>('#compressButton');
  const stopButton = document.querySelector<HTMLButtonElement>('#stopButton');
  const statusText = document.querySelector<HTMLElement>('#statusText');
  const metricsPanel = document.querySelector<HTMLElement>('#metricsPanel');
  const originalStage = document.querySelector<HTMLElement>('#originalStage');
  const compressedStage = document.querySelector<HTMLElement>('#compressedStage');
  const originalCanvas = document.querySelector<HTMLCanvasElement>('#originalCanvas');
  const compressedCanvas = document.querySelector<HTMLCanvasElement>('#compressedCanvas');

  if (
    !fileInput ||
    !uploadCard ||
    !fileNameText ||
    !fileMetaText ||
    !methodSelect ||
    !svdParams ||
    !svdModeSelect ||
    !svdModeHint ||
    !dwtParams ||
    !rankSlider ||
    !rankInput ||
    !rankLabel ||
    !waveletSelect ||
    !dwtLevelSlider ||
    !dwtLevelLabel ||
    !dwtThresholdSlider ||
    !dwtThresholdLabel ||
    !thresholdModeSelect ||
    !targetRatioInput ||
    !solveSvdRankButton ||
    !solveDwtThresholdButton ||
    !compressButton ||
    !stopButton ||
    !statusText ||
    !metricsPanel ||
    !originalStage ||
    !compressedStage ||
    !originalCanvas ||
    !compressedCanvas
  ) {
    throw new Error('UI 初始化失败：存在缺失的 DOM 节点');
  }

  const syncRankValue = (raw: number) => {
    const minRank = Number(rankSlider.min);
    const maxRank = Number(rankSlider.max);
    const clamped = Math.max(minRank, Math.min(Math.round(raw), maxRank));
    const rankText = String(clamped);
    rankSlider.value = rankText;
    rankInput.value = rankText;
    rankLabel.textContent = rankText;
    return clamped;
  };
  syncRankValue(Number(rankSlider.value));

  const bindNumberDisplay = (input: HTMLInputElement, label: HTMLElement) => {
    const sync = () => {
      label.textContent = input.value;
    };
    sync();
    input.addEventListener('input', sync);
  };
  bindNumberDisplay(dwtLevelSlider, dwtLevelLabel);
  bindNumberDisplay(dwtThresholdSlider, dwtThresholdLabel);

  const readTargetRatio = (): number | null => {
    const target = Number(targetRatioInput.value);
    if (!Number.isFinite(target) || target <= 0) {
      setStatus(statusText, '请输入有效的目标压缩率（> 0）', 'error');
      return null;
    }
    return target;
  };

  const formatRatio = (value: number, digits = 2): string => {
    if (!Number.isFinite(value)) return '∞';
    return `${value.toFixed(digits)}x`;
  };

  const setStageFilled = (stage: HTMLElement, filled: boolean) => {
    stage.classList.toggle('is-empty', !filled);
  };

  const runInNextTask = <T>(fn: () => T): Promise<T> =>
    new Promise<T>((resolve, reject) => {
      setTimeout(() => {
        try {
          resolve(fn());
        } catch (error) {
          reject(error);
        }
      }, 0);
    });

  const findBestSvdRankForTarget = (width: number, height: number, targetRatio: number) => {
    const maxRank = Math.max(1, Math.min(width, height));
    const approx = (width * height) / (4 * targetRatio * (width + height + 1));

    const candidates = new Set<number>([1, maxRank, Math.floor(approx), Math.ceil(approx)]);
    for (let delta = -4; delta <= 4; delta += 1) {
      candidates.add(Math.floor(approx) + delta);
      candidates.add(Math.ceil(approx) + delta);
    }

    let bestRank = 1;
    let bestRatio = estimateSVDCompressionRatio(width, height, 1);
    let bestDist = Math.abs(bestRatio - targetRatio);

    for (const candidate of candidates) {
      const rank = clampRankByShape(candidate, width, height);
      const ratio = estimateSVDCompressionRatio(width, height, rank);
      const dist = Math.abs(ratio - targetRatio);
      if (dist < bestDist || (dist === bestDist && rank < bestRank)) {
        bestDist = dist;
        bestRank = rank;
        bestRatio = ratio;
      }
    }

    return { rank: bestRank, ratio: bestRatio };
  };

  const syncControlByMethod = () => {
    const method = methodSelect.value as CompressionMethod;
    const isSVD = method === 'svd';

    svdParams.classList.toggle('hidden', !isSVD);
    dwtParams.classList.toggle('hidden', isSVD);
    solveSvdRankButton.classList.toggle('hidden', !isSVD);
    solveDwtThresholdButton.classList.toggle('hidden', isSVD);

    rankSlider.disabled = !isSVD;
    rankInput.disabled = !isSVD;
    rankSlider.parentElement?.classList.toggle('disabled', !isSVD);
  };

  const syncSvdModeHint = () => {
    const isRandom = svdModeSelect.value === 'randomized';
    svdModeHint.style.display = isRandom ? 'block' : 'none';
  };

  let originalImageData: ImageData | null = null;
  let isProcessing = false;
  let lastRenderToken = 0;
  let hasCompressedOnce = false;

  const setProcessingState = (processing: boolean) => {
    isProcessing = processing;
    methodSelect.disabled = processing;
    fileInput.disabled = processing;
    compressButton.disabled = processing || !originalImageData;
    compressButton.style.display = processing ? 'none' : '';
    stopButton.style.display = processing ? '' : 'none';
    rankSlider.disabled = processing || methodSelect.value !== 'svd';
    rankInput.disabled = processing || methodSelect.value !== 'svd';
    targetRatioInput.disabled = processing;
    solveSvdRankButton.disabled = processing;
    solveDwtThresholdButton.disabled = processing;
  };

  syncControlByMethod();
  syncSvdModeHint();
  renderMetricsPlaceholder(metricsPanel);

  methodSelect.addEventListener('change', () => {
    syncControlByMethod();
    syncSvdModeHint();
  });

  svdModeSelect.addEventListener('change', () => {
    syncSvdModeHint();
    // 切换模式时清理缓存，需要重新 prepare
    svdWorker?.invalidate();
  });

  stopButton.addEventListener('click', () => {
    if (!isProcessing) return;

    // 使所有待处理的异步结果失效
    lastRenderToken += 1;

    // 终止现有 Worker 并重建
    if (svdWorker) {
      svdWorker.dispose();
      svdWorker = null;
      try {
        svdWorker = createSVDWorkerClient();
      } catch (error) {
        workerDisabledReason = error instanceof Error ? error.message : 'Worker 重建失败';
      }
    }

    hasCompressedOnce = false;
    setProcessingState(false);
    setStatus(statusText, '计算已停止。', 'error');
  });

  const loadSelectedFile = async (file: File) => {
    try {
      setStatus(statusText, '正在加载图片...');
      const image = await fileToImageElement(file);
      drawImageToCanvas(image, originalCanvas);
      fitCanvasPreview(originalCanvas);
      setStageFilled(originalStage, true);

      const loaded = getImageDataFromCanvas(originalCanvas);
      // 显式拷贝，确保后续压缩流程不会污染“原图”数据。
      originalImageData = new ImageData(new Uint8ClampedArray(loaded.data), loaded.width, loaded.height);
      svdWorker?.invalidate();
      hasCompressedOnce = false;
      fileNameText.textContent = file.name;
      fileMetaText.textContent = `${loaded.width} x ${loaded.height} · ${(file.size / 1024).toFixed(1)} KB`;

      compressedCanvas.width = originalCanvas.width;
      compressedCanvas.height = originalCanvas.height;
      compressedCanvas.getContext('2d')?.clearRect(0, 0, compressedCanvas.width, compressedCanvas.height);
      fitCanvasPreview(compressedCanvas);
      setStageFilled(compressedStage, false);

      const maxRank = Math.max(1, Math.min(originalCanvas.width, originalCanvas.height));
      rankSlider.max = String(maxRank);
      rankInput.max = String(maxRank);
      if (Number(rankSlider.value) > maxRank) {
        syncRankValue(Math.max(1, Math.floor(maxRank / 4)));
      } else {
        syncRankValue(Number(rankSlider.value));
      }

      compressButton.disabled = false;
      renderMetricsPlaceholder(metricsPanel);
      if (workerDisabledReason) {
        setStatus(statusText, `图片已加载：${loaded.width} x ${loaded.height}（${workerDisabledReason}）`, 'ok');
      } else {
        setStatus(statusText, `图片已加载：${loaded.width} x ${loaded.height}`, 'ok');
      }
    } catch (error) {
      const msg = error instanceof Error ? error.message : '加载图片失败';
      setStatus(statusText, msg, 'error');
    }
  };

  fileInput.addEventListener('change', async () => {
    if (isProcessing) return;
    const file = fileInput.files?.[0];
    if (!file) return;
    await loadSelectedFile(file);
  });

  const preventDragDefaults = (event: DragEvent) => {
    event.preventDefault();
    event.stopPropagation();
  };

  ['dragenter', 'dragover'].forEach((eventName) => {
    uploadCard.addEventListener(eventName, (event) => {
      preventDragDefaults(event as DragEvent);
      uploadCard.classList.add('is-dragover');
    });
  });

  ['dragleave', 'dragend'].forEach((eventName) => {
    uploadCard.addEventListener(eventName, (event) => {
      preventDragDefaults(event as DragEvent);
      uploadCard.classList.remove('is-dragover');
    });
  });

  uploadCard.addEventListener('drop', async (event) => {
    preventDragDefaults(event);
    uploadCard.classList.remove('is-dragover');
    if (isProcessing) return;

    const file = event.dataTransfer?.files?.[0];
    if (!file || !file.type.startsWith('image/')) {
      setStatus(statusText, '请拖入图片文件。', 'error');
      return;
    }

    await loadSelectedFile(file);
  });

  compressButton.addEventListener('click', async () => {
    if (isProcessing) return;
    if (!originalImageData) {
      setStatus(statusText, '请先上传图片', 'error');
      return;
    }

    try {
      setProcessingState(true);
      const method = methodSelect.value as CompressionMethod;
      const k = Number(rankSlider.value);
      setStatus(statusText, `正在执行 ${method.toUpperCase()} 压缩...`);
      await nextFrame();

      const start = performance.now();
      if (method === 'svd') {
        const currentSvdMode = svdModeSelect.value as 'full' | 'randomized';
        const modeLabel = currentSvdMode === 'randomized' ? 'Randomized SVD' : 'Full SVD';
        let result: CompressionResult;
        if (svdWorker) {
          if (!svdWorker.hasPrepared()) {
            setStatus(statusText, `${modeLabel} 预计算中...`);
            await svdWorker.prepare(
              originalImageData,
              (p) => {
                setStatus(statusText, `${modeLabel} 预计算中... ${(p * 100).toFixed(0)}%`);
              },
              currentSvdMode,
              currentSvdMode === 'randomized' ? k : undefined,
            );
          }
          result = await svdWorker.reconstruct(k, (p) => {
            setStatus(statusText, `${modeLabel} 按 rank 重建中... ${(p * 100).toFixed(0)}%`);
          });
        } else {
          // Worker 不可用时，使用非阻塞回退路径，保证页面不假死。
          result = await compressImageBySVDNonBlocking(originalImageData, k, {
            blockSize: 48,
            onProgress: (p) => {
              setStatus(statusText, `${modeLabel} 回退计算中... ${(p * 100).toFixed(0)}%`);
            },
          }, currentSvdMode);
        }
        const elapsed = performance.now() - start;

        putImageDataToCanvas(result.imageData, compressedCanvas);
        fitCanvasPreview(compressedCanvas);
        setStageFilled(compressedStage, true);

        const metrics = evaluateQuality(originalImageData, result.imageData);
        renderMetrics(metricsPanel, metrics, result.estimatedCompressionRatio);

        setStatus(statusText, `${modeLabel} 压缩完成，用时 ${elapsed.toFixed(1)} ms`, 'ok');
        hasCompressedOnce = true;
        return;
      }

      if (method === 'dwt') {
        const dwtOptions: DWTParams = {
          wavelet: waveletSelect.value as DWTParams['wavelet'],
          levels: Number(dwtLevelSlider.value),
          threshold: Number(dwtThresholdSlider.value),
          thresholdMode: thresholdModeSelect.value as DWTParams['thresholdMode'],
        };

        const result = compressImageByDWT(originalImageData, dwtOptions);
        const elapsed = performance.now() - start;

        putImageDataToCanvas(result.imageData, compressedCanvas);
        fitCanvasPreview(compressedCanvas);
        setStageFilled(compressedStage, true);

        const metrics = evaluateQuality(originalImageData, result.imageData);
        renderMetrics(metricsPanel, metrics, result.estimatedCompressionRatio);

        const info = result.notes ? ` (${result.notes})` : '';
        setStatus(statusText, `DWT 压缩完成，用时 ${elapsed.toFixed(1)} ms${info}`, 'ok');
        return;
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : '压缩失败';
      setStatus(statusText, message, 'error');
    } finally {
      setProcessingState(false);
    }
  });

  const updatePreviewByRankForFullSVD = async () => {
    if (!originalImageData) return;
    if (isProcessing) return;
    if (!svdWorker || !svdWorker.hasPrepared()) return;

    const token = ++lastRenderToken;
    const k = Number(rankSlider.value);

    try {
      const result = await svdWorker.reconstruct(k);
      if (token !== lastRenderToken) return;

      putImageDataToCanvas(result.imageData, compressedCanvas);
      fitCanvasPreview(compressedCanvas);
      setStageFilled(compressedStage, true);

      const metrics = evaluateQuality(originalImageData, result.imageData);
      renderMetrics(metricsPanel, metrics, result.estimatedCompressionRatio);
      setStatus(statusText, `rank=${result.rankUsed} 预览已更新`, 'ok');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'rank 预览更新失败';
      setStatus(statusText, message, 'error');
    }
  };

  const onRankChanged = async () => {
    if (!originalImageData) return;
    if (methodSelect.value !== 'svd') return;

    const currentSvdMode = svdModeSelect.value as 'full' | 'randomized';
    if (currentSvdMode === 'randomized') {
      // Randomized SVD recalculates only on "Start Compression".
      if (svdWorker?.hasPrepared()) {
        svdWorker.invalidate();
      }
      return;
    }

    if (!hasCompressedOnce) return;

    await updatePreviewByRankForFullSVD();
  };

  solveSvdRankButton.addEventListener('click', () => {
    if (isProcessing) return;
    if (!originalImageData) {
      setStatus(statusText, '请先上传图片。', 'error');
      return;
    }

    const targetRatio = readTargetRatio();
    if (targetRatio === null) return;

    const { rank, ratio } = findBestSvdRankForTarget(originalImageData.width, originalImageData.height, targetRatio);
    syncRankValue(rank);
    void onRankChanged();
      setStatus(
      statusText,
      `SVD 目标 ${targetRatio.toFixed(2)}x -> k=${rank}（估算 ${formatRatio(ratio)}）`,
      'ok',
    );
  });

  solveDwtThresholdButton.addEventListener('click', async () => {
    if (isProcessing) return;
    if (!originalImageData) {
      setStatus(statusText, '请先上传图片。', 'error');
      return;
    }
    const sourceImage = originalImageData;

    const targetRatio = readTargetRatio();
    if (targetRatio === null) return;
    try {
      solveDwtThresholdButton.disabled = true;
      setStatus(statusText, '正在求解 DWT λ，请稍候...');
      await nextFrame();

      const suggestion = await runInNextTask(() => suggestDWTThresholdForTargetRatio(
        sourceImage,
        {
          wavelet: waveletSelect.value as DWTParams['wavelet'],
          levels: Number(dwtLevelSlider.value),
          thresholdMode: thresholdModeSelect.value as DWTParams['thresholdMode'],
        },
        targetRatio,
        Number(dwtThresholdSlider.min),
        Number(dwtThresholdSlider.max),
      ));

      dwtThresholdSlider.value = String(suggestion.threshold);
      dwtThresholdLabel.textContent = dwtThresholdSlider.value;

      const outOfRange = targetRatio < suggestion.minReachableRatio || targetRatio > suggestion.maxReachableRatio;
      const rangeText = `可达范围 ${formatRatio(suggestion.minReachableRatio)} ~ ${formatRatio(suggestion.maxReachableRatio)}`;
      const closestText = `DWT 目标 ${targetRatio.toFixed(2)}x -> λ=${suggestion.threshold}（估算 ${formatRatio(suggestion.estimatedCompressionRatio)}）`;

      if (outOfRange) {
        setStatus(statusText, `${closestText}，目标超出当前 λ 范围，已取最接近值。${rangeText}`, 'ok');
      } else {
        setStatus(statusText, `${closestText}，${rangeText}`, 'ok');
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'DWT λ 求解失败';
      setStatus(statusText, message, 'error');
    } finally {
      solveDwtThresholdButton.disabled = isProcessing;
    }
  });

  rankSlider.addEventListener('input', () => {
    syncRankValue(Number(rankSlider.value));
    void onRankChanged();
  });

  rankInput.addEventListener('input', () => {
    if (rankInput.value.trim() === '') return;
    const parsed = Number(rankInput.value);
    if (!Number.isFinite(parsed)) return;
    syncRankValue(parsed);
  });

  rankInput.addEventListener('change', () => {
    if (rankInput.value.trim() === '') {
      syncRankValue(Number(rankSlider.value));
      return;
    }
    const parsed = Number(rankInput.value);
    if (!Number.isFinite(parsed)) {
      syncRankValue(Number(rankSlider.value));
      return;
    }
    syncRankValue(parsed);
    void onRankChanged();
  });

  window.addEventListener('beforeunload', () => {
    svdWorker?.dispose();
  });
}

