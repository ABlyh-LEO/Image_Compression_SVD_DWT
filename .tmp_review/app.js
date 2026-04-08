import { compressImageBySVDNonBlocking } from './core/compress';
import { compressImageByDWT } from './core/dwt';
import SVDChannelWorkerConstructor from './workers/svd-channel.worker.ts?worker&inline';
import { drawImageToCanvas, fileToImageElement, getImageDataFromCanvas, putImageDataToCanvas, } from './core/image-io';
import { evaluateQuality } from './core/metrics';
import { renderMetrics, setStatus } from './ui/controls';
import { fitCanvasPreview } from './ui/render';
function createSVDWorkerClient() {
    const cpuCores = Math.max(1, navigator.hardwareConcurrency ?? 4);
    // 保持算法正确性前提下，独立重任务最多是 RGB 三通道；这里按设备核心数自适应。
    const workerCount = Math.max(1, Math.min(3, cpuCores));
    const workers = Array.from({ length: workerCount }, () => new SVDChannelWorkerConstructor());
    let seq = 1;
    const runTask = (worker, payload, transferList) => {
        const id = seq;
        seq += 1;
        return new Promise((resolve, reject) => {
            const onMessage = (event) => {
                const response = event.data;
                if (response.id !== id)
                    return;
                worker.removeEventListener('message', onMessage);
                worker.removeEventListener('error', onError);
                if (response.ok) {
                    resolve(response);
                }
                else {
                    reject(new Error(response.error));
                }
            };
            const onError = (event) => {
                worker.removeEventListener('message', onMessage);
                worker.removeEventListener('error', onError);
                reject(new Error(event.message || 'SVD Worker 运行错误'));
            };
            worker.addEventListener('message', onMessage);
            worker.addEventListener('error', onError);
            const request = { id, ...payload };
            worker.postMessage(request, transferList ?? []);
        });
    };
    const splitChannels = (imageData) => {
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
    const mergeChannels = (width, height, r, g, b) => {
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
    const perChannelKCache = new Map();
    const clampRank = (k) => Math.max(1, Math.min(k, Math.min(imageWidth, imageHeight)));
    const computeRatio = (k) => {
        const originalStorage = 3 * imageWidth * imageHeight;
        const compressedStorage = 3 * k * (imageHeight + imageWidth + 1);
        return originalStorage / compressedStorage;
    };
    return {
        async prepare(imageData, onProgress, svdMode = 'full', rank) {
            const safeCopy = new ImageData(new Uint8ClampedArray(imageData.data), imageData.width, imageData.height);
            const { r, g, b, width, height } = splitChannels(safeCopy);
            imageWidth = width;
            imageHeight = height;
            perChannelKCache.clear();
            prepared = false;
            const channels = [r, g, b];
            let done = 0;
            await Promise.all(channels.map(async (channel, idx) => {
                const worker = workers[idx % workers.length];
                await runTask(worker, {
                    type: 'prepare',
                    channel,
                    width,
                    height,
                    svdMode,
                    rank,
                }, [channel.buffer]);
                done += 1;
                onProgress?.(done / 3);
            }));
            prepared = true;
        },
        async reconstruct(k, onProgress) {
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
            const outputs = new Array(3);
            let done = 0;
            await Promise.all([0, 1, 2].map(async (idx) => {
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
            }));
            perChannelKCache.set(kClamped, [outputs[0], outputs[1], outputs[2]]);
            const image = mergeChannels(imageWidth, imageHeight, outputs[0], outputs[1], outputs[2]);
            return {
                imageData: image,
                rankUsed: kClamped,
                estimatedCompressionRatio: computeRatio(kClamped),
            };
        },
        hasPrepared() {
            return prepared;
        },
        invalidate() {
            prepared = false;
            perChannelKCache.clear();
            workers.forEach((worker) => {
                void runTask(worker, { type: 'dispose' });
            });
        },
        dispose() {
            workers.forEach((w) => w.terminate());
        },
    };
}
function nextFrame() {
    return new Promise((resolve) => {
        requestAnimationFrame(() => resolve());
    });
}
function template() {
    return `
    <main class="shell">
      <section class="hero glass">
        <h1>SVD 图像压缩实验台</h1>
        <p>
          纯前端实现 · SVD（手写）+ DWT（可选扩展） · 多指标评估
        </p>
      </section>

      <section class="panel glass controls">
        <div class="algo-group">
          <label for="methodSelect">压缩方法</label>
          <select id="methodSelect">
            <option value="svd">SVD 压缩</option>
            <option value="dwt">DWT 压缩</option>
          </select>
        </div>

        <label class="file-upload">
          <span>选择图片</span>
          <input id="fileInput" type="file" accept="image/*" />
        </label>

        <div class="param-panels">
          <div id="svdParams" class="param-panel">
            <h4>SVD 参数</h4>
            <div class="algo-group">
              <label for="svdModeSelect">SVD 模式</label>
              <select id="svdModeSelect">
                <option value="full">全分解（精确）</option>
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
            <p id="svdModeHint" class="todo-hint" style="display:none">
              随机 SVD 模式下，修改 rank 后需点击“开始压缩”才会重新分解。
            </p>
          </div>

          <div id="dwtParams" class="param-panel hidden">
            <h4>DWT 参数</h4>
            <div class="dwt-grid">
              <div class="dwt-item">
                <label for="waveletSelect">小波基</label>
                <select id="waveletSelect">
                  <option value="haar">Haar</option>
                  <option value="db2">Daubechies-2</option>
                  <option value="db4">Daubechies-4</option>
                </select>
                <span class="hint">Haar 最快但有块效应；DB4 更平滑但更慢</span>
              </div>

              <div class="dwt-item">
                <label for="dwtLevelSlider">分解层数: <strong id="dwtLevelLabel">2</strong></label>
                <input id="dwtLevelSlider" type="range" min="1" max="6" value="2" />
                <span class="hint">层数越多，低频越集中，同等 λ 下压缩越强</span>
              </div>

              <div class="dwt-item">
                <label for="dwtThresholdSlider">阈值强度 λ: <strong id="dwtThresholdLabel">20</strong></label>
                <input id="dwtThresholdSlider" type="range" min="0" max="100" value="20" />
                <span class="hint">λ=0 无损；λ↑ 压缩率↑ 但质量↓（推荐 10~40）</span>
              </div>

              <div class="dwt-item">
                <label for="thresholdModeSelect">阈值策略</label>
                <select id="thresholdModeSelect">
                  <option value="hard">硬阈值</option>
                  <option value="soft">软阈值</option>
                </select>
                <span class="hint">硬阈值保留锐度；软阈值更平滑无振铃</span>
              </div>
            </div>
          </div>
        </div>

        <div class="button-row">
          <button id="compressButton" disabled>开始压缩</button>
          <button id="stopButton" class="stop-btn" style="display:none">停止计算</button>
        </div>
        <p id="statusText" class="status">请先上传一张图片。</p>
      </section>

      <section class="panel glass canvas-grid">
        <article>
          <h3>原图</h3>
          <canvas id="originalCanvas"></canvas>
        </article>
        <article>
          <h3>压缩重建图</h3>
          <canvas id="compressedCanvas"></canvas>
        </article>
      </section>

      <section class="panel glass metrics">
        <h3>压缩效果指标</h3>
        <div id="metricsPanel" class="metric-grid"></div>
      </section>
    </main>
  `;
}
export function initApp(root) {
    root.innerHTML = template();
    // 优先尝试 Worker；若浏览器策略限制再自动降级到主线程。
    let svdWorker = null;
    let workerDisabledReason = '';
    try {
        svdWorker = createSVDWorkerClient();
    }
    catch (error) {
        workerDisabledReason = error instanceof Error ? error.message : 'Worker 初始化失败';
    }
    const fileInput = document.querySelector('#fileInput');
    const methodSelect = document.querySelector('#methodSelect');
    const svdParams = document.querySelector('#svdParams');
    const svdModeSelect = document.querySelector('#svdModeSelect');
    const svdModeHint = document.querySelector('#svdModeHint');
    const dwtParams = document.querySelector('#dwtParams');
    const rankSlider = document.querySelector('#rankSlider');
    const rankInput = document.querySelector('#rankInput');
    const rankLabel = document.querySelector('#rankLabel');
    const waveletSelect = document.querySelector('#waveletSelect');
    const dwtLevelSlider = document.querySelector('#dwtLevelSlider');
    const dwtLevelLabel = document.querySelector('#dwtLevelLabel');
    const dwtThresholdSlider = document.querySelector('#dwtThresholdSlider');
    const dwtThresholdLabel = document.querySelector('#dwtThresholdLabel');
    const thresholdModeSelect = document.querySelector('#thresholdModeSelect');
    const compressButton = document.querySelector('#compressButton');
    const stopButton = document.querySelector('#stopButton');
    const statusText = document.querySelector('#statusText');
    const metricsPanel = document.querySelector('#metricsPanel');
    const originalCanvas = document.querySelector('#originalCanvas');
    const compressedCanvas = document.querySelector('#compressedCanvas');
    if (!fileInput ||
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
        !compressButton ||
        !stopButton ||
        !statusText ||
        !metricsPanel ||
        !originalCanvas ||
        !compressedCanvas) {
        throw new Error('UI 初始化失败：存在缺失 DOM 节点');
    }
    const syncRankValue = (raw) => {
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
    const bindNumberDisplay = (input, label) => {
        const sync = () => {
            label.textContent = input.value;
        };
        sync();
        input.addEventListener('input', sync);
    };
    bindNumberDisplay(dwtLevelSlider, dwtLevelLabel);
    bindNumberDisplay(dwtThresholdSlider, dwtThresholdLabel);
    const syncControlByMethod = () => {
        const method = methodSelect.value;
        const isSVD = method === 'svd';
        svdParams.classList.toggle('hidden', !isSVD);
        dwtParams.classList.toggle('hidden', isSVD);
        rankSlider.disabled = !isSVD;
        rankInput.disabled = !isSVD;
        rankSlider.parentElement?.classList.toggle('disabled', !isSVD);
    };
    const syncSvdModeHint = () => {
        const isRandom = svdModeSelect.value === 'randomized';
        svdModeHint.style.display = isRandom ? 'block' : 'none';
    };
    let originalImageData = null;
    let isProcessing = false;
    let lastRenderToken = 0;
    let hasCompressedOnce = false;
    const setProcessingState = (processing) => {
        isProcessing = processing;
        methodSelect.disabled = processing;
        fileInput.disabled = processing;
        compressButton.disabled = processing || !originalImageData;
        compressButton.style.display = processing ? 'none' : '';
        stopButton.style.display = processing ? '' : 'none';
        rankSlider.disabled = processing || methodSelect.value !== 'svd';
        rankInput.disabled = processing || methodSelect.value !== 'svd';
    };
    syncControlByMethod();
    syncSvdModeHint();
    methodSelect.addEventListener('change', () => {
        syncControlByMethod();
        syncSvdModeHint();
    });
    svdModeSelect.addEventListener('change', () => {
        syncSvdModeHint();
        // 切换模式时清除缓存，需重新 prepare
        svdWorker?.invalidate();
    });
    stopButton.addEventListener('click', () => {
        if (!isProcessing)
            return;
        // 使所有待处理的异步结果失效
        lastRenderToken += 1;
        // 终止现有 Worker 并重建
        if (svdWorker) {
            svdWorker.dispose();
            svdWorker = null;
            try {
                svdWorker = createSVDWorkerClient();
            }
            catch (error) {
                workerDisabledReason = error instanceof Error ? error.message : 'Worker 重建失败';
            }
        }
        hasCompressedOnce = false;
        setProcessingState(false);
        setStatus(statusText, '计算已停止。', 'error');
    });
    fileInput.addEventListener('change', async () => {
        if (isProcessing)
            return;
        const file = fileInput.files?.[0];
        if (!file)
            return;
        try {
            setStatus(statusText, '正在加载图片...');
            const image = await fileToImageElement(file);
            drawImageToCanvas(image, originalCanvas);
            fitCanvasPreview(originalCanvas);
            const loaded = getImageDataFromCanvas(originalCanvas);
            // 显式拷贝，保证后续任何压缩流程都不会污染“原图”数据。
            originalImageData = new ImageData(new Uint8ClampedArray(loaded.data), loaded.width, loaded.height);
            svdWorker?.invalidate();
            hasCompressedOnce = false;
            compressedCanvas.width = originalCanvas.width;
            compressedCanvas.height = originalCanvas.height;
            compressedCanvas.getContext('2d')?.clearRect(0, 0, compressedCanvas.width, compressedCanvas.height);
            fitCanvasPreview(compressedCanvas);
            const maxRank = Math.max(1, Math.min(originalCanvas.width, originalCanvas.height));
            rankSlider.max = String(maxRank);
            rankInput.max = String(maxRank);
            if (Number(rankSlider.value) > maxRank) {
                syncRankValue(Math.max(1, Math.floor(maxRank / 4)));
            }
            else {
                syncRankValue(Number(rankSlider.value));
            }
            compressButton.disabled = false;
            metricsPanel.innerHTML = '';
            if (workerDisabledReason) {
                setStatus(statusText, `图片加载完成：${loaded.width} x ${loaded.height}（${workerDisabledReason}）`, 'ok');
            }
            else {
                setStatus(statusText, `图片加载完成：${loaded.width} x ${loaded.height}`, 'ok');
            }
        }
        catch (error) {
            const msg = error instanceof Error ? error.message : '加载图片失败';
            setStatus(statusText, msg, 'error');
        }
    });
    compressButton.addEventListener('click', async () => {
        if (isProcessing)
            return;
        if (!originalImageData) {
            setStatus(statusText, '请先上传图片', 'error');
            return;
        }
        try {
            setProcessingState(true);
            const method = methodSelect.value;
            const k = Number(rankSlider.value);
            setStatus(statusText, `正在执行 ${method.toUpperCase()} 压缩...`);
            await nextFrame();
            const start = performance.now();
            if (method === 'svd') {
                const currentSvdMode = svdModeSelect.value;
                const modeLabel = currentSvdMode === 'randomized' ? '随机 SVD' : 'SVD 全分解';
                let result;
                if (svdWorker) {
                    if (!svdWorker.hasPrepared()) {
                        setStatus(statusText, `${modeLabel} 预计算中...`);
                        await svdWorker.prepare(originalImageData, (p) => {
                            setStatus(statusText, `${modeLabel} 预计算中... ${(p * 100).toFixed(0)}%`);
                        }, currentSvdMode, currentSvdMode === 'randomized' ? k : undefined);
                    }
                    result = await svdWorker.reconstruct(k, (p) => {
                        setStatus(statusText, `${modeLabel} 按 rank 重建中... ${(p * 100).toFixed(0)}%`);
                    });
                }
                else {
                    // Worker 不可用时，使用非阻塞块 SVD 回退，确保页面不假死。
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
                const metrics = evaluateQuality(originalImageData, result.imageData);
                renderMetrics(metricsPanel, metrics, result.estimatedCompressionRatio);
                setStatus(statusText, `${modeLabel} 压缩完成，用时 ${elapsed.toFixed(1)} ms`, 'ok');
                hasCompressedOnce = true;
                return;
            }
            if (method === 'dwt') {
                const dwtOptions = {
                    wavelet: waveletSelect.value,
                    levels: Number(dwtLevelSlider.value),
                    threshold: Number(dwtThresholdSlider.value),
                    thresholdMode: thresholdModeSelect.value,
                };
                const result = compressImageByDWT(originalImageData, dwtOptions);
                const elapsed = performance.now() - start;
                putImageDataToCanvas(result.imageData, compressedCanvas);
                fitCanvasPreview(compressedCanvas);
                const metrics = evaluateQuality(originalImageData, result.imageData);
                renderMetrics(metricsPanel, metrics, result.estimatedCompressionRatio);
                const info = result.notes ? ` (${result.notes})` : '';
                setStatus(statusText, `DWT 压缩完成，用时 ${elapsed.toFixed(1)} ms${info}`, 'ok');
                return;
            }
        }
        catch (error) {
            const message = error instanceof Error ? error.message : '压缩失败';
            setStatus(statusText, message, 'error');
        }
        finally {
            setProcessingState(false);
        }
    });
    const updatePreviewByRankForFullSVD = async () => {
        if (!originalImageData)
            return;
        if (isProcessing)
            return;
        if (!svdWorker || !svdWorker.hasPrepared())
            return;
        const token = ++lastRenderToken;
        const k = Number(rankSlider.value);
        try {
            const result = await svdWorker.reconstruct(k);
            if (token !== lastRenderToken)
                return;
            putImageDataToCanvas(result.imageData, compressedCanvas);
            fitCanvasPreview(compressedCanvas);
            const metrics = evaluateQuality(originalImageData, result.imageData);
            renderMetrics(metricsPanel, metrics, result.estimatedCompressionRatio);
            setStatus(statusText, `rank=${result.rankUsed} 预览已更新`, 'ok');
        }
        catch (error) {
            const message = error instanceof Error ? error.message : 'rank 预览更新失败';
            setStatus(statusText, message, 'error');
        }
    };
    const onRankChanged = async () => {
        if (!originalImageData)
            return;
        if (methodSelect.value !== 'svd')
            return;
        const currentSvdMode = svdModeSelect.value;
        if (currentSvdMode === 'randomized') {
            // 随机 SVD 在点击“开始压缩”时才应用 rank；先失效旧缓存，避免使用旧分解结果。
            if (svdWorker?.hasPrepared()) {
                svdWorker.invalidate();
            }
            return;
        }
        if (!hasCompressedOnce)
            return;
        await updatePreviewByRankForFullSVD();
    };
    rankSlider.addEventListener('input', () => {
        syncRankValue(Number(rankSlider.value));
        void onRankChanged();
    });
    rankInput.addEventListener('input', () => {
        if (rankInput.value.trim() === '')
            return;
        const parsed = Number(rankInput.value);
        if (!Number.isFinite(parsed))
            return;
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
