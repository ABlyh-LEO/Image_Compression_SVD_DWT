# SVD 图像压缩（前端作业框架）

本项目为纯前端（TypeScript + Vite）作业框架，重点满足：

1. **SVD 必须手动实现**（不可调用库函数）
2. 基于 SVD 进行图像压缩
3. 使用多个指标评估压缩效果
4. 提供美观 GUI 进行演示

---

## 你需要完成的核心 TODO

- 手写 SVD：
  - [src/core/svd.ts](src/core/svd.ts)
- Optional 扩展预留：
  - DWT： [src/core/dwt.ts](src/core/dwt.ts)

当前 `computeSVD()` 已预留接口与注释，调用时会抛出错误提示你实现。
`DWT` 也已预留模块和 TODO 标识，可在后续阶段逐步接入。

---

## 已完成内容

- 现代化 GUI（浅色风格，适合课程展示）
- 图片上传、压缩按钮
- 方法切换后的参数面板：
  - 选择 SVD：显示 `rank-k` 参数
  - 选择 DWT：显示 `wavelet`、`levels`、`threshold` 参数（TODO 预留）
- 原图/重建图对比显示
- 质量指标：
  - MSE
  - RMSE
  - MAE
  - PSNR
  - SSIM（全局简化版）
  - NCC
- 估计压缩比计算
- **防御式拷贝**（避免图像处理过程修改原数据）

---

## 运行

1. 安装依赖
   - `npm install`
2. 启动开发环境
   - `npm run dev`
3. 打开浏览器访问本地地址

> Windows PowerShell 可能出现 `npm.ps1` 签名策略错误。可直接改用：
> - `npm.cmd install`
> - `npm.cmd run dev`
> - `npm.cmd run build:single`

---

## 打包成“双击 index.html 可用”

可以。已新增“单文件构建”模式，把 JS/CSS 内联到 HTML，避免部分浏览器在 `file://` 下加载模块脚本导致白屏。

### 步骤

1. 安装新依赖（仅第一次需要）：
  - `npm install`
2. 执行单文件构建：
  - `npm run build:single`
2. 进入构建产物目录：
  - `dist/`
3. 直接双击：
  - `dist/index.html`

### 说明

- 普通 `npm run build` 产物仍推荐在本地服务下访问（`npm run preview`）。
- 需要双击离线演示时，请优先使用 `npm run build:single`。
- 如果后续加入依赖网络上下文的功能（例如复杂 Worker/某些 WASM 场景），`file://` 仍可能受限。
- 课堂演示更稳妥的方式仍是：`npm run preview` 后用本地地址访问。

---

## 代码结构

- [src/app.ts](src/app.ts)：页面流程与事件编排
- [src/core/image-io.ts](src/core/image-io.ts)：Canvas 与矩阵数据转换
- [src/core/matrix.ts](src/core/matrix.ts)：矩阵工具与深拷贝
- [src/core/svd.ts](src/core/svd.ts)：**手写 SVD（TODO）**
- [src/core/compress.ts](src/core/compress.ts)：rank-k 截断重建流程
- [src/core/metrics.ts](src/core/metrics.ts)：图像质量指标
- [src/core/dwt.ts](src/core/dwt.ts)：DWT 扩展预留（TODO）

DWT 参数接入 TODO：
- [src/app.ts](src/app.ts)
- [src/core/dwt.ts](src/core/dwt.ts)

---

## 作业撰写建议

建议报告结构：

1. SVD 数学原理（分解、截断近似、存储量分析）
2. 手写 SVD 实现思路（特征值求解方法、复杂度）
3. 实验设计（不同 `k` 下的视觉与指标变化）
4. 结果分析（PSNR/SSIM 与压缩比关系）
5. 可选对比（DWT / 其他方法）
