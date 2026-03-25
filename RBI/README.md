# RBI 零信任浏览器演示（Local Demo）

本仓库演示「远端浏览器隔离（RBI）」的最小可运行骨架：**每个会话独立 Docker 容器**、**到期自动销毁**、**管理台创建会话**、用户侧仅远程观看/操作页面（复制与 DevTools 在浏览器策略与注入层做演示级限制）。

## 架构（与你的结构图对应）

```
用户浏览器
    ↓
Web 前端（React / Vite）
    ↓
后台 API（FastAPI）—— 管理 + 会话调度
    ↓
调度服务（内嵌于 API：后台任务扫描 TTL，销毁容器）
    ↓
Docker 容器（Chromium + Puppeteer，Headless）
    ↓
媒体流：演示使用 CDP Page Screencast → WebSocket（低延迟 MJPEG 风格帧）
```

### 关于「WebRTC」

生产环境 RBI 常见做法是把容器内的显示管线接到 **WebRTC SFU**（如 mediasoup、Janus、LiveKit Egress）。本演示为减少依赖、保证可一键跑通，选用 **CDP Screencast + WebSocket** 作为「视频流」载体；语义上与 WebRTC 同属「隔离渲染、客户端只收媒体」的路径。

对接说明见 `docs/WEBRTC.md`。

## 前置要求

- Docker Desktop（或 Linux Docker）+ Docker Compose v2
- 本机可用 `docker` 命令；Compose 会挂载 `docker.sock` 以便 API 动态创建会话容器

## 快速开始

```bash
cd [YOU PATH]AI-learning/RBI
cp .env.example .env
# 编辑 .env 中的 ADMIN_TOKEN / RBI_INTERNAL_SECRET

# 需要本机 Docker daemon 已启动
make images
make up
```

- 管理 + API：`http://localhost:8787/docs`（也可直接用 Swagger 调 admin 接口）
- 前端（nginx）：`http://localhost:5174` → 管理台路由为 `/#/admin`

若不想使用 `Makefile`，等价命令为：

```bash
docker build -t rbi-browser-session ./browser
docker compose build api frontend
docker compose up -d
```

### 默认账号

- 管理接口在请求头携带：`Authorization: Bearer <ADMIN_TOKEN>`

## 调用流程（演示）

1. `POST /admin/sessions` 创建会话（指定 `ttl_seconds`、`start_url`、`allowed_hosts`）
2. 浏览器打开前端「观看页」，粘贴 `viewer_token`
3. 会话到期后容器被停止删除；同时会话记录清理

## 安全说明（必读）

这是 **演示工程**，不是商用零信任产品：

- `viewer_token` 为演示用共享密钥；生产应使用短期 JWT、设备绑定、mTLS
- 反复制 / 禁 DevTools 在浏览器侧可被专业用户绕过；RBI 的核心价值在「数据不出容器 + 出站策略 + 审计」
- 务必将 `ADMIN_TOKEN`、`RBI_INTERNAL_SECRET` 设为强随机
