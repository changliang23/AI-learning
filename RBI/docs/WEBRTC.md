# WebRTC 对接说明（生产向）

## 当前演示

会话容器内使用 **Chromium + Puppeteer**，通过 **Chrome DevTools Protocol** 的 `Page.startScreencast` 将画面帧推到 API，再由 **WebSocket** 传给前端 `<canvas>`。

## 为什么要换成 WebRTC

- **规模化**：SFU 集群更适合多路并发、带宽自适应
- **终端兼容**：WebRTC 在弱网下的卡顿 recovery 更成熟
- **企业集成**：可与既有会议 / 协作系统的 TURN / SSO 统一

## 推荐落地路径（摘要）

1. 容器内仍跑隔离 Chromium（或 Kasm / Selkies 一类方案）
2. 用 **GStreamer / FFmpeg** 抓取显示管线（X11/Wayland/pipewire）
3. 将 H.264/VP8 编码后送入 **mediasoup / Janus / LiveKit** 作为 Publisher
4. 前端用标准 `RTCPeerConnection` 订阅；输入事件走 **DataChannel** 或独立 REST（带会话网关）

本仓库的 `api` 与前端可保留「会话 / TTL / 白名单」逻辑，仅替换「媒体平面」实现。
