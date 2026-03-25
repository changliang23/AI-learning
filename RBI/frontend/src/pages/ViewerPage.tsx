import { useEffect, useMemo, useRef, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { apiUrl } from '../api';

const VIEW_W = 1280;
const VIEW_H = 720;

export default function ViewerPage() {
  const [params] = useSearchParams();
  const sid = params.get('s') || '';
  const token = params.get('t') || '';

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [status, setStatus] = useState<string>('disconnected');
  const [error, setError] = useState<string | null>(null);

  const canRun = Boolean(sid && token);

  const wsUrl = useMemo(() => {
    if (!canRun) return '';
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const path = `/ws/session/${encodeURIComponent(sid)}/stream`;
    return `${proto}//${host}${path}?token=${encodeURIComponent(token)}`;
  }, [canRun, sid, token]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !wsUrl) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = VIEW_W;
    canvas.height = VIEW_H;

    const ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';
    ws.onopen = () => {
      setError(null);
      setStatus('connected');
    };
    ws.onclose = () => setStatus('disconnected');
    ws.onerror = () => setError('WebSocket error');

    let cancelled = false;

    ws.onmessage = async (ev) => {
      if (cancelled) return;
      try {
        const bytes = new Uint8Array(ev.data as ArrayBuffer);
        const blob = new Blob([bytes], { type: 'image/jpeg' });
        const bmp = await createImageBitmap(blob);
        ctx.drawImage(bmp, 0, 0, VIEW_W, VIEW_H);
        bmp.close();
      } catch {
        // ignore bad frames
      }
    };

    return () => {
      cancelled = true;
      try {
        ws.close();
      } catch {
        // ignore
      }
    };
  }, [wsUrl]);

  function scalePoint(clientX: number, clientY: number) {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    const x = ((clientX - rect.left) / rect.width) * VIEW_W;
    const y = ((clientY - rect.top) / rect.height) * VIEW_H;
    return { x, y };
  }

  async function sendInput(body: unknown) {
    if (!canRun) return;
    const r = await fetch(apiUrl(`/api/sessions/${encodeURIComponent(sid)}/input?token=${encodeURIComponent(token)}`), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) setError(await r.text());
  }

  const [navUrl, setNavUrl] = useState('https://example.com');

  async function navigate() {
    if (!canRun) return;
    setError(null);
    const r = await fetch(apiUrl(`/api/sessions/${encodeURIComponent(sid)}/navigate?token=${encodeURIComponent(token)}`), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: navUrl }),
    });
    if (!r.ok) setError(await r.text());
  }

  return (
    <div className="page">
      <header className="header">
        <div>
          <div className="title">隔离浏览器观看端（演示）</div>
          <div className="sub">
            状态：<span className="mono">{status}</span> · 画面来自 CDP Screencast（WebSocket）
          </div>
        </div>
        <Link className="link" to="/admin">
          回到管理台
        </Link>
      </header>

      {!canRun && (
        <section className="card">
          <div className="muted">
            请在 Hash 路由参数中提供 <span className="mono">s</span>（session_id）与 <span className="mono">t</span>
            （viewer_token），或从管理台一键打开。
          </div>
        </section>
      )}

      {canRun && (
        <>
          <section className="card rowgap viewer">
            <div className="row">
              <label className="label">导航（受白名单约束）</label>
              <input className="input" value={navUrl} onChange={(e) => setNavUrl(e.target.value)} />
              <button className="btn secondary" onClick={() => void navigate()}>
                跳转
              </button>
            </div>
            {error && <div className="error">{error}</div>}
            <div className="muted">
              说明：站点侧仍可尝试绕过，演示目标是隔离容器 + 网关策略 + 审计。
            </div>
          </section>

          <section className="card viewer">
            <div
              className="canvasWrap"
              onContextMenu={(e) => e.preventDefault()}
              onMouseMove={(e) => {
                const { x, y } = scalePoint(e.clientX, e.clientY);
                void sendInput({ action: 'mousemove', payload: { x, y } });
              }}
              onMouseDown={(e) => {
                const { x, y } = scalePoint(e.clientX, e.clientY);
                const button = e.button === 2 ? 'right' : e.button === 1 ? 'middle' : 'left';
                void sendInput({ action: 'mousedown', payload: { x, y, button } });
              }}
              onMouseUp={(e) => {
                const { x, y } = scalePoint(e.clientX, e.clientY);
                const button = e.button === 2 ? 'right' : e.button === 1 ? 'middle' : 'left';
                void sendInput({ action: 'mouseup', payload: { x, y, button } });
              }}
              onWheel={(e) => {
                e.preventDefault();
                void sendInput({ action: 'wheel', payload: { deltaY: e.deltaY } });
              }}
            >
              <canvas ref={canvasRef} className="canvas" />
            </div>
          </section>
        </>
      )}
    </div>
  );
}
