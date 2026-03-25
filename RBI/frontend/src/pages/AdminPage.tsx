import { useCallback, useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { apiUrl } from '../api';

type AdminSession = {
  session_id: string;
  container_name: string;
  start_url: string;
  allowed_hosts: string[];
  created_at: number;
  expires_at: number;
};

type Created = {
  session_id: string;
  viewer_token: string;
  expires_at: number;
  viewer_path: string;
};

const LS_KEY = 'rbi_admin_token';

export default function AdminPage() {
  const [token, setToken] = useState(() => localStorage.getItem(LS_KEY) || '');
  const [startUrl, setStartUrl] = useState('https://example.com');
  const [allowedHosts, setAllowedHosts] = useState('example.com');
  const [ttlSeconds, setTtlSeconds] = useState(900);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [created, setCreated] = useState<Created | null>(null);
  const [rows, setRows] = useState<AdminSession[]>([]);

  const authHeader = useMemo(() => ({ Authorization: `Bearer ${token}` }), [token]);

  useEffect(() => {
    localStorage.setItem(LS_KEY, token);
  }, [token]);

  const refreshList = useCallback(async () => {
    setError(null);
    const r = await fetch(apiUrl('/admin/sessions'), { headers: { Authorization: `Bearer ${token}` } });
    if (!r.ok) {
      setError(await r.text());
      return;
    }
    setRows((await r.json()) as AdminSession[]);
  }, [token]);

  async function createSession() {
    setBusy(true);
    setError(null);
    setCreated(null);
    try {
      const hosts = allowedHosts
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean);
      const r = await fetch(apiUrl('/admin/sessions'), {
        method: 'POST',
        headers: { ...authHeader, 'Content-Type': 'application/json' },
        body: JSON.stringify({
          start_url: startUrl,
          allowed_hosts: hosts,
          ttl_seconds: ttlSeconds,
        }),
      });
      if (!r.ok) throw new Error(await r.text());
      const data = (await r.json()) as Created;
      setCreated(data);
      await refreshList();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function deleteSession(id: string) {
    setError(null);
    const r = await fetch(apiUrl(`/admin/sessions/${id}`), { method: 'DELETE', headers: authHeader });
    if (!r.ok) {
      setError(await r.text());
      return;
    }
    await refreshList();
  }

  useEffect(() => {
    void refreshList();
  }, [refreshList]);

  const viewerLink =
    created &&
    `${window.location.origin}${window.location.pathname}#/view?s=${created.session_id}&t=${created.viewer_token}`;

  return (
    <div className="page">
      <header className="header">
        <div>
          <div className="title">RBI 管理台（演示）</div>
          <div className="sub">创建隔离会话、设置 TTL 与域名白名单</div>
        </div>
        <Link className="link" to="/view">
          打开观看页
        </Link>
      </header>

      <section className="card">
        <div className="row">
          <label className="label">ADMIN_TOKEN</label>
          <input className="input" value={token} onChange={(e) => setToken(e.target.value)} autoComplete="off" />
        </div>
        <div className="row">
          <label className="label">起始 URL</label>
          <input className="input" value={startUrl} onChange={(e) => setStartUrl(e.target.value)} />
        </div>
        <div className="row">
          <label className="label">允许的域名（逗号分隔）</label>
          <input className="input" value={allowedHosts} onChange={(e) => setAllowedHosts(e.target.value)} />
        </div>
        <div className="row">
          <label className="label">TTL（秒）</label>
          <input
            className="input"
            type="number"
            value={ttlSeconds}
            min={30}
            max={86400}
            onChange={(e) => setTtlSeconds(Number(e.target.value))}
          />
        </div>

        <div className="actions">
          <button className="btn" disabled={busy || !token} onClick={() => void createSession()}>
            创建会话
          </button>
          <button className="btn secondary" disabled={!token} onClick={() => void refreshList()}>
            刷新列表
          </button>
        </div>

        {error && <div className="error">{error}</div>}

        {created && (
          <div className="success">
            <div>
              <div className="muted">session_id</div>
              <div className="mono">{created.session_id}</div>
            </div>
            <div>
              <div className="muted">viewer_token（保密）</div>
              <div className="mono">{created.viewer_token}</div>
            </div>
            <div>
              <div className="muted">一键打开</div>
              <a className="link" href={viewerLink || '#'} rel="noreferrer">
                {viewerLink}
              </a>
            </div>
          </div>
        )}
      </section>

      <section className="card">
        <div className="title sm">会话列表</div>
        <table className="table">
          <thead>
            <tr>
              <th>session</th>
              <th>容器</th>
              <th>起始</th>
              <th>到期（epoch）</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {rows.map((s) => (
              <tr key={s.session_id}>
                <td className="mono">{s.session_id}</td>
                <td className="mono">{s.container_name}</td>
                <td className="ellipsis">{s.start_url}</td>
                <td className="mono">{Math.floor(s.expires_at)}</td>
                <td>
                  <button className="btn danger" onClick={() => void deleteSession(s.session_id)}>
                    销毁
                  </button>
                </td>
              </tr>
            ))}
            {rows.length === 0 && (
              <tr>
                <td colSpan={5} className="muted">
                  暂无会话
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </section>
    </div>
  );
}
