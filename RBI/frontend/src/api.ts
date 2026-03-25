const prefix = (import.meta.env.VITE_API_BASE || '').replace(/\/$/, '');

export function apiUrl(path: string): string {
  if (path.startsWith('http')) return path;
  return `${prefix}${path}`;
}
