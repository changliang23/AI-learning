import http from 'http';
import express from 'express';
import { WebSocketServer } from 'ws';
import puppeteer from 'puppeteer';

const PORT = Number(process.env.PORT || 8080);
const START_URL = process.env.START_URL || 'https://example.com';
const TTL_MS = Math.max(30_000, Number(process.env.SESSION_TTL_SEC || 900) * 1000);
const SECRET = process.env.RBI_INTERNAL_SECRET || '';

let ready = false;
let browser;
let page;
const clients = new Set();

setTimeout(() => {
  console.error('RBI session TTL elapsed, exiting');
  process.exit(0);
}, TTL_MS);

const app = express();
app.use(express.json({ limit: '128kb' }));

app.get('/health', (_req, res) => {
  if (!ready) return res.status(503).json({ ok: false });
  res.json({ ok: true });
});

function requireSecret(req, res, next) {
  if (req.headers['x-rbi-secret'] !== SECRET) {
    return res.status(403).json({ error: 'forbidden' });
  }
  next();
}

app.post('/navigate', requireSecret, async (req, res) => {
  const url = req.body?.url;
  if (!url) return res.status(400).json({ error: 'url required' });
  try {
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 120000 });
    res.json({ ok: true });
  } catch (e) {
    res.status(500).json({ error: String(e) });
  }
});

app.post('/input', requireSecret, async (req, res) => {
  const { action, payload } = req.body || {};
  const p = payload || {};
  try {
    if (action === 'mousemove') {
      await page.mouse.move(Number(p.x), Number(p.y));
    } else if (action === 'mousedown') {
      await page.mouse.move(Number(p.x), Number(p.y));
      await page.mouse.down({ button: p.button || 'left' });
    } else if (action === 'mouseup') {
      await page.mouse.move(Number(p.x), Number(p.y));
      await page.mouse.up({ button: p.button || 'left' });
    } else if (action === 'click') {
      await page.mouse.click(Number(p.x), Number(p.y), {
        button: p.button || 'left',
        clickCount: Number(p.clickCount || 1),
      });
    } else if (action === 'wheel') {
      await page.mouse.wheel({ deltaY: Number(p.deltaY || 0) });
    } else if (action === 'keydown') {
      await page.keyboard.down(String(p.key));
    } else if (action === 'keyup') {
      await page.keyboard.up(String(p.key));
    } else if (action === 'type') {
      await page.keyboard.type(String(p.text || ''), { delay: 10 });
    } else {
      return res.status(400).json({ error: 'unknown action' });
    }
    res.json({ ok: true });
  } catch (e) {
    res.status(500).json({ error: String(e) });
  }
});

const server = http.createServer(app);
const wss = new WebSocketServer({ server, path: '/stream' });

wss.on('connection', (ws) => {
  clients.add(ws);
  ws.on('close', () => clients.delete(ws));
});

async function initBrowser() {
  browser = await puppeteer.launch({
    headless: 'new',
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-dev-shm-usage',
      '--disable-breakpad',
      '--disable-infobars',
      '--noerrdialogs',
      '--window-size=1280,720',
    ],
  });

  page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 720, deviceScaleFactor: 1 });

  await page.evaluateOnNewDocument(() => {
    const block = (e) => {
      e.preventDefault();
      e.stopPropagation();
    };
    ['copy', 'cut', 'paste', 'contextmenu'].forEach((ev) => {
      document.addEventListener(ev, block, true);
    });
    document.addEventListener(
      'keydown',
      (e) => {
        if (e.key === 'F12') {
          block(e);
        }
        const k = e.key && e.key.toUpperCase();
        if (e.ctrlKey && e.shiftKey && ['I', 'J', 'C'].includes(k)) {
          block(e);
        }
      },
      true,
    );
  });

  const client = await page.target().createCDPSession();
  await client.send('Page.enable');

  await client.send('Page.startScreencast', {
    format: 'jpeg',
    quality: 55,
    maxWidth: 1280,
    maxHeight: 720,
    everyNthFrame: 1,
  });

  client.on('Page.screencastFrame', async (evt) => {
    const buf = Buffer.from(evt.data, 'base64');
    for (const ws of clients) {
      if (ws.readyState === 1) ws.send(buf);
    }
    try {
      await client.send('Page.screencastFrameAck', { sessionId: evt.sessionId });
    } catch {
      // ignore
    }
  });

  await page.goto(START_URL, { waitUntil: 'domcontentloaded', timeout: 120000 });
  ready = true;
}

server.listen(PORT, async () => {
  console.log('RBI session runner listening on', PORT);
  try {
    await initBrowser();
  } catch (e) {
    console.error(e);
    process.exit(1);
  }
});
