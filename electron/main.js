import { app, BrowserWindow, ipcMain, net, protocol, session, shell } from 'electron';
import path from 'node:path';
import { execFile, spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, '..');
const DIST = path.join(ROOT, 'dist');
const isDev = process.argv.includes('--dev');

const CHROME_URL = 'https://www.google.com/';

function spawnDetached(command, args = []) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      detached: true,
      stdio: 'ignore',
      windowsHide: true
    });

    child.once('error', reject);
    child.once('spawn', () => {
      child.unref();
      resolve(true);
    });
  });
}

function execFileText(command, args = []) {
  return new Promise((resolve, reject) => {
    execFile(command, args, { windowsHide: true }, (error, stdout) => {
      if (error) {
        reject(error);
        return;
      }
      resolve(String(stdout || ''));
    });
  });
}

async function findChromeOnWindows() {
  const candidates = [
    process.env.LOCALAPPDATA && path.join(process.env.LOCALAPPDATA, 'Google', 'Chrome', 'Application', 'chrome.exe'),
    process.env.PROGRAMFILES && path.join(process.env.PROGRAMFILES, 'Google', 'Chrome', 'Application', 'chrome.exe'),
    process.env['PROGRAMFILES(X86)'] && path.join(process.env['PROGRAMFILES(X86)'], 'Google', 'Chrome', 'Application', 'chrome.exe')
  ].filter(Boolean);

  for (const candidate of candidates) {
    if (existsSync(candidate)) return candidate;
  }

  const registryKeys = [
    'HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\App Paths\\chrome.exe',
    'HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\App Paths\\chrome.exe',
    'HKLM\\SOFTWARE\\WOW6432Node\\Microsoft\\Windows\\CurrentVersion\\App Paths\\chrome.exe'
  ];

  for (const key of registryKeys) {
    try {
      const output = await execFileText('reg.exe', ['query', key, '/ve']);
      const match = output.match(/REG_SZ\s+(.+?)\s*$/im);
      const chromePath = match?.[1]?.trim().replace(/^"|"$/g, '');
      if (chromePath && existsSync(chromePath)) return chromePath;
    } catch {

    }
  }

  return null;
}

async function openChrome(url = CHROME_URL) {
  try {
    if (process.platform === 'win32') {
      const chromePath = await findChromeOnWindows();

      if (chromePath) {
        await spawnDetached(chromePath, ['--new-window', url]);
        return { ok: true, browser: 'chrome', path: chromePath };
      }

      try {
        await spawnDetached('cmd.exe', [
          '/d',
          '/s',
          '/c',
          'start',
          '',
          'chrome.exe',
          '--new-window',
          url
        ]);
        return { ok: true, browser: 'chrome-shell' };
      } catch (error) {
        console.warn('Chrome через Windows Shell не запустился:', error);
      }
    } else if (process.platform === 'darwin') {
      try {
        await spawnDetached('open', ['-a', 'Google Chrome', url]);
        return { ok: true, browser: 'chrome' };
      } catch {

      }
    } else {
      for (const command of ['google-chrome', 'google-chrome-stable', 'chromium', 'chromium-browser']) {
        try {
          await spawnDetached(command, ['--new-window', url]);
          return { ok: true, browser: 'chrome' };
        } catch {
          
        }
      }
    }

  
    await shell.openExternal(url);
    return { ok: true, browser: 'default' };
  } catch (error) {
    console.error('Не удалось открыть браузер:', error);
    return { ok: false, error: error?.message || String(error) };
  }
}
function configureDesktopActions() {
  ipcMain.handle('desktop:open-chrome', async (event) => {
    const senderUrl = event.senderFrame?.url || event.sender?.getURL?.() || '';

    if (!isAllowedOrigin(senderUrl)) {
      return { ok: false, error: 'Blocked origin' };
    }

    return openChrome(CHROME_URL);
  });
}

protocol.registerSchemesAsPrivileged([
  {
    scheme: 'faceapp',
    privileges: {
      standard: true,
      secure: true,
      supportFetchAPI: true,
      stream: true,
      codeCache: true
    }
  }
]);

function isAllowedOrigin(url = '') {
  return (
    url.startsWith('faceapp://bundle/') ||
    url === 'faceapp://bundle' ||
    url.startsWith('http://127.0.0.1:5173/') ||
    url === 'http://127.0.0.1:5173'
  );
}

function configurePermissions() {
  session.defaultSession.setPermissionCheckHandler(
    (webContents, permission, requestingOrigin, details) => {
      if (permission !== 'media') return false;

      const currentUrl =
        details?.requestingUrl ||
        requestingOrigin ||
        webContents?.getURL?.() ||
        '';

      return isAllowedOrigin(currentUrl);
    }
  );

  session.defaultSession.setPermissionRequestHandler(
    (webContents, permission, callback, details) => {
      if (permission !== 'media') {
        callback(false);
        return;
      }

      const currentUrl =
        details?.requestingUrl || webContents?.getURL?.() || '';

      callback(isAllowedOrigin(currentUrl));
    }
  );
}

async function registerProductionProtocol() {
  if (isDev) return;

  await protocol.handle('faceapp', (request) => {
    const url = new URL(request.url);

    if (url.host !== 'bundle') {
      return new Response('Not found', { status: 404 });
    }

    const requestedPath = decodeURIComponent(url.pathname);
    const relativePath = requestedPath === '/'
      ? 'index.html'
      : requestedPath.replace(/^\/+/, '');

    const filePath = path.resolve(DIST, relativePath);
    const relative = path.relative(DIST, filePath);
    const isSafe = relative && !relative.startsWith('..') && !path.isAbsolute(relative);

    if (!isSafe && relativePath !== 'index.html') {
      return new Response('Forbidden', { status: 403 });
    }

    return net.fetch(pathToFileURL(filePath).toString());
  });
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1080,
    height: 760,
    minWidth: 760,
    minHeight: 540,
    backgroundColor: '#080b10',
    autoHideMenuBar: true,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      preload: path.join(__dirname, 'preload.cjs')
    }
  });

  if (isDev) {
    win.loadURL('http://127.0.0.1:5173');
  } else {
    win.loadURL('faceapp://bundle/index.html');
  }
}

app.whenReady().then(async () => {
  configurePermissions();
  configureDesktopActions();
  await registerProductionProtocol();
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});
