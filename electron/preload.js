import { contextBridge, ipcRenderer } from 'electron';

contextBridge.exposeInMainWorld('desktopActions', {
  openChrome: () => ipcRenderer.invoke('desktop:open-chrome')
});
