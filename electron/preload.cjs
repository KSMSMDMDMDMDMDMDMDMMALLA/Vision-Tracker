const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('desktopActions', {
  openChrome: () => ipcRenderer.invoke('desktop:open-chrome')
});
