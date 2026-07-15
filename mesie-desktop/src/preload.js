/**
 * MESIE Desktop — Preload Script
 *
 * Exposes safe IPC methods to the renderer process via contextBridge.
 */

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('mesie', {
  // SDK Operations
  getVersion: () => ipcRenderer.invoke('mesie:version'),
  validate: (recordPath) => ipcRenderer.invoke('mesie:validate', recordPath),
  generate: (opts) => ipcRenderer.invoke('mesie:generate', opts),
  match: (opts) => ipcRenderer.invoke('mesie:match', opts),
  embed: (recordPath) => ipcRenderer.invoke('mesie:embed', recordPath),

  // Knowledge & AI
  knowledgeStats: () => ipcRenderer.invoke('mesie:knowledge-stats'),
  searchResearch: (opts) => ipcRenderer.invoke('mesie:search-research', opts),
  monteCarlo: (trials) => ipcRenderer.invoke('mesie:monte-carlo', trials),

  // Dialogs
  openFile: () => ipcRenderer.invoke('dialog:open-file'),
});
