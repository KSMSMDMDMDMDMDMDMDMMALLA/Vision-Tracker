# Face Tracker Electron

Electron-приложение, которое открывает веб-камеру, находит лицо через MediaPipe Face Detector и плавно ведёт квадрат поверх лица.

## Запуск

Требуется Node.js 20.19+ (рекомендуется актуальный Node.js 22 LTS).

```bash
npm install
npm run dev
```

При первом `npm run dev` скрипт:
- копирует WASM-файлы MediaPipe из `node_modules` в `public/wasm`;
- скачивает официальную модель BlazeFace short-range в `public/models`.

## Проверка production-режима

```bash
npm start
```

Сначала собирается Vite, затем Electron загружает приложение через локальную защищённую схему `faceapp://`.

## by alwaysnear
