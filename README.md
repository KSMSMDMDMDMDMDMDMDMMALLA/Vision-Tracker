# Face Tracker Electron

Утилита для фиксации действия человека, лица, настроения, дистанции.
Может помочь как для встроенного кода для камеры, так и для ознакомления кода как вообще делают AI камеры в городах.

## Запуск

Требуется Node.js 20.19+ (рекомендуется актуальный Node.js 22 LTS).

```bash
npm install
npm run dev
```

При первом `npm run dev` скрипт:
- копирует WASM-файлы MediaPipe из `node_modules` в `public/wasm`;
- скачивает официальную модель BlazeFace short-range в `public/models`.

## Запуск

```bash
npm start
```


## by alwaysnear
