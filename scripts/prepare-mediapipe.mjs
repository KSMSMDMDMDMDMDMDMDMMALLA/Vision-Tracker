import fs from 'node:fs/promises';
import path from 'node:path';

const root = process.cwd();
const sourceWasm = path.join(
  root,
  'node_modules',
  '@mediapipe',
  'tasks-vision',
  'wasm'
);
const targetWasm = path.join(root, 'public', 'wasm');
const modelDir = path.join(root, 'public', 'models');

const models = [
  {
    name: 'BlazeFace short-range',
    path: path.join(modelDir, 'blaze_face_short_range.tflite'),
    url: 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite'
  },
  {
    name: 'Hand Landmarker',
    path: path.join(modelDir, 'hand_landmarker.task'),
    url: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task'
  },
  {
    name: 'Face Landmarker',
    path: path.join(modelDir, 'face_landmarker.task'),
    url: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task'
  }
];

await fs.mkdir(targetWasm, { recursive: true });
await fs.mkdir(modelDir, { recursive: true });

try {
  await fs.access(sourceWasm);
} catch {
  throw new Error('MediaPipe не установлен. Сначала выполните: npm install');
}

await fs.cp(sourceWasm, targetWasm, { recursive: true, force: true });

async function ensureModel(model) {
  try {
    await fs.access(model.path);
    return;
  } catch {
  }

  console.log(`[setup] Загружаю ${model.name}...`);

  const response = await fetch(model.url);
  if (!response.ok) {
    throw new Error(
      `Не удалось скачать ${model.name}: ${response.status} ${response.statusText}`
    );
  }

  const buffer = Buffer.from(await response.arrayBuffer());
  await fs.writeFile(model.path, buffer);
  console.log(`[setup] ${model.name} сохранён.`);
}

for (const model of models) {
  await ensureModel(model);
}

console.log('[setup] MediaPipe assets готовы.');
