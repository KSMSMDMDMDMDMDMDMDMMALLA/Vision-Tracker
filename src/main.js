import './style.css';
import {
  FaceDetector,
  FaceLandmarker,
  FilesetResolver,
  HandLandmarker
} from '@mediapipe/tasks-vision';

const video = document.querySelector('#camera');
const canvas = document.querySelector('#overlay');
const cameraWrap = document.querySelector('#cameraWrap');
const ctx = canvas.getContext('2d');

const statusBadge = document.querySelector('#statusBadge');
const statusText = document.querySelector('#statusText');
const faceState = document.querySelector('#faceState');
const handState = document.querySelector('#handState');
const fingerCountText = document.querySelector('#fingerCountText');
const fpsText = document.querySelector('#fpsText');
const gestureState = document.querySelector('#gestureState');
const moodPanel = document.querySelector('#moodPanel');
const moodEmoji = document.querySelector('#moodEmoji');
const moodText = document.querySelector('#moodText');
const moodConfidence = document.querySelector('#moodConfidence');
const smileMetric = document.querySelector('#smileMetric');
const browMetric = document.querySelector('#browMetric');
const frownMetric = document.querySelector('#frownMetric');

// Детекторы работают реже рендера, а графика плавно интерполируется в каждом кадре.
// Это заметно легче для CPU/GPU, когда одновременно включены лицо + руки.
const FACE_DETECTION_HZ = 30;
const HAND_DETECTION_HZ = 30;
const EXPRESSION_DETECTION_HZ = 12;
const FACE_DETECTION_INTERVAL = 1000 / FACE_DETECTION_HZ;
const HAND_DETECTION_INTERVAL = 1000 / HAND_DETECTION_HZ;
const EXPRESSION_DETECTION_INTERVAL = 1000 / EXPRESSION_DETECTION_HZ;

const FACE_SMOOTH_SPEED = 18;
const HAND_SMOOTH_SPEED = 24;
const LOST_FACE_DELAY = 220;
const LOST_HAND_DELAY = 180;
const BOX_MARGIN = 0.12;

// Оценка выражения лица. Это именно выражение, а не достоверное внутреннее настроение.
const EXPRESSION_SMOOTH_ALPHA = 0.28;
const EXPRESSION_LOST_DELAY = 420;
const EXPRESSION_SWITCH_MARGIN = 0.08;
const EXPRESSION_MIN_HOLD = 220;

const MOODS = {
  neutral: { label: 'Нейтральный', emoji: '•', accent: '#d7e0ea' },
  happy: { label: 'Весёлый', emoji: '☺', accent: '#50f0a8' },
  angry: { label: 'Злой', emoji: '⚠', accent: '#ff667a' }
};

// Жест «круг указательным пальцем» → открыть Chrome.
const GESTURE_MIN_POINTS = 22;
const GESTURE_MIN_DURATION = 420;
const GESTURE_MAX_DURATION = 1900;
const GESTURE_POINT_DISTANCE = 0.006;
const GESTURE_MIN_DIAMETER = 0.10;
const GESTURE_MAX_DIAMETER = 0.72;
const GESTURE_LOST_DELAY = 220;
const GESTURE_COOLDOWN = 3200;
const GESTURE_GHOST_TIME = 650;

const FINGERS = [
  { name: 'thumb', tip: 4, dip: 3, pip: 2, mcp: 1, isThumb: true },
  { name: 'index', tip: 8, dip: 7, pip: 6, mcp: 5 },
  { name: 'middle', tip: 12, dip: 11, pip: 10, mcp: 9 },
  { name: 'ring', tip: 16, dip: 15, pip: 14, mcp: 13 },
  { name: 'pinky', tip: 20, dip: 19, pip: 18, mcp: 17 }
];

let faceDetector = null;
let faceLandmarker = null;
let handLandmarker = null;

let targetBox = null;
let currentBox = null;
let lastFaceSeenAt = 0;
let lastHandSeenAt = 0;

let targetHands = [];
let currentHands = [];

let lastFaceDetectionAt = -Infinity;
let lastHandDetectionAt = -Infinity;
let lastExpressionDetectionAt = -Infinity;
let lastVideoTime = -1;
let lastRenderAt = performance.now();
let renderedFrames = 0;
let fpsWindowStart = performance.now();
let stopped = false;

const fingerCountHistory = [];
let stableFingerCount = 0;

let gestureTrail = [];
let gestureGhostTrail = [];
let gestureGhostUntil = 0;
let gestureHandKey = null;
let lastGesturePointAt = 0;
let gestureCooldownUntil = 0;
let gestureFeedbackUntil = 0;
let chromeOpening = false;

let lastExpressionSeenAt = 0;
let expressionSmoothed = {
  smile: 0,
  browDown: 0,
  frown: 0,
  mouthPress: 0,
  eyeSquint: 0
};
let currentMood = 'neutral';
let pendingMood = 'neutral';
let pendingMoodSince = 0;
let moodConfidenceValue = 0;

function setStatus(text, isError = false) {
  statusText.textContent = text;
  statusBadge.classList.toggle('error', isError);
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function distance3D(a, b) {
  if (!a || !b) return 0;
  const dx = (a.x ?? 0) - (b.x ?? 0);
  const dy = (a.y ?? 0) - (b.y ?? 0);
  const dz = (a.z ?? 0) - (b.z ?? 0);
  return Math.hypot(dx, dy, dz);
}

function angle3D(a, b, c) {
  if (!a || !b || !c) return 0;

  const ab = {
    x: a.x - b.x,
    y: a.y - b.y,
    z: (a.z ?? 0) - (b.z ?? 0)
  };
  const cb = {
    x: c.x - b.x,
    y: c.y - b.y,
    z: (c.z ?? 0) - (b.z ?? 0)
  };

  const dot = ab.x * cb.x + ab.y * cb.y + ab.z * cb.z;
  const magA = Math.hypot(ab.x, ab.y, ab.z);
  const magC = Math.hypot(cb.x, cb.y, cb.z);

  if (!magA || !magC) return 0;

  const cos = clamp(dot / (magA * magC), -1, 1);
  return Math.acos(cos) * (180 / Math.PI);
}

function makeSquareBox(box) {
  const side = Math.max(box.width, box.height) * (1 + BOX_MARGIN * 2);
  const centerX = box.originX + box.width / 2;
  const centerY = box.originY + box.height / 2;

  const x = clamp(centerX - side / 2, 0, video.videoWidth);
  const y = clamp(centerY - side / 2, 0, video.videoHeight);
  const maxSide = Math.min(
    side,
    video.videoWidth - x,
    video.videoHeight - y
  );

  return { x, y, w: maxSide, h: maxSide };
}

function pickPrimaryFace(detections) {
  if (!detections?.length) return null;

  return detections.reduce((best, item) => {
    if (!best) return item;

    const bestBox = best.boundingBox;
    const box = item.boundingBox;
    const bestArea = bestBox.width * bestBox.height;
    const area = box.width * box.height;

    return area > bestArea ? item : best;
  }, null);
}

function isFingerExtended(landmarks, finger) {
  const wrist = landmarks[0];
  const tip = landmarks[finger.tip];
  const dip = landmarks[finger.dip];
  const pip = landmarks[finger.pip];
  const mcp = landmarks[finger.mcp];

  if (!wrist || !tip || !dip || !pip || !mcp) return false;

  if (finger.isThumb) {
    // Для большого пальца проверяем не направление по Y, а раскрытие относительно ладони.
    // Поэтому алгоритм работает и при повороте кисти.
    const palmWidth = Math.max(distance3D(landmarks[5], landmarks[17]), 0.001);
    const thumbSpread = distance3D(tip, landmarks[5]) / palmWidth;
    const thumbStraightness = angle3D(pip, dip, tip);
    const tipFromWrist = distance3D(tip, wrist);
    const dipFromWrist = distance3D(dip, wrist);

    return (
      thumbStraightness > 145 &&
      thumbSpread > 0.62 &&
      tipFromWrist > dipFromWrist * 1.03
    );
  }

  const pipAngle = angle3D(mcp, pip, dip);
  const dipAngle = angle3D(pip, dip, tip);
  const tipFromWrist = distance3D(tip, wrist);
  const pipFromWrist = distance3D(pip, wrist);

  return (
    pipAngle > 150 &&
    dipAngle > 145 &&
    tipFromWrist > pipFromWrist * 1.12
  );
}

function getRaisedFingerTips(landmarks) {
  return FINGERS.filter((finger) => isFingerExtended(landmarks, finger)).map(
    (finger) => finger.tip
  );
}

function updateFingerCount(rawCount) {
  fingerCountHistory.push(rawCount);
  if (fingerCountHistory.length > 7) fingerCountHistory.shift();

  const frequencies = new Map();
  for (const value of fingerCountHistory) {
    frequencies.set(value, (frequencies.get(value) || 0) + 1);
  }

  stableFingerCount = [...frequencies.entries()].sort((a, b) => {
    if (b[1] !== a[1]) return b[1] - a[1];
    return b[0] - a[0];
  })[0]?.[0] ?? rawCount;

  fingerCountText.textContent = `Пальцы: ${stableFingerCount}`;
}



function average(values) {
  const valid = values.filter(Number.isFinite);
  if (!valid.length) return 0;
  return valid.reduce((sum, value) => sum + value, 0) / valid.length;
}

function getBlendshapeScore(map, ...names) {
  return average(names.map((name) => map.get(name) ?? 0));
}

function smoothExpressionFeature(key, rawValue) {
  expressionSmoothed[key] +=
    (clamp(rawValue, 0, 1) - expressionSmoothed[key]) * EXPRESSION_SMOOTH_ALPHA;
  return expressionSmoothed[key];
}

function classifyExpression(categories, now) {
  if (!categories?.length) return null;

  const scores = new Map(
    categories.map((category) => [
      category.categoryName || category.displayName,
      Number(category.score) || 0
    ])
  );

  const smile = smoothExpressionFeature(
    'smile',
    getBlendshapeScore(scores, 'mouthSmileLeft', 'mouthSmileRight')
  );
  const cheek = getBlendshapeScore(scores, 'cheekSquintLeft', 'cheekSquintRight');
  const browDown = smoothExpressionFeature(
    'browDown',
    getBlendshapeScore(scores, 'browDownLeft', 'browDownRight')
  );
  const frown = smoothExpressionFeature(
    'frown',
    getBlendshapeScore(scores, 'mouthFrownLeft', 'mouthFrownRight')
  );
  const mouthPress = smoothExpressionFeature(
    'mouthPress',
    getBlendshapeScore(scores, 'mouthPressLeft', 'mouthPressRight')
  );
  const eyeSquint = smoothExpressionFeature(
    'eyeSquint',
    getBlendshapeScore(scores, 'eyeSquintLeft', 'eyeSquintRight')
  );

  const happyRaw = clamp(smile * 0.82 + cheek * 0.18, 0, 1);
  const angryRaw = clamp(
    browDown * 0.52 + frown * 0.23 + mouthPress * 0.15 + eyeSquint * 0.10,
    0,
    1
  );

  const happy = clamp(happyRaw - angryRaw * 0.18, 0, 1);
  const angry = clamp(angryRaw - happyRaw * 0.28, 0, 1);
  const expressionStrength = Math.max(happy, angry);
  const neutral = clamp(1 - expressionStrength * 1.38, 0, 1);

  const candidates = [
    ['happy', happy],
    ['angry', angry],
    ['neutral', neutral]
  ].sort((a, b) => b[1] - a[1]);

  let nextMood = candidates[0][0];
  let nextScore = candidates[0][1];

  if (happy < 0.26 && angry < 0.30) {
    nextMood = 'neutral';
    nextScore = Math.max(neutral, 0.58);
  }

  const currentScore =
    currentMood === 'happy' ? happy :
    currentMood === 'angry' ? angry :
    neutral;

  if (
    nextMood !== currentMood &&
    nextScore < currentScore + EXPRESSION_SWITCH_MARGIN
  ) {
    nextMood = currentMood;
    nextScore = currentScore;
  }

  if (nextMood !== currentMood) {
    if (pendingMood !== nextMood) {
      pendingMood = nextMood;
      pendingMoodSince = now;
    } else if (now - pendingMoodSince >= EXPRESSION_MIN_HOLD) {
      currentMood = nextMood;
      pendingMood = nextMood;
    }
  } else {
    pendingMood = currentMood;
    pendingMoodSince = now;
  }

  const activeScore =
    currentMood === 'happy' ? happy :
    currentMood === 'angry' ? angry :
    neutral;

  moodConfidenceValue = clamp(0.45 + activeScore * 0.55, 0, 0.99);

  return {
    mood: currentMood,
    confidence: moodConfidenceValue,
    smile,
    browDown,
    frown,
    happy,
    angry,
    neutral
  };
}

function updateMoodUi(expression) {
  if (!moodPanel) return;

  if (!expression) {
    moodPanel.dataset.mood = 'unknown';
    moodEmoji.textContent = '—';
    moodText.textContent = 'Лицо не найдено';
    moodConfidence.textContent = '--%';
    smileMetric.textContent = '0%';
    browMetric.textContent = '0%';
    frownMetric.textContent = '0%';
    return;
  }

  const meta = MOODS[expression.mood] || MOODS.neutral;
  moodPanel.dataset.mood = expression.mood;
  moodEmoji.textContent = meta.emoji;
  moodText.textContent = meta.label;
  moodConfidence.textContent = `${Math.round(expression.confidence * 100)}%`;
  smileMetric.textContent = `${Math.round(expression.smile * 100)}%`;
  browMetric.textContent = `${Math.round(expression.browDown * 100)}%`;
  frownMetric.textContent = `${Math.round(expression.frown * 100)}%`;
}

function distance2D(a, b) {
  if (!a || !b) return 0;
  return Math.hypot((a.x ?? 0) - (b.x ?? 0), (a.y ?? 0) - (b.y ?? 0));
}

function setGestureState(text, state = 'idle', holdMs = 0) {
  if (!gestureState) return;
  gestureState.textContent = text;
  gestureState.dataset.state = state;
  gestureFeedbackUntil = holdMs > 0 ? performance.now() + holdMs : 0;
}

function normalizeAngleDelta(value) {
  let angle = value;
  while (angle > Math.PI) angle -= Math.PI * 2;
  while (angle < -Math.PI) angle += Math.PI * 2;
  return angle;
}

function detectCircleGesture(points) {
  if (points.length < GESTURE_MIN_POINTS) return null;

  const first = points[0];
  const last = points[points.length - 1];
  const duration = last.t - first.t;

  if (duration < GESTURE_MIN_DURATION || duration > GESTURE_MAX_DURATION) {
    return null;
  }

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  for (const point of points) {
    minX = Math.min(minX, point.x);
    minY = Math.min(minY, point.y);
    maxX = Math.max(maxX, point.x);
    maxY = Math.max(maxY, point.y);
  }

  const width = maxX - minX;
  const height = maxY - minY;
  const diameter = (width + height) / 2;
  const minSide = Math.min(width, height);
  const maxSide = Math.max(width, height);

  if (
    minSide < GESTURE_MIN_DIAMETER ||
    maxSide > GESTURE_MAX_DIAMETER ||
    minSide / Math.max(maxSide, 0.0001) < 0.62
  ) {
    return null;
  }

  const center = {
    x: (minX + maxX) / 2,
    y: (minY + maxY) / 2
  };

  const radii = points.map((point) => distance2D(point, center));
  const meanRadius = radii.reduce((sum, value) => sum + value, 0) / radii.length;

  if (!meanRadius) return null;

  const radiusVariance = radii.reduce(
    (sum, value) => sum + (value - meanRadius) ** 2,
    0
  ) / radii.length;
  const radiusDeviation = Math.sqrt(radiusVariance) / meanRadius;

  if (radiusDeviation > 0.34) return null;

  const closure = distance2D(first, last) / meanRadius;
  if (closure > 0.9) return null;

  let pathLength = 0;
  let signedSweep = 0;
  let absoluteSweep = 0;
  let previousAngle = Math.atan2(first.y - center.y, first.x - center.x);

  for (let i = 1; i < points.length; i += 1) {
    const current = points[i];
    const previous = points[i - 1];
    pathLength += distance2D(previous, current);

    const angle = Math.atan2(current.y - center.y, current.x - center.x);
    const delta = normalizeAngleDelta(angle - previousAngle);

    // Игнорируем микродрожание пальца около одной точки.
    if (Math.abs(delta) > 0.01) {
      signedSweep += delta;
      absoluteSweep += Math.abs(delta);
    }

    previousAngle = angle;
  }

  const circumference = Math.PI * Math.max(diameter, 0.0001);
  const lengthRatio = pathLength / circumference;
  const directionConsistency = Math.abs(signedSweep) / Math.max(absoluteSweep, 0.0001);

  if (lengthRatio < 0.68 || lengthRatio > 1.58) return null;
  if (Math.abs(signedSweep) < Math.PI * 1.55) return null;
  if (directionConsistency < 0.68) return null;

  return {
    center,
    radius: meanRadius,
    score: clamp(
      1 - radiusDeviation * 1.5 - Math.abs(1 - lengthRatio) * 0.35,
      0,
      1
    )
  };
}

function getGesturePointerHand() {
  return currentHands.find((hand) => {
    const raised = hand.raisedTips || [];
    const indexUp = raised.includes(8);
    const otherLongFingerUp = [12, 16, 20].some((tip) => raised.includes(tip));

    // Большой палец разрешаем держать как удобно, но средний/безымянный/мизинец
    // должны быть сложены — так круг почти невозможно запустить случайно.
    return indexUp && !otherLongFingerUp;
  });
}

async function triggerCircleGesture(now) {
  if (chromeOpening || now < gestureCooldownUntil) return;

  chromeOpening = true;
  gestureCooldownUntil = now + GESTURE_COOLDOWN;
  gestureGhostTrail = gestureTrail.map((point) => ({ ...point }));
  gestureGhostUntil = now + GESTURE_GHOST_TIME;
  gestureTrail = [];
  gestureHandKey = null;

  setGestureState('КРУГ РАСПОЗНАН → CHROME', 'success', 1800);

  try {
    if (!window.desktopActions?.openChrome) {
      throw new Error('Electron API недоступен');
    }

    const result = await window.desktopActions.openChrome();
    if (!result?.ok) {
      throw new Error(result?.error || 'Не удалось открыть Chrome');
    }

    if (result.browser === 'default') {
      setGestureState('Круг распознан → открыт браузер по умолчанию', 'success', 1800);
    }
  } catch (error) {
    console.error('Circle gesture action error:', error);
    setGestureState('Жест распознан, но браузер не открылся', 'error', 2200);
  } finally {
    chromeOpening = false;
  }
}

function updateGestureTracking(now) {
  if (gestureFeedbackUntil && now > gestureFeedbackUntil) {
    gestureFeedbackUntil = 0;
    setGestureState('');
  }

  if (now < gestureCooldownUntil) return;

  const hand = getGesturePointerHand();
  const point = hand?.landmarks?.[8];

  if (!hand || !point) {
    if (now - lastGesturePointAt > GESTURE_LOST_DELAY) {
      gestureTrail = [];
      gestureHandKey = null;
      if (!gestureFeedbackUntil) {
        setGestureState('');
      }
    }
    return;
  }

  if (gestureHandKey && gestureHandKey !== hand.key) {
    gestureTrail = [];
  }
  gestureHandKey = hand.key;
  lastGesturePointAt = now;

  const nextPoint = {
    x: clamp(point.x, 0, 1),
    y: clamp(point.y, 0, 1),
    t: now
  };

  const previous = gestureTrail[gestureTrail.length - 1];

  // Сильный скачок означает, что трекинг перескочил на другую руку/точку.
  if (previous && distance2D(previous, nextPoint) > 0.18) {
    gestureTrail = [];
  }

  const last = gestureTrail[gestureTrail.length - 1];
  if (!last || distance2D(last, nextPoint) >= GESTURE_POINT_DISTANCE) {
    gestureTrail.push(nextPoint);
  }

  while (
    gestureTrail.length > 1 &&
    now - gestureTrail[0].t > GESTURE_MAX_DURATION
  ) {
    gestureTrail.shift();
  }

  if (!gestureFeedbackUntil && gestureTrail.length > 2) {
    setGestureState('Указательный палец отслеживается · рисуй круг', 'tracking');
  }

  const circle = detectCircleGesture(gestureTrail);
  if (circle) {
    void triggerCircleGesture(now);
  }
}

function buildHandTargets(result) {
  const landmarksList = result?.landmarks || [];
  const worldList = result?.worldLandmarks || [];
  const handednessList = result?.handedness || result?.handednesses || [];

  return landmarksList.map((landmarks, index) => {
    const handedness = handednessList[index]?.[0]?.categoryName || `Hand${index}`;

    return {
      key: `${handedness}-${index}`,
      handedness,
      landmarks: landmarks.map((point) => ({ ...point })),
      worldLandmarks: (worldList[index] || []).map((point) => ({ ...point })),
      raisedTips: getRaisedFingerTips(landmarks)
    };
  });
}

function detectFace(timestamp) {
  if (!faceDetector) return;

  try {
    const result = faceDetector.detectForVideo(video, timestamp);
    const face = pickPrimaryFace(result.detections);

    if (face?.boundingBox) {
      targetBox = makeSquareBox(face.boundingBox);
      lastFaceSeenAt = performance.now();

      const score = face.categories?.[0]?.score;
      faceState.textContent = Number.isFinite(score)
        ? `Лицо найдено · ${Math.round(score * 100)}%`
        : 'Лицо найдено';
    } else if (performance.now() - lastFaceSeenAt > LOST_FACE_DELAY) {
      targetBox = null;
      faceState.textContent = 'Лицо не найдено';
    }
  } catch (error) {
    console.error('Face detection error:', error);
  }
}


function detectExpression(timestamp) {
  if (!faceLandmarker) return;

  try {
    const result = faceLandmarker.detectForVideo(video, timestamp);
    const categories = result?.faceBlendshapes?.[0]?.categories || [];

    if (categories.length) {
      lastExpressionSeenAt = performance.now();
      const expression = classifyExpression(categories, performance.now());
      updateMoodUi(expression);
    } else if (performance.now() - lastExpressionSeenAt > EXPRESSION_LOST_DELAY) {
      expressionSmoothed = {
        smile: 0,
        browDown: 0,
        frown: 0,
        mouthPress: 0,
        eyeSquint: 0
      };
      currentMood = 'neutral';
      pendingMood = 'neutral';
      moodConfidenceValue = 0;
      updateMoodUi(null);
    }
  } catch (error) {
    console.error('Expression detection error:', error);
  }
}

function detectHands(timestamp) {
  if (!handLandmarker) return;

  try {
    const result = handLandmarker.detectForVideo(video, timestamp);
    const hands = buildHandTargets(result);

    if (hands.length) {
      targetHands = hands;
      lastHandSeenAt = performance.now();

      const rawCount = hands.reduce(
        (sum, hand) => sum + hand.raisedTips.length,
        0
      );
      updateFingerCount(rawCount);

      handState.textContent = `Рук: ${hands.length} · Пальцев: ${stableFingerCount}`;
    } else if (performance.now() - lastHandSeenAt > LOST_HAND_DELAY) {
      targetHands = [];
      fingerCountHistory.length = 0;
      stableFingerCount = 0;
      fingerCountText.textContent = 'Пальцы: 0';
      handState.textContent = 'Рука не найдена';
    }
  } catch (error) {
    console.error('Hand detection error:', error);
  }
}

function detectFrame(timestamp) {
  if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;

  if (timestamp - lastFaceDetectionAt >= FACE_DETECTION_INTERVAL) {
    detectFace(timestamp);
    lastFaceDetectionAt = timestamp;
  }

  if (timestamp - lastHandDetectionAt >= HAND_DETECTION_INTERVAL) {
    detectHands(timestamp);
    lastHandDetectionAt = timestamp;
  }

  if (timestamp - lastExpressionDetectionAt >= EXPRESSION_DETECTION_INTERVAL) {
    detectExpression(timestamp);
    lastExpressionDetectionAt = timestamp;
  }
}

function videoFrameLoop(now) {
  if (stopped) return;
  detectFrame(now);
  video.requestVideoFrameCallback(videoFrameLoop);
}

function fallbackDetectionLoop(now) {
  if (stopped) return;

  if (video.currentTime !== lastVideoTime) {
    detectFrame(now);
    lastVideoTime = video.currentTime;
  }

  requestAnimationFrame(fallbackDetectionLoop);
}

function syncCanvasSize() {
  const rect = cameraWrap.getBoundingClientRect();
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const width = Math.max(1, Math.round(rect.width * dpr));
  const height = Math.max(1, Math.round(rect.height * dpr));

  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return rect;
}

function getVideoTransform(screenW, screenH) {
  const videoW = video.videoWidth;
  const videoH = video.videoHeight;

  if (!videoW || !videoH) return null;

  const scale = Math.max(screenW / videoW, screenH / videoH);
  const renderedW = videoW * scale;
  const renderedH = videoH * scale;

  return {
    scale,
    offsetX: (screenW - renderedW) / 2,
    offsetY: (screenH - renderedH) / 2,
    videoW,
    videoH
  };
}

function mapVideoBoxToScreen(box, screenW, screenH) {
  const transform = getVideoTransform(screenW, screenH);
  if (!transform) return null;

  const { scale, offsetX, offsetY } = transform;
  const width = box.w * scale;
  const height = box.h * scale;
  const unmirroredX = offsetX + box.x * scale;

  return {
    x: screenW - (unmirroredX + width),
    y: offsetY + box.y * scale,
    w: width,
    h: height
  };
}

function mapLandmarkToScreen(point, screenW, screenH) {
  const transform = getVideoTransform(screenW, screenH);
  if (!transform || !point) return null;

  const unmirroredX =
    transform.offsetX + point.x * transform.videoW * transform.scale;
  const y =
    transform.offsetY + point.y * transform.videoH * transform.scale;

  return {
    x: screenW - unmirroredX,
    y
  };
}

function lerpBox(current, target, alpha) {
  if (!current) return { ...target };

  return {
    x: current.x + (target.x - current.x) * alpha,
    y: current.y + (target.y - current.y) * alpha,
    w: current.w + (target.w - current.w) * alpha,
    h: current.h + (target.h - current.h) * alpha
  };
}

function lerpPoint(current, target, alpha) {
  if (!current) return { ...target };
  return {
    x: current.x + (target.x - current.x) * alpha,
    y: current.y + (target.y - current.y) * alpha,
    z: (current.z ?? 0) + ((target.z ?? 0) - (current.z ?? 0)) * alpha,
    visibility: target.visibility
  };
}

function lerpHands(current, target, alpha) {
  return target.map((targetHand, index) => {
    const sameKey = current.find((hand) => hand.key === targetHand.key);
    const fallback = current[index];
    const previous = sameKey || fallback;

    return {
      ...targetHand,
      landmarks: targetHand.landmarks.map((point, pointIndex) =>
        lerpPoint(previous?.landmarks?.[pointIndex], point, alpha)
      ),
      worldLandmarks: targetHand.worldLandmarks.map((point, pointIndex) =>
        lerpPoint(previous?.worldLandmarks?.[pointIndex], point, alpha)
      )
    };
  });
}

function drawTrackingBox(box, screenW, screenH) {
  if (!box) return;

  const mapped = mapVideoBoxToScreen(box, screenW, screenH);
  if (!mapped) return;

  const x = mapped.x;
  const y = mapped.y;
  const w = mapped.w;
  const h = mapped.h;
  const corner = Math.min(28, Math.max(14, w * 0.16));

  ctx.save();
  ctx.lineWidth = 2;
  ctx.strokeStyle = 'rgba(0, 255, 150, 0.28)';
  ctx.strokeRect(x, y, w, h);

  ctx.lineWidth = 4;
  ctx.lineCap = 'round';
  ctx.strokeStyle = '#00ff96';
  ctx.shadowColor = 'rgba(0, 255, 150, 0.6)';
  ctx.shadowBlur = 12;

  ctx.beginPath();

  ctx.moveTo(x, y + corner);
  ctx.lineTo(x, y);
  ctx.lineTo(x + corner, y);

  ctx.moveTo(x + w - corner, y);
  ctx.lineTo(x + w, y);
  ctx.lineTo(x + w, y + corner);

  ctx.moveTo(x + w, y + h - corner);
  ctx.lineTo(x + w, y + h);
  ctx.lineTo(x + w - corner, y + h);

  ctx.moveTo(x + corner, y + h);
  ctx.lineTo(x, y + h);
  ctx.lineTo(x, y + h - corner);

  ctx.stroke();
  ctx.restore();
}

function getFingerDistanceLabel(hand, tipA, tipB, pointA, pointB) {
  const worldA = hand.worldLandmarks?.[tipA];
  const worldB = hand.worldLandmarks?.[tipB];

  if (worldA && worldB) {
    const cm = distance3D(worldA, worldB) * 100;
    if (Number.isFinite(cm) && cm > 0.05 && cm < 50) {
      return `~${cm.toFixed(1)} см`;
    }
  }

  const px = Math.hypot(pointB.x - pointA.x, pointB.y - pointA.y);
  return `${Math.round(px)} px`;
}

function drawDistanceLabel(text, x, y) {
  ctx.save();
  ctx.font = '600 11px Inter, system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';

  const paddingX = 7;
  const width = ctx.measureText(text).width + paddingX * 2;
  const height = 20;

  ctx.fillStyle = 'rgba(4, 12, 18, 0.78)';
  ctx.strokeStyle = 'rgba(80, 205, 255, 0.55)';
  ctx.lineWidth = 1;

  ctx.beginPath();
  ctx.roundRect(x - width / 2, y - height / 2, width, height, 6);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = '#dff7ff';
  ctx.fillText(text, x, y + 0.5);
  ctx.restore();
}

function drawHandTracking(hand, screenW, screenH) {
  if (!hand?.landmarks?.length) return;

  const raisedTips = hand.raisedTips || [];
  const mappedTips = raisedTips
    .map((tipIndex) => ({
      tipIndex,
      point: mapLandmarkToScreen(hand.landmarks[tipIndex], screenW, screenH)
    }))
    .filter((item) => item.point);

  if (!mappedTips.length) return;

  ctx.save();

  // Тонкие линии между поднятыми пальцами.
  if (mappedTips.length >= 2) {
    ctx.lineWidth = 1.25;
    ctx.strokeStyle = 'rgba(86, 210, 255, 0.92)';
    ctx.shadowColor = 'rgba(86, 210, 255, 0.35)';
    ctx.shadowBlur = 7;

    for (let i = 0; i < mappedTips.length - 1; i += 1) {
      const a = mappedTips[i];
      const b = mappedTips[i + 1];

      ctx.beginPath();
      ctx.moveTo(a.point.x, a.point.y);
      ctx.lineTo(b.point.x, b.point.y);
      ctx.stroke();

      const label = getFingerDistanceLabel(
        hand,
        a.tipIndex,
        b.tipIndex,
        a.point,
        b.point
      );

      drawDistanceLabel(
        label,
        (a.point.x + b.point.x) / 2,
        (a.point.y + b.point.y) / 2 - 14
      );
    }
  }

  // Маленькие точки только на кончиках поднятых пальцев.
  for (const { point } of mappedTips) {
    ctx.beginPath();
    ctx.arc(point.x, point.y, 4, 0, Math.PI * 2);
    ctx.fillStyle = '#62d8ff';
    ctx.shadowColor = 'rgba(98, 216, 255, 0.9)';
    ctx.shadowBlur = 10;
    ctx.fill();

    ctx.beginPath();
    ctx.arc(point.x, point.y, 8, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(98, 216, 255, 0.38)';
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  ctx.restore();
}


function drawGestureTrail(points, screenW, screenH, recognized = false) {
  if (!points?.length) return;

  const mapped = points
    .map((point) => mapLandmarkToScreen(point, screenW, screenH))
    .filter(Boolean);

  if (mapped.length < 2) return;

  ctx.save();
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.lineWidth = recognized ? 3 : 2;
  ctx.strokeStyle = recognized
    ? 'rgba(255, 76, 104, 0.98)'
    : 'rgba(255, 92, 118, 0.78)';
  ctx.shadowColor = 'rgba(255, 76, 104, 0.55)';
  ctx.shadowBlur = recognized ? 14 : 8;

  ctx.beginPath();
  ctx.moveTo(mapped[0].x, mapped[0].y);
  for (let i = 1; i < mapped.length; i += 1) {
    ctx.lineTo(mapped[i].x, mapped[i].y);
  }
  ctx.stroke();

  const tip = mapped[mapped.length - 1];
  ctx.beginPath();
  ctx.arc(tip.x, tip.y, recognized ? 7 : 5, 0, Math.PI * 2);
  ctx.strokeStyle = 'rgba(255, 226, 232, 0.95)';
  ctx.lineWidth = 1.5;
  ctx.stroke();

  ctx.restore();
}


function drawMoodTag(box, screenW, screenH) {
  if (!box || performance.now() - lastExpressionSeenAt > EXPRESSION_LOST_DELAY) return;

  const mapped = mapVideoBoxToScreen(box, screenW, screenH);
  if (!mapped) return;

  const meta = MOODS[currentMood] || MOODS.neutral;
  const title = `${meta.label.toUpperCase()} · ${Math.round(moodConfidenceValue * 100)}%`;
  const details = `улыбка ${Math.round(expressionSmoothed.smile * 100)}%  ·  брови ${Math.round(expressionSmoothed.browDown * 100)}%`;

  ctx.save();
  ctx.font = '800 12px Inter, system-ui, sans-serif';
  const titleWidth = ctx.measureText(title).width;
  ctx.font = '600 10px Inter, system-ui, sans-serif';
  const detailsWidth = ctx.measureText(details).width;

  const width = Math.max(150, titleWidth + 24, detailsWidth + 24);
  const height = 48;
  let x = mapped.x;
  let y = mapped.y + mapped.h + 8;

  if (y + height > screenH - 6) {
    y = mapped.y - height - 8;
  }
  x = clamp(x, 6, Math.max(6, screenW - width - 6));

  ctx.fillStyle = 'rgba(5, 10, 15, 0.86)';
  ctx.strokeStyle = meta.accent;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(x, y, width, height, 9);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = meta.accent;
  ctx.font = '800 12px Inter, system-ui, sans-serif';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  ctx.fillText(title, x + 12, y + 16);

  ctx.fillStyle = 'rgba(225, 233, 243, 0.76)';
  ctx.font = '600 10px Inter, system-ui, sans-serif';
  ctx.fillText(details, x + 12, y + 33);
  ctx.restore();
}

function drawOverlay() {
  const rect = syncCanvasSize();
  ctx.clearRect(0, 0, rect.width, rect.height);

  drawTrackingBox(currentBox, rect.width, rect.height);
  drawMoodTag(currentBox, rect.width, rect.height);

  for (const hand of currentHands) {
    drawHandTracking(hand, rect.width, rect.height);
  }

  drawGestureTrail(gestureTrail, rect.width, rect.height);

  if (performance.now() < gestureGhostUntil) {
    drawGestureTrail(gestureGhostTrail, rect.width, rect.height, true);
  }
}

function updateFps(now) {
  renderedFrames += 1;
  const elapsed = now - fpsWindowStart;

  if (elapsed >= 500) {
    const fps = Math.round((renderedFrames * 1000) / elapsed);
    fpsText.textContent = `${fps} FPS`;
    renderedFrames = 0;
    fpsWindowStart = now;
  }
}

function renderLoop(now) {
  if (stopped) return;

  const dt = Math.min((now - lastRenderAt) / 1000, 0.05);
  lastRenderAt = now;

  if (targetBox) {
    const alpha = 1 - Math.exp(-FACE_SMOOTH_SPEED * dt);
    currentBox = lerpBox(currentBox, targetBox, alpha);
  } else if (performance.now() - lastFaceSeenAt > LOST_FACE_DELAY) {
    currentBox = null;
  }

  if (targetHands.length) {
    const alpha = 1 - Math.exp(-HAND_SMOOTH_SPEED * dt);
    currentHands = lerpHands(currentHands, targetHands, alpha);
  } else if (performance.now() - lastHandSeenAt > LOST_HAND_DELAY) {
    currentHands = [];
  }

  updateGestureTracking(now);
  drawOverlay();
  updateFps(now);
  requestAnimationFrame(renderLoop);
}

async function createWithGpuFallback(createGpu, createCpu, label) {
  try {
    return await createGpu();
  } catch (gpuError) {
    console.warn(`${label}: GPU delegate недоступен, использую CPU.`, gpuError);
    return createCpu();
  }
}

async function createTrackers() {
  const wasmRoot = new URL('./wasm/', window.location.href).toString();
  const faceModelPath = new URL(
    './models/blaze_face_short_range.tflite',
    window.location.href
  ).toString();
  const handModelPath = new URL(
    './models/hand_landmarker.task',
    window.location.href
  ).toString();
  const faceLandmarkerModelPath = new URL(
    './models/face_landmarker.task',
    window.location.href
  ).toString();

  const vision = await FilesetResolver.forVisionTasks(wasmRoot);

  const faceCommon = {
    baseOptions: { modelAssetPath: faceModelPath },
    runningMode: 'VIDEO',
    minDetectionConfidence: 0.55,
    minSuppressionThreshold: 0.3
  };

  const faceLandmarkerCommon = {
    baseOptions: { modelAssetPath: faceLandmarkerModelPath },
    runningMode: 'VIDEO',
    numFaces: 1,
    minFaceDetectionConfidence: 0.5,
    minFacePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: false
  };

  const handCommon = {
    baseOptions: { modelAssetPath: handModelPath },
    runningMode: 'VIDEO',
    numHands: 2,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5
  };

  faceDetector = await createWithGpuFallback(
    () =>
      FaceDetector.createFromOptions(vision, {
        ...faceCommon,
        baseOptions: { ...faceCommon.baseOptions, delegate: 'GPU' }
      }),
    () => FaceDetector.createFromOptions(vision, faceCommon),
    'FaceDetector'
  );

  faceLandmarker = await createWithGpuFallback(
    () =>
      FaceLandmarker.createFromOptions(vision, {
        ...faceLandmarkerCommon,
        baseOptions: {
          ...faceLandmarkerCommon.baseOptions,
          delegate: 'GPU'
        }
      }),
    () => FaceLandmarker.createFromOptions(vision, faceLandmarkerCommon),
    'FaceLandmarker'
  );

  handLandmarker = await createWithGpuFallback(
    () =>
      HandLandmarker.createFromOptions(vision, {
        ...handCommon,
        baseOptions: { ...handCommon.baseOptions, delegate: 'GPU' }
      }),
    () => HandLandmarker.createFromOptions(vision, handCommon),
    'HandLandmarker'
  );
}

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 1280 },
      height: { ideal: 720 },
      frameRate: { ideal: 60, max: 60 },
      facingMode: 'user'
    },
    audio: false
  });

  video.srcObject = stream;

  await new Promise((resolve) => {
    if (video.readyState >= HTMLMediaElement.HAVE_METADATA) {
      resolve();
      return;
    }

    video.addEventListener('loadedmetadata', resolve, { once: true });
  });

  await video.play();
}

async function init() {
  try {
    setStatus('Загрузка моделей...');
    await createTrackers();

    setStatus('Запрашиваю камеру...');
    await startCamera();

    setStatus('Отслеживание активно');
    setGestureState('');

    requestAnimationFrame(renderLoop);

    if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
      video.requestVideoFrameCallback(videoFrameLoop);
    } else {
      requestAnimationFrame(fallbackDetectionLoop);
    }
  } catch (error) {
    console.error(error);
    setStatus('Ошибка запуска', true);
    faceState.textContent = error?.message || 'Не удалось запустить камеру';
    handState.textContent = 'Трекер рук недоступен';
    updateMoodUi(null);
  }
}

window.addEventListener('beforeunload', () => {
  stopped = true;

  const stream = video.srcObject;
  if (stream) {
    for (const track of stream.getTracks()) track.stop();
  }

  faceDetector?.close?.();
  faceLandmarker?.close?.();
  handLandmarker?.close?.();
});

init();
