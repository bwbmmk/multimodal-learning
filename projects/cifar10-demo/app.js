const elements = {
  modelStatus: document.getElementById('model-status'),
  modelStatusText: document.getElementById('model-status-text'),
  fileInput: document.getElementById('file-input'),
  fileError: document.getElementById('file-error'),
  dropZone: document.getElementById('drop-zone'),
  previewBlock: document.getElementById('preview-block'),
  originalPreview: document.getElementById('original-preview'),
  tensorPreview: document.getElementById('tensor-preview'),
  fileName: document.getElementById('file-name'),
  sampleGrid: document.getElementById('sample-grid'),
  predictionCard: document.getElementById('prediction-card'),
  predictionPlaceholder: document.getElementById('prediction-placeholder'),
  predictionContent: document.getElementById('prediction-content'),
  predictionLoading: document.getElementById('prediction-loading'),
  predictedClass: document.getElementById('predicted-class'),
  predictedEnglish: document.getElementById('predicted-english'),
  predictedConfidence: document.getElementById('predicted-confidence'),
  sampleTruth: document.getElementById('sample-truth'),
  rankingList: document.getElementById('ranking-list'),
  inferenceTime: document.getElementById('inference-time'),
  phoneAccess: document.getElementById('phone-access'),
  phoneQr: document.getElementById('phone-qr'),
  phoneUrl: document.getElementById('phone-url'),
};

const state = {
  session: null,
  metadata: null,
  currentImage: null,
  currentSample: null,
  objectUrl: null,
  predictionId: 0,
  userInitiated: false,
};

const NARROW_SCREEN = '(max-width: 54rem)';

function setModelStatus(type, text, shortText) {
  const narrow = window.matchMedia(NARROW_SCREEN).matches;
  elements.modelStatus.className = `model-status is-${type}`;
  elements.modelStatusText.textContent = narrow && shortText ? shortText : text;
}

function showFileError(message = '') {
  elements.fileError.textContent = message;
  elements.fileError.hidden = !message;
}

function loadImage(source) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('图片读取失败。如果是 iPhone 的 HEIC 格式，请先导出为 JPG 再试。'));
    image.src = source;
  });
}

function renderSamples() {
  const fragment = document.createDocumentFragment();
  state.metadata.samples.forEach((sample, index) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'sample-button';
    button.dataset.sampleIndex = String(index);
    button.setAttribute('aria-label', `分类示例：${sample.chinese}`);

    const image = document.createElement('img');
    image.src = sample.file;
    image.alt = sample.chinese;
    image.width = 64;
    image.height = 64;

    const label = document.createElement('span');
    label.textContent = sample.chinese;
    button.append(image, label);
    button.addEventListener('click', () => {
      state.userInitiated = true;
      selectSample(index);
    });
    fragment.appendChild(button);
  });
  elements.sampleGrid.replaceChildren(fragment);
}

function markSelectedSample(index) {
  elements.sampleGrid.querySelectorAll('.sample-button').forEach((button) => {
    const selected = Number(button.dataset.sampleIndex) === index;
    button.classList.toggle('is-selected', selected);
    button.setAttribute('aria-pressed', String(selected));
  });
}

async function selectSample(index) {
  const sample = state.metadata.samples[index];
  showFileError();
  try {
    const image = await loadImage(sample.file);
    state.currentSample = sample;
    markSelectedSample(index);
    useImage(image, `${sample.chinese} · CIFAR-10 测试集`);
  } catch (error) {
    showFileError(error.message);
  }
}

function useImage(image, name) {
  state.currentImage = image;
  elements.originalPreview.src = image.src;
  elements.fileName.textContent = name;
  elements.fileName.title = name;
  elements.previewBlock.hidden = false;
  drawModelInput(image);
  runPrediction();
}

function getResizeMode() {
  return document.querySelector('input[name="resize-mode"]:checked').value;
}

function drawModelInput(image) {
  const canvas = elements.tensorPreview;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  const width = image.naturalWidth || image.width;
  const height = image.naturalHeight || image.height;

  context.fillStyle = '#ffffff';
  context.fillRect(0, 0, 32, 32);
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = 'high';

  if (getResizeMode() === 'crop') {
    const side = Math.min(width, height);
    const sourceX = (width - side) / 2;
    const sourceY = (height - side) / 2;
    context.drawImage(image, sourceX, sourceY, side, side, 0, 0, 32, 32);
  } else {
    context.drawImage(image, 0, 0, width, height, 0, 0, 32, 32);
  }
}

function imageToTensor() {
  const context = elements.tensorPreview.getContext('2d', { willReadFrequently: true });
  const pixels = context.getImageData(0, 0, 32, 32).data;
  const data = new Float32Array(3 * 32 * 32);
  const channelSize = 32 * 32;
  // 必须与训练时的 Normalize 一致：CIFAR-10 三通道真实均值/标准差
  const mean = [0.4914, 0.4822, 0.4465];
  const std = [0.2470, 0.2435, 0.2616];

  for (let pixel = 0; pixel < channelSize; pixel += 1) {
    const rgbaIndex = pixel * 4;
    data[pixel] = (pixels[rgbaIndex] / 255 - mean[0]) / std[0];
    data[channelSize + pixel] = (pixels[rgbaIndex + 1] / 255 - mean[1]) / std[1];
    data[channelSize * 2 + pixel] = (pixels[rgbaIndex + 2] / 255 - mean[2]) / std[2];
  }

  return new ort.Tensor('float32', data, [1, 3, 32, 32]);
}

function softmax(logits) {
  const maximum = Math.max(...logits);
  const exponentials = Array.from(logits, (value) => Math.exp(value - maximum));
  const sum = exponentials.reduce((total, value) => total + value, 0);
  return exponentials.map((value) => value / sum);
}

function formatProbability(probability) {
  return `${(probability * 100).toFixed(2)}%`;
}

function setPredictionLoading(loading) {
  elements.predictionCard.classList.remove('is-empty');
  elements.predictionPlaceholder.hidden = true;
  elements.predictionContent.hidden = loading;
  elements.predictionLoading.hidden = !loading;
}

async function runPrediction() {
  if (!state.session || !state.currentImage) return;
  const predictionId = ++state.predictionId;
  setPredictionLoading(true);
  elements.sampleTruth.hidden = true;
  showFileError();

  try {
    const input = imageToTensor();
    const startedAt = performance.now();
    const output = await state.session.run({ images: input });
    const elapsed = performance.now() - startedAt;
    if (predictionId !== state.predictionId) return;

    const probabilities = softmax(output.logits.data);
    const ranking = probabilities
      .map((probability, classIndex) => ({ probability, classIndex }))
      .sort((a, b) => b.probability - a.probability);

    renderPrediction(ranking, elapsed);
  } catch (error) {
    if (predictionId !== state.predictionId) return;
    console.error(error);
    setPredictionLoading(false);
    elements.predictionContent.hidden = true;
    elements.predictionPlaceholder.hidden = false;
    elements.predictionPlaceholder.querySelector('p').textContent = '这次推理没有完成，请换一张图片重试。';
    showFileError(`推理失败：${error.message}`);
  }
}

function renderPrediction(ranking, elapsed) {
  const best = ranking[0];
  const bestClass = state.metadata.samples[best.classIndex];
  elements.predictedClass.textContent = bestClass.chinese;
  elements.predictedEnglish.textContent = bestClass.english;
  elements.predictedConfidence.textContent = formatProbability(best.probability);
  elements.inferenceTime.textContent = `${elapsed.toFixed(0)} ms · [1, 3, 32, 32]`;

  const fragment = document.createDocumentFragment();
  ranking.slice(0, 5).forEach((item, index) => {
    const classInfo = state.metadata.samples[item.classIndex];
    const row = document.createElement('div');
    row.className = 'rank-row';

    const rank = document.createElement('span');
    rank.className = 'rank-index';
    rank.textContent = String(index + 1).padStart(2, '0');

    const name = document.createElement('span');
    name.className = 'rank-name';
    name.textContent = classInfo.chinese;

    const track = document.createElement('div');
    track.className = 'rank-track';
    const bar = document.createElement('div');
    bar.className = 'rank-bar';
    bar.style.width = `${Math.max(item.probability * 100, 0.5)}%`;
    track.appendChild(bar);

    const value = document.createElement('span');
    value.className = 'rank-value';
    value.textContent = formatProbability(item.probability);
    row.append(rank, name, track, value);
    fragment.appendChild(row);
  });
  elements.rankingList.replaceChildren(fragment);

  if (state.currentSample) {
    elements.sampleTruth.textContent = `这张示例的真实类别：${state.currentSample.chinese}`;
    elements.sampleTruth.hidden = false;
  }

  setPredictionLoading(false);

  // 手机上是单列布局，结果卡片在整个输入面板下方，选完图要把它带到视野里
  if (state.userInitiated && window.matchMedia(NARROW_SCREEN).matches) {
    elements.predictionCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }
}

async function handleFile(file) {
  if (!file) return;
  if (!file.type.startsWith('image/')) {
    showFileError('请选择 PNG、JPG、WebP 等图片文件。');
    return;
  }
  if (file.size > 20 * 1024 * 1024) {
    showFileError('图片超过 20 MB，请换一张较小的图片。');
    return;
  }

  showFileError();
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.objectUrl = URL.createObjectURL(file);

  try {
    const image = await loadImage(state.objectUrl);
    state.currentSample = null;
    state.userInitiated = true;
    markSelectedSample(-1);
    useImage(image, file.name);
  } catch (error) {
    showFileError(error.message);
  }
}

elements.fileInput.addEventListener('change', (event) => {
  handleFile(event.target.files[0]);
  event.target.value = '';
});

['dragenter', 'dragover'].forEach((eventName) => {
  elements.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropZone.classList.add('is-dragging');
  });
});

['dragleave', 'drop'].forEach((eventName) => {
  elements.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropZone.classList.remove('is-dragging');
  });
});

elements.dropZone.addEventListener('drop', (event) => {
  handleFile(event.dataTransfer.files[0]);
});

document.querySelectorAll('input[name="resize-mode"]').forEach((input) => {
  input.addEventListener('change', () => {
    if (!state.currentImage) return;
    drawModelInput(state.currentImage);
    runPrediction();
  });
});

async function showPhoneAccess() {
  // 只在电脑上（localhost）显示扫码入口，手机自己打开时没必要再扫自己
  const loopback = ['127.0.0.1', 'localhost', '::1', '[::1]'];
  if (!loopback.includes(location.hostname)) return;

  try {
    const response = await fetch(`assets/local.json?t=${Date.now()}`);
    if (!response.ok) return;
    const data = await response.json();
    if (!data.phoneUrl) return;
    elements.phoneQr.src = `assets/qr.svg?t=${Date.now()}`;
    elements.phoneUrl.textContent = data.phoneUrl;
    elements.phoneAccess.hidden = false;
  } catch (error) {
    // 不是用 serve.py 启动的就没有这两个文件，安静跳过
  }
}

async function init() {
  showPhoneAccess();
  try {
    setModelStatus('loading', '正在加载 ResNet18 模型', '加载模型中');
    const metadataPromise = fetch('assets/samples.json', { cache: 'no-store' }).then((response) => {
      if (!response.ok) throw new Error('示例图片清单读取失败');
      return response.json();
    });

    ort.env.wasm.wasmPaths = new URL('vendor/', document.baseURI).href;
    ort.env.wasm.numThreads = 1;
    const metadata = await metadataPromise;
    const session = await ort.InferenceSession.create(metadata.modelFile || './model.onnx', {
      executionProviders: ['wasm'],
    });
    state.metadata = metadata;
    state.session = session;
    renderSamples();
    setModelStatus('ready', '模型就绪 · 本地 WASM', '模型就绪');
    await selectSample(0);
  } catch (error) {
    console.error(error);
    setModelStatus('error', '模型加载失败');
    elements.predictionPlaceholder.querySelector('p').textContent = '模型没有加载成功。请通过本地服务器打开页面，不要直接双击 HTML 文件。';
    elements.sampleGrid.textContent = '示例加载失败';
  }
}

init();
