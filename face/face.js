// 顔の形スキャン（イケオジ計画）
// 顔の読み取りはすべてブラウザ内で行い、画像・数値はどこにも送信しない。

const MP_VERSION = '1.0.1';
const MP_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`;
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const DEBUG = new URLSearchParams(location.search).has('debug');
const SAMPLES_NEEDED = 30;

// ---- 測る場所（MediaPipeの478点のうち、輪郭の左右の点） ----
const P = {
  top: 10, chin: 152,          // ひたいの上（生え際より少し下）・あご先
  cheekL: 234, cheekR: 454,    // ほお骨のいちばん外側
  jawL: 172, jawR: 397,        // エラのあたり
  foreL: 54, foreR: 284,       // ひたいの左右
  chinL: 150, chinR: 379,      // あご先の少し外側（あごの細さ）
  eyeL: 33, eyeR: 263, nose: 1,
};
const FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
  152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10];

// 平均的な顔（MediaPipeの標準の顔モデル canonical_face_model を正面から見た値）
// L=縦の長さ/ほおの幅, J=エラの幅/ほおの幅, F=ひたいの幅/ほおの幅, C=あご先の幅/ほおの幅
const AVG = { L: 1.15, J: 0.78, F: 0.82, C: 0.53 };
const PITCH_BASE = 0.314;   // (鼻先-目)/(あご-目) の平均。顔の上下の傾きの目安

// 顔の形ごとの「典型的な比率」。平均からのずれで判定する
const PROTO = {
  oval:   { L: 1.15, J: 0.77, F: 0.82, C: 0.52 },
  round:  { L: 1.05, J: 0.81, F: 0.83, C: 0.57 },
  long:   { L: 1.27, J: 0.77, F: 0.81, C: 0.52 },
  base:   { L: 1.10, J: 0.86, F: 0.82, C: 0.58 },
  invtri: { L: 1.15, J: 0.71, F: 0.86, C: 0.46 },
};
const SD = { L: 0.05, J: 0.035, F: 0.035, C: 0.04 };

// ---- 顔の形ごとの髪型の目安 ----
const TYPES = {
  oval: {
    name: '卵型',
    feature: '縦と横のバランスが平均に近く、あごのラインがなめらかな形です。',
    good: ['多くの髪型が似合いやすいので、仕事や好みで選んで大丈夫です', '短髪で前髪を上げる（アップバング）と、清潔感が出やすい', 'ツーブロックなど、サイドをすっきりさせた髪型もなじみやすい'],
    avoid: ['大きく外すものは少ない形です。長さより、寝ぐせやパサつきなど手入れのほうが印象を左右します'],
    say: '「全体を短めにして、前髪は上げても下ろしてもいい長さにしてください」',
  },
  round: {
    name: '丸顔',
    feature: '縦の長さが短めで、ほおからあごにかけて丸みがある形です。',
    good: ['トップ（頭の上）に少し高さを出して、縦のラインをつくる', 'サイド（耳の上）は短く、すっきりさせる', '前髪は上げるか斜めに流して、ひたいを少し見せる'],
    avoid: ['サイドにボリュームがある丸いシルエット（顔の丸さが強調されやすい）', '重い前髪をまっすぐ下ろす'],
    say: '「トップは少し長めに残して、サイドは短くしてください。前髪は上げられる長さで」',
  },
  long: {
    name: '面長',
    feature: '横幅に対して、縦の長さが長めの形です。',
    good: ['トップの高さは抑えめにする（立てすぎない）', 'サイドは刈り上げすぎず、少し厚みを残す', '前髪を下ろすか軽く流して、ひたいを少し隠すと縦の長さがやわらぐ'],
    avoid: ['トップを高く立てる（顔がさらに長く見えやすい）', 'サイドを極端に短く刈り上げる'],
    say: '「トップは高くせず抑えめで、サイドは刈り上げすぎずに少し残してください。前髪は下ろせる長さで」',
  },
  base: {
    name: 'ベース型',
    feature: 'エラ（耳の下のあごの角）に幅があり、あごのラインが直線的な形です。',
    good: ['トップに少し高さを出して、縦のラインをつくる', 'サイドはすっきり短めにする', '毛先に少し動きをつけると、直線的な印象がやわらぐ'],
    avoid: ['あごのあたりで髪を切りそろえる（エラの横幅が目立ちやすい）', '全体を角ばったシルエットにする'],
    say: '「トップに少し高さが出るようにして、サイドはすっきりさせてください。毛先は少し動きが出るように」',
  },
  invtri: {
    name: '逆三角',
    feature: 'ひたいが広めで、あご先に向かって細くなる形です。',
    good: ['トップのボリュームは抑えめにする', '前髪を少し下ろすか流して、ひたいの広さをやわらげる', 'サイドはすっきりさせ、毛先の動きで質感を出す'],
    avoid: ['トップを大きくふくらませる（上が重く、あごがより細く見えやすい）', '前髪を全部上げて、ひたいを大きく出す'],
    say: '「トップは抑えめにして、前髪は少し下ろせる長さにしてください。サイドはすっきりで」',
  },
};
const ORDER = ['oval', 'round', 'long', 'base', 'invtri'];

// 公開日が来た記事だけ表示する
const LINKS = [
  { title: '美容室と床屋、40代・50代はどっちに行くべきか｜違いと選び方、「髪型の伝え方」まで', url: 'https://www.ikeojikeikaku.com/entry/barber-or-salon', from: '2026-10-07T19:00:00+09:00' },
  { title: '整髪料の選び方｜40代・50代の「ベタつく・崩れる・ぺたんこになる」を防ぐ、乾かし方と価格帯別3本', url: 'https://www.ikeojikeikaku.com/entry/hair-styling-products', from: '2026-10-05T07:00:00+09:00' },
];

// ---- 画面 ----
const $ = (id) => document.getElementById(id);
function show(id) {
  document.querySelectorAll('.screen').forEach((s) => { s.hidden = s.id !== id; });
  window.scrollTo(0, 0);
}
function introMessage(text) {
  const m = $('intro-msg');
  m.textContent = text;
  m.hidden = !text;
  if (text) m.scrollIntoView({ block: 'center' });
}

// ---- MediaPipe ----
let landmarker = null;
let currentMode = null;
async function getLandmarker(mode) {
  if (!landmarker) {
    const { FaceLandmarker, FilesetResolver } = await import(`${MP_BASE}/vision_bundle.mjs`);
    const fileset = await FilesetResolver.forVisionTasks(`${MP_BASE}/wasm`);
    const opts = {
      baseOptions: { modelAssetPath: MODEL_URL },
      runningMode: mode,
      numFaces: 1,
      outputFaceBlendshapes: true,
    };
    try {
      landmarker = await FaceLandmarker.createFromOptions(fileset, { ...opts, baseOptions: { ...opts.baseOptions, delegate: 'GPU' } });
    } catch {
      landmarker = await FaceLandmarker.createFromOptions(fileset, opts);
    }
    currentMode = mode;
  } else if (currentMode !== mode) {
    await landmarker.setOptions({ runningMode: mode });
    currentMode = mode;
  }
  return landmarker;
}

// ---- 計測 ----
function measure(lm, W, H) {
  const pt = (i) => [lm[i].x * W, lm[i].y * H];
  const d = (a, b) => { const p = pt(a), q = pt(b); return Math.hypot(p[0] - q[0], p[1] - q[1]); };
  const cheek = d(P.cheekL, P.cheekR);
  const [ex1, ey1] = pt(P.eyeL), [ex2, ey2] = pt(P.eyeR);
  const eyeY = (ey1 + ey2) / 2;
  const [nx] = pt(P.nose), [cx1] = pt(P.cheekL), [cx2] = pt(P.cheekR);
  return {
    L: d(P.top, P.chin) / cheek,
    J: d(P.jawL, P.jawR) / cheek,
    F: d(P.foreL, P.foreR) / cheek,
    C: d(P.chinL, P.chinR) / cheek,
    size: cheek / W,
    roll: Math.atan2(ey2 - ey1, ex2 - ex1) * 180 / Math.PI,
    yaw: (nx - Math.min(cx1, cx2)) / Math.abs(cx2 - cx1) - 0.5,
    pitch: (pt(P.nose)[1] - eyeY) / (pt(P.chin)[1] - eyeY) - PITCH_BASE,
  };
}

function jawOpen(result) {
  const cats = result.faceBlendshapes?.[0]?.categories || [];
  return cats.find((c) => c.categoryName === 'jawOpen')?.score ?? 0;
}

// 撮り方のチェック。問題があれば言葉で返す
function checkPose(m, open, sizeRange = [0.28, 0.8]) {
  if (m.size < sizeRange[0]) return 'もう少しスマホを顔に近づけてください';
  if (m.size > sizeRange[1]) return 'もう少しスマホを顔から離してください';
  if (Math.abs(m.roll) > 6) return '顔が傾いています。まっすぐにしてください';
  if (Math.abs(m.yaw) > 0.07) return '顔が横を向いています。正面を向いてください';
  if (m.pitch > 0.08) return 'あごが上がっています。少しあごを引いてください';
  if (m.pitch < -0.08) return 'うつむいています。少し顔を上げてください';
  if (open > 0.2) return '口を閉じて、真顔にしてください';
  return null;
}

function median(arr) {
  const s = [...arr].sort((a, b) => a - b);
  const n = s.length;
  return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}

function classify(f) {
  const scores = {};
  let total = 0;
  for (const k of ORDER) {
    let d2 = 0;
    for (const key of ['L', 'J', 'F', 'C']) d2 += ((f[key] - PROTO[k][key]) / SD[key]) ** 2;
    scores[k] = Math.exp(-d2 / 2);
    total += scores[k];
  }
  const ranked = ORDER.map((k) => ({ k, p: scores[k] / (total || 1) })).sort((a, b) => b.p - a.p);
  return ranked;
}

function level(value, avg, tol, words) {
  const r = (value - avg) / avg;
  if (r < -tol) return words[0];
  if (r > tol) return words[2];
  return words[1];
}

// ---- カメラでスキャン ----
let stream = null;
let rafId = 0;
function stopCamera() {
  cancelAnimationFrame(rafId);
  rafId = 0;
  if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
  const v = $('video');
  v.srcObject = null;
}

async function startCamera() {
  introMessage('');
  if (!navigator.mediaDevices?.getUserMedia) {
    introMessage('このブラウザではカメラを使えません。「写真を選んで調べる」か「質問で調べる」をお使いください。');
    return;
  }
  show('s-scan');
  $('scan-status').textContent = 'カメラを起動しています…';
  $('scan-bar').style.width = '0%';
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 720 }, height: { ideal: 960 } },
      audio: false,
    });
  } catch (e) {
    show('s-intro');
    introMessage('カメラを使えませんでした。ブラウザの設定でカメラの使用を許可するか、「写真を選んで調べる」「質問で調べる」をお使いください。');
    return;
  }
  const video = $('video');
  video.srcObject = stream;
  await video.play();
  $('scan-status').textContent = '顔の読み取りを準備しています…（初回は少し時間がかかります）';
  let lmk;
  try {
    lmk = await getLandmarker('VIDEO');
  } catch (e) {
    stopCamera();
    show('s-intro');
    introMessage('読み取りの準備に失敗しました。通信状態を確かめて、もう一度お試しください。');
    console.error(e);
    return;
  }
  if (!stream) return; // 準備中に「やめる」が押された

  const canvas = $('overlay');
  const ctx = canvas.getContext('2d');
  const samples = [];
  let lastTs = -1;

  const loop = () => {
    if (!stream) return;
    const W = video.videoWidth, H = video.videoHeight;
    if (W && H && video.currentTime !== lastTs) {
      lastTs = video.currentTime;
      if (canvas.width !== W) { canvas.width = W; canvas.height = H; }
      const res = lmk.detectForVideo(video, performance.now());
      ctx.clearRect(0, 0, W, H);
      const lm = res.faceLandmarks?.[0];
      if (!lm) {
        $('scan-status').textContent = '顔が見つかりません。画面の中に顔を入れてください';
      } else {
        const m = measure(lm, W, H);
        const problem = checkPose(m, jawOpen(res));
        drawOutline(ctx, lm, W, H, problem ? '#e8e2d0' : '#c9a44c');
        if (problem) {
          $('scan-status').textContent = problem;
        } else {
          samples.push(m);
          $('scan-status').textContent = 'そのまま動かないでください…';
        }
        if (DEBUG) $('scan-status').textContent += ` [L${m.L.toFixed(3)} J${m.J.toFixed(3)} F${m.F.toFixed(3)} C${m.C.toFixed(3)} s${m.size.toFixed(2)} r${m.roll.toFixed(1)} y${m.yaw.toFixed(3)} p${m.pitch.toFixed(3)}]`;
        $('scan-bar').style.width = `${Math.min(100, samples.length / SAMPLES_NEEDED * 100)}%`;
        if (samples.length >= SAMPLES_NEEDED) {
          stopCamera();
          const f = {};
          for (const key of ['L', 'J', 'F', 'C']) f[key] = median(samples.map((s) => s[key]));
          showResult(classify(f), f);
          return;
        }
      }
    }
    rafId = requestAnimationFrame(loop);
  };
  rafId = requestAnimationFrame(loop);
}

function drawOutline(ctx, lm, W, H, color) {
  ctx.strokeStyle = color;
  ctx.lineWidth = Math.max(2, W / 200);
  ctx.beginPath();
  FACE_OVAL.forEach((i, n) => {
    const x = lm[i].x * W, y = lm[i].y * H;
    if (n) ctx.lineTo(x, y); else ctx.moveTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = color;
  for (const i of [P.top, P.chin, P.cheekL, P.cheekR, P.jawL, P.jawR, P.foreL, P.foreR]) {
    ctx.beginPath();
    ctx.arc(lm[i].x * W, lm[i].y * H, Math.max(3, W / 120), 0, Math.PI * 2);
    ctx.fill();
  }
}

// ---- 写真で調べる ----
async function analyzePhoto(file) {
  introMessage('写真を読み取っています…（初回は少し時間がかかります）');
  let bmp;
  try {
    bmp = await createImageBitmap(file, { imageOrientation: 'from-image' });
  } catch {
    introMessage('写真を読み込めませんでした。別の写真でお試しください。');
    return;
  }
  try {
    const lmk = await getLandmarker('IMAGE');
    const res = lmk.detect(bmp);
    const lm = res.faceLandmarks?.[0];
    if (!lm) { introMessage('写真の中に顔が見つかりませんでした。正面から撮った、顔が大きめに写っている写真でお試しください。'); return; }
    const m = measure(lm, bmp.width, bmp.height);
    const problem = checkPose(m, jawOpen(res), [0.08, 1]);
    if (problem) { introMessage(`この写真では正しく測れません（${problem}）。正面から撮った、真顔の写真でお試しください。`); return; }
    introMessage('');
    showResult(classify(m), m);
  } catch (e) {
    console.error(e);
    introMessage('読み取りに失敗しました。通信状態を確かめて、もう一度お試しください。');
  } finally {
    bmp.close?.();
  }
}

// ---- 質問で調べる ----
const QUIZ = [
  {
    q: '顔の縦の長さ（生え際〜あご先）と、横幅（いちばん広いところ）を比べると？',
    a: [
      { t: '縦のほうがかなり長い', s: { long: 3 } },
      { t: '縦のほうが少し長い', s: { oval: 2, invtri: 1 } },
      { t: '縦と横が同じくらい', s: { round: 2, base: 2 } },
    ],
  },
  {
    q: 'エラ（耳の下の、あごの角）は？',
    a: [
      { t: '張っていて、角ばって見える', s: { base: 3 } },
      { t: '目立たず、全体に丸みがある', s: { round: 3 } },
      { t: '目立たず、すっきりしている', s: { oval: 1, long: 1, invtri: 1 } },
    ],
  },
  {
    q: 'ひたいとあご先を比べると？',
    a: [
      { t: 'ひたいが広く、あご先は細い', s: { invtri: 3 } },
      { t: 'どちらかが目立つ感じはない', s: { oval: 1, round: 1, base: 1, long: 1 } },
    ],
  },
];

function renderQuiz() {
  const wrap = $('quiz');
  wrap.innerHTML = '';
  QUIZ.forEach((item, qi) => {
    const div = document.createElement('div');
    div.className = 'q';
    const p = document.createElement('p');
    p.textContent = `Q${qi + 1}. ${item.q}`;
    div.appendChild(p);
    item.a.forEach((ans, ai) => {
      const label = document.createElement('label');
      const input = document.createElement('input');
      input.type = 'radio'; input.name = `q${qi}`; input.value = ai;
      input.addEventListener('change', () => {
        $('btn-quiz-done').disabled = QUIZ.some((_, i) => !document.querySelector(`input[name="q${i}"]:checked`));
      });
      label.append(input, ans.t);
      div.appendChild(label);
    });
    wrap.appendChild(div);
  });
  $('btn-quiz-done').disabled = true;
}

function quizResult() {
  const score = Object.fromEntries(ORDER.map((k) => [k, 0]));
  QUIZ.forEach((item, qi) => {
    const ai = Number(document.querySelector(`input[name="q${qi}"]:checked`).value);
    for (const [k, v] of Object.entries(item.a[ai].s)) score[k] += v;
  });
  const total = Object.values(score).reduce((a, b) => a + b, 0) || 1;
  return ORDER.map((k) => ({ k, p: score[k] / total })).sort((a, b) => b.p - a.p);
}

// ---- 結果 ----
let lastText = '';
function showResult(ranked, f) {
  const top = ranked[0], second = ranked[1];
  const near = second.p >= top.p * 0.6;
  $('r-type').textContent = TYPES[top.k].name;
  $('r-sub').textContent = near ? `${TYPES[second.k].name}にも近い形です` : 'いちばん近い形です（目安）';

  if (f) {
    const list = $('r-measure-list');
    list.innerHTML = '';
    const rows = [
      ['縦の長さ（横幅に対して）', level(f.L, AVG.L, 0.04, ['短め', '平均的', '長め'])],
      ['エラのあたりの幅', level(f.J, AVG.J, 0.05, ['細め', '平均的', '広め'])],
      ['ひたいの幅', level(f.F, AVG.F, 0.04, ['狭め', '平均的', '広め'])],
      ['あご先', level(f.C, AVG.C, 0.06, ['細め', '平均的', '広め'])],
    ];
    for (const [k, v] of rows) {
      const li = document.createElement('li');
      li.innerHTML = `<span>${k}</span><b>${v}</b>`;
      list.appendChild(li);
    }
    if (DEBUG) {
      const li = document.createElement('li');
      li.textContent = `L${f.L.toFixed(3)} J${f.J.toFixed(3)} F${f.F.toFixed(3)} C${f.C.toFixed(3)} / ${ranked.map((r) => `${r.k}${(r.p * 100).toFixed(0)}`).join(' ')}`;
      list.appendChild(li);
    }
    $('r-measure').hidden = false;
  } else {
    $('r-measure').hidden = true;
  }

  renderTabs(top.k);
  renderHair(top.k);
  renderLinks();

  const t = TYPES[top.k];
  lastText = [
    `顔の形スキャンの結果：${t.name}${near ? `（${TYPES[second.k].name}にも近い）` : ''}`,
    `似合いやすい髪型：${t.good.join('／')}`,
    `美容院での伝え方：${t.say}`,
    'https://tools.ikeojikeikaku.com/face/',
  ].join('\n');
  show('s-result');
}

function renderHair(k) {
  const t = TYPES[k];
  $('r-hair-title').textContent = `${t.name}に似合いやすい髪型`;
  const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  $('r-hair').innerHTML = `
    <div class="hair">
      <p>${esc(t.feature)}</p>
      <h3>おすすめのポイント</h3>
      <ul>${t.good.map((g) => `<li>${esc(g)}</li>`).join('')}</ul>
      <h3>避けたほうがいい髪型</h3>
      <ul>${t.avoid.map((g) => `<li>${esc(g)}</li>`).join('')}</ul>
      <h3>美容院での伝え方の例</h3>
      <p class="say">${esc(t.say)}</p>
    </div>`;
}

function renderTabs(active) {
  const wrap = $('r-tabs');
  wrap.innerHTML = '';
  for (const k of ORDER) {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = TYPES[k].name;
    b.setAttribute('aria-pressed', String(k === active));
    b.addEventListener('click', () => {
      wrap.querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
      renderHair(k);
      $('r-hair-title').scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
    wrap.appendChild(b);
  }
}

function renderLinks() {
  const now = Date.now();
  const items = LINKS.filter((l) => Date.parse(l.from) <= now);
  const ul = $('r-links-list');
  ul.innerHTML = '';
  for (const l of items) {
    const li = document.createElement('li');
    const a = document.createElement('a');
    a.href = l.url; a.textContent = l.title;
    li.appendChild(a);
    ul.appendChild(li);
  }
  $('r-links').hidden = items.length === 0;
}

// ---- イベント ----
$('btn-camera').addEventListener('click', startCamera);
$('btn-cancel').addEventListener('click', () => { stopCamera(); show('s-intro'); });
$('file-input').addEventListener('change', (e) => {
  const file = e.target.files?.[0];
  e.target.value = '';
  if (file) analyzePhoto(file);
});
$('btn-quiz').addEventListener('click', () => { renderQuiz(); show('s-quiz'); });
$('btn-quiz-done').addEventListener('click', () => showResult(quizResult(), null));
document.querySelectorAll('.js-back').forEach((b) => b.addEventListener('click', () => { introMessage(''); show('s-intro'); }));
$('btn-copy').addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(lastText);
    $('btn-copy').textContent = 'コピーしました';
  } catch {
    $('btn-copy').textContent = 'コピーできませんでした';
  }
  setTimeout(() => { $('btn-copy').textContent = '結果をコピーする'; }, 2000);
});
document.addEventListener('visibilitychange', () => {
  if (document.hidden && stream) { stopCamera(); show('s-intro'); }
});

// 試験用：?debug=1 のとき、コンソールから画像URLで確かめられるようにする
if (DEBUG) {
  window.__testImage = async (url) => {
    const blob = await (await fetch(url)).blob();
    await analyzePhoto(blob);
    return lastText;
  };
}
