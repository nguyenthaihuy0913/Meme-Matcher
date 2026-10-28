import { modelWeights } from './model_weights.js';

export const memes = [
  ['Không khớp', 'Đứng bình thường hoặc làm hành động khác.'],
  ['Ngón tay gần má', 'Dựng ngón trỏ gần má; phiên bản quy ước của hình 1.'],
  ['Chỉ vào mình', 'Ngón trỏ hướng vào ngực.'],
  ['Suỵt', 'Ngón trỏ dựng giữa môi, các ngón khác gập.'],
  ['Tay sau gáy', 'Một tay sau đầu, khuỷu tay mở.'],
  ['Ngước lên', 'Ngẩng đầu nhìn lên.'],
  ['Ôm đầu bất ngờ', 'Hai tay sau đầu, khuỷu tay mở; há miệng.'],
  ['Shaka 🤙', 'Duỗi cái và út, gập ba ngón giữa; tay gần mặt.'],
  ['Hai ngón chạm nhau', 'Hai ngón trỏ hướng vào nhau dưới cằm.'],
  ['Khoe cơ', 'Hai khuỷu tay mở, gập tay và nắm tay.'],
  ['Chỉ ngang bất ngờ', 'Duỗi tay chỉ sang bên và há miệng.'],
  ['Chỉ ngón cái', 'Ngón cái chỉ sang bên, ngón khác gập; cười/há miệng.'],
  ['Suy nghĩ nghiêng', 'Quay mặt sang bên và đặt tay gần cằm.'],
  ['Suy nghĩ ngước', 'Ngẩng đầu, đưa ngón tay gần môi/cằm.']
];

import {clamp, extract} from './features.js';
export {extract} from './features.js';

import {predictModel} from './ml.js';

function predictML(f, weights) {
  return !f.face && !f.pose ? null : predictModel(weights,f);
}

export function classify(f, weights = modelWeights) {
  const ruleScores = [
    0,
    f.indexCheek * 0.85,
    f.index * f.chest,
    f.indexMouth * (1 - 0.7 * f.up),
    f.oneHead * (1 - f.twoHead) * (0.6 + 0.4 * f.turn),
    f.face * f.up * (0.7 + 0.3 * f.jaw) * (1 - f.oneHead * 0.5) * (f.hands ? 0.7 : 1.0),
    f.twoHead * (0.65 + 0.35 * f.jaw),
    f.shaka * 0.9,
    f.tipsTogether,
    f.flex * (f.fists === 2 ? 1 : 0.4),
    f.straight * (0.6 + 0.4 * Math.max(f.index, f.jaw)),
    f.thumb * (0.65 + 0.35 * Math.max(f.jaw, f.smile)),
    f.chin * f.turn,
    f.chin * f.up
  ];
  const mlProbs = predictML(f, weights);
  const scores = [0];
  for (let i = 1; i <= 13; i++) {
    const ml = mlProbs ? mlProbs[i] : 0, rule = ruleScores[i] || 0;
    const combined = mlProbs
      ? Math.max(ml * 0.7 + rule * 0.3, rule * 0.85, ml * 0.85)
      : rule;
    // A learned prior must not invent a hand or arm gesture after tracking is lost.
    const supported = ([1, 2, 3, 7, 11].includes(i) ? f.hands > 0 : true)
      && (i === 8 ? f.hands === 2 : true)
      && ([4, 6, 9, 10].includes(i) ? !!f.pose || f.hands > 0 : true)
      && ([1, 3, 5, 12, 13].includes(i) ? !!f.face : true);
    scores.push(supported ? clamp(combined) : 0);
  }
  const ranked = scores.slice(1).map((score, i) => ({ id: i + 1, score: clamp(score) })).sort((a, b) => b.score - a.score);
  const normalScore = mlProbs ? mlProbs[0] * 0.8 : 0;
  const lead = ranked[0];
  const matched = lead.score >= 0.48 && (lead.score - ranked[1].score) >= 0.08 && lead.score > normalScore;
  return { label: matched ? lead.id : 0, uncertain: !matched, top: ranked.slice(0, 3), scores };
}

// Time-based smoothing works across different webcam frame rates. Reject stale
// detections immediately; require a short stable gesture before showing a label.
export function createPredictionTracker() {
  let scores = Array(14).fill(0), last = null, candidate = 0, since = 0;
  return {
    reset() { scores.fill(0); last = null; candidate = 0; since = 0; },
    update(features, now) {
      const raw = classify(features);
      if ((!features.face && !features.pose && !features.hands) || (last !== null && (now - last > 600 || now < last))) this.reset();
      const alpha = last === null ? 1 : 1 - Math.exp(-(now - last) / 160);
      last = now;
      const current = Array(14).fill(0);
      raw.top.forEach(x => { current[x.id] = x.score; });
      scores = scores.map((s, id) => current[id] > 0 ? s + alpha * (current[id] - s) : 0);
      const top = scores.slice(1).map((score, i) => ({ id: i + 1, score })).sort((a, b) => b.score - a.score).slice(0, 3);
      const next = raw.label && top[0].id === raw.label && top[0].score >= 0.48 && top[0].score - top[1].score >= 0.08 ? raw.label : 0;
      if (next !== candidate) { candidate = next; since = now; }
      const label = candidate && now - since >= 200 ? candidate : 0;
      return { label, uncertain: !label, top };
    }
  };
}
