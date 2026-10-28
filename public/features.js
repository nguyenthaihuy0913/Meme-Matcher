export const FEATURE_VERSION = 2;
export const clamp = x => Math.max(0, Math.min(1, x));
const dist = (a, b) => (a && b) ? Math.hypot(a.x - b.x, a.y - b.y) : 99;
const angle = (a, b, c) => {
  const u = [a.x - b.x, a.y - b.y], v = [c.x - b.x, c.y - b.y];
  return Math.acos(Math.max(-1, Math.min(1, (u[0] * v[0] + u[1] * v[1]) / (Math.hypot(...u) * Math.hypot(...v) || 1)))) * 180 / Math.PI;
};

export function extract(face, hands, pose, width = 640, height = 480) {
  const aspect = width / height, correct = p => p ? { ...p, x: p.x * aspect } : null;
  const f = face?.faceLandmarks?.[0]?.map(correct), p = pose?.landmarks?.[0]?.map(correct);
  const blends = Object.fromEntries((face?.faceBlendshapes?.[0]?.categories || []).map(c => [c.categoryName, c.score]));
  const scale = f ? dist(f[234], f[454]) : (p ? dist(p[11], p[12]) : 0.25);
  const near = (a, b, r = 0.7) => clamp(1 - dist(a, b) / (Math.max(scale, 0.03) * r));

  let up = 0, turn = 0;
  const mat = face?.facialTransformationMatrixes?.[0]?.data;
  if (mat) {
    // MediaPipe's column-major matrix: negative X pitch means looking up.
    // Taking the absolute value used to confuse looking down with looking up.
    const pitch = -Math.atan2(mat[6], mat[10]);
    up = clamp(pitch / 0.40);
    turn = clamp(Math.abs(Math.atan2(mat[8], mat[10])) / 0.65);
  }
  if (f && !mat) {
    const dChin = f[152].y - f[1].y;
    const dFace = dist(f[10], f[152]) || 1.0;
    const upPts = clamp((dChin / dFace - 0.42) / 0.22);
    up = Math.max(up, upPts);
  }

  const h = (hands?.landmarks || []).map(raw => {
    const l = raw.map(correct);
    const extended = [angle(l[1], l[2], l[4]) > 145, ...[5, 9, 13, 17].map(i => angle(l[i], l[i + 1], l[i + 3]) > 155)];
    const index = extended[1] && !extended[2] && !extended[3] && !extended[4];
    return {
      extended,
      index: Number(index),
      fist: Number(!extended.slice(1).some(Boolean)),
      shaka: Number(extended[0] && extended[4] && !extended[1] && !extended[2] && !extended[3]),
      thumb: Number(extended[0] && !extended.slice(1).some(Boolean)),
      mouth: f ? near(l[8], f[13]) : 0,
      chin: f ? near(l[8], f[152], 1) : 0,
      cheek: f ? Math.max(near(l[8], f[234]), near(l[8], f[454])) : 0,
      vertical: clamp(1 - Math.abs(l[8].x - l[5].x) / (dist(l[8], l[5]) || 1)),
      tip: l[8],
      wrist: l[0]
    };
  });

  const visible = i => p && p[i] && (p[i].visibility ?? 1) > 0.2;
  const arms = p ? [[11, 13, 15], [12, 14, 16]].map(([s, e, w]) => {
    if (![s, e, w].every(visible)) return { head: 0, flex: 0, straight: 0, chest: 0 };
    const a = angle(p[s], p[e], p[w]);
    const head = f ? Math.max(near(p[w], f[234], 2.2), near(p[w], f[454], 2.2), near(p[w], f[10], 2.0)) : near(p[w], p[0], 2.2);
    return {
      head,
      flex: Number(a > 40 && a < 120 && p[e].y < p[s].y + 0.13 && Math.abs(p[e].x - p[s].x) > scale * 0.35),
      straight: Number(a > 145 && Math.abs(p[w].x - p[s].x) > scale),
      chest: near(p[w], { x: (p[11].x + p[12].x) / 2, y: (p[11].y + p[12].y) / 2 + 0.15 }, 1.4)
    };
  }) : [];

  let handNearHead = 0;
  if (f) {
    for (const hand of h) {
      for (const pt of [hand.tip, hand.wrist]) {
        const prox = Math.max(near(pt, f[234], 2.2), near(pt, f[454], 2.2), near(pt, f[10], 2.0));
        if (prox > handNearHead) handNearHead = prox;
      }
    }
  }

  const armHeads = arms.map(x => x.head).filter(v => v > 0.05);
  const oneHead = Math.max(0, ...arms.map(x => x.head), handNearHead);
  const twoHead = armHeads.length >= 2 ? Math.min(...armHeads) : 0;
  const flex = arms.reduce((n, x) => n + x.flex, 0) / 2;
  const straight = Math.max(0, ...arms.map(x => x.straight));
  const chest = Math.max(0, ...arms.map(x => x.chest));

  const max = fn => Math.max(0, ...h.map(fn));
  const jaw = clamp((blends.jawOpen || 0) * 2);
  const smile = Math.max(blends.mouthSmileLeft || 0, blends.mouthSmileRight || 0);

  const extra = {};
  // Continuous geometry preserves near-matches that hard finger/arm thresholds
  // discard. Magnitudes and sorted hands keep left/right mirror poses equivalent.
  for (const [name, indices] of Object.entries({thumb:[1,2,4],index:[5,6,8],middle:[9,10,12],ring:[13,14,16],pinky:[17,18,20]})) {
    const values = (hands?.landmarks || []).map(raw => {
      const l=raw.map(correct);
      return clamp((angle(...indices.map(i=>l[i]))-60)/120);
    }).sort((a,b)=>b-a);
    extra[`${name}Extension`] = values[0] || 0;
    extra[`${name}ExtensionSecond`] = values[1] || 0;
  }
  extra.headPitch = mat ? Math.atan2(mat[6],mat[10])/Math.PI : 0;
  extra.headYaw = mat ? Math.abs(Math.atan2(mat[8],mat[10]))/Math.PI : 0;
  extra.headRoll = mat ? Math.abs(Math.atan2(mat[1],mat[0]))/Math.PI : 0;
  extra.noseChinRatio = f ? clamp((f[152].y-f[1].y)/(dist(f[10],f[152])||1)) : 0;
  extra.faceTilt = f ? clamp(Math.abs(f[33].y-f[263].y)/(dist(f[33],f[263])||1)) : 0;
  const geometry = p ? [[11,13,15],[12,14,16]].map(([s,e,w])=>{
    if (![s,e,w].every(visible)) return null;
    const shoulderScale=Math.max(dist(p[11],p[12]),0.03);
    return {bend:angle(p[s],p[e],p[w])/180,
      wristHeight:Math.max(-2,Math.min(2,(p[s].y-p[w].y)/shoulderScale)),
      elbowHeight:Math.max(-2,Math.min(2,(p[s].y-p[e].y)/shoulderScale)),
      wristSide:clamp(Math.abs(p[w].x-p[s].x)/shoulderScale/2)};
  }).filter(Boolean) : [];
  for(const name of ['bend','wristHeight','elbowHeight','wristSide']) {
    const values=geometry.map(x=>x[name]).sort((a,b)=>b-a);
    extra[`${name}Max`]=values[0]??0; extra[`${name}Min`]=values.at(-1)??0;
  }
  extra.armsVisible=geometry.length/2;
  extra.handDistance=h.length===2?clamp(dist(h[0].wrist,h[1].wrist)/Math.max(scale,0.03)/4):0;
  extra.handHeadDistance=h.length&&f?clamp(Math.min(...h.map(x=>dist(x.wrist,f[1])))/Math.max(scale,0.03)/4):0;
  return {
    face: Number(!!f),
    hands: h.length,
    pose: Number(!!p),
    jaw,
    smile,
    up,
    turn,
    indexMouth: max(x => x.index * x.mouth * x.vertical),
    indexCheek: max(x => x.index * x.cheek),
    chin: max(x => x.chin),
    shaka: max(x => x.shaka),
    thumb: max(x => x.thumb),
    index: max(x => x.index),
    fists: h.filter(x => x.fist).length,
    tipsTogether: h.length === 2 ? h[0].index * h[1].index * near(h[0].tip, h[1].tip, 0.8) : 0,
    oneHead,
    twoHead,
    flex,
    straight,
    chest,
    ...extra
  };
}

