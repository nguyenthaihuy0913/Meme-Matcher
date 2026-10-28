export const DURATION = 20000;
export const MATCH_THRESHOLD = 0.62;
export const HOLD_TIME = 650;
export const messages = [
  'Dù kết quả ra sao, bất kể bạn là ai, không rào cản nào có thể đánh gục giấc mơ công nghệ của bạn. Hãy tiếp tục tiến về phía trước!',
  'Thuật toán không có định kiến! Các bạn nữ hãy tự tin bứt phá, làm chủ công nghệ và dẫn đầu tương lai.',
  'Chung tay dùng công nghệ để lan tỏa bình đẳng giới, thúc đẩy các sáng kiến trẻ và kiến tạo giá trị tích cực cho cộng đồng!',
  'Bất kể bạn là ai, hãy tự tin dấn thân, bứt phá giới hạn và làm chủ tương lai công nghệ!',
  'Công nghệ không giới hạn – Cơ hội dành cho tất cả mọi người!',
  'Mã nguồn không có giới tính, tương lai cần nhân tài.',
  'Thuật toán không thiên kiến, thế giới mới công bằng.',
  'Công nghệ không phân biệt nam nữ, chỉ cần tài năng và khát vọng.',
  'Sáng tạo không rào cản, công nghệ không giới hạn.',
  'Hãy để tiếng nói của mọi giới tính cùng định hình kỷ nguyên số.',
  'Công nghệ thuộc về tất cả chúng ta – Bình đẳng trong từng dòng code.',
  'Dải ngân hà số rực rỡ nhất khi mọi vì sao đều có cơ hội tỏa sáng.',
  'Thuật toán chỉ là đường nối, chính các vì sao – thuộc mọi giới tính – mới tạo nên chòm sao kiệt tác.',
  'Vũ trụ công nghệ bao la, không có bầu trời nào giới hạn ánh sáng của một vì sao.',
  'Mỗi dòng code là một đốm sáng, mỗi tài năng là một vì sao đưa thế giới số qua đêm tối.',
  'Bình đẳng giới là khi mọi vì sao trong hệ mặt trời công nghệ đều nhận được lực hấp dẫn như nhau.',
  'Đừng chỉ nhìn lên, hãy để công nghệ là bệ phóng đưa mọi cô gái chạm đến vì sao của chính mình.',
  'Định vị lại tương lai: Nơi giới tính không quyết định độ sáng của một ngôi sao băng.',
  'Công nghệ đỉnh cao là kính thiên văn giúp thế giới nhìn thấy mọi vì sao bị lãng quên.',
  'Bàn phím và chuột là công cụ vẽ nên bản đồ sao, nơi mọi giới tính cùng định vị thiên hà mới.'
];

export function shuffledMemes(random = Math.random) {
  const ids = Array.from({ length: 13 }, (_, i) => i + 1);
  for (let i = ids.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [ids[i], ids[j]] = [ids[j], ids[i]];
  }
  return ids.slice(0, 5);
}

export class MemeGame {
  constructor(random = Math.random) { this.random = random; this.state = 'idle'; }
  start(now) {
    this.targets = shuffledMemes(this.random);
    this.started = now; this.deadline = now + DURATION;
    this.completed = 0; this.holdStart = null; this.lastSample = null;
    this.match = 0; this.hold = 0; this.peaks = Array(5).fill(0);
    this.state = 'playing'; this.message = '';
  }
  get target() { return this.targets?.[this.completed] ?? null; }
  remaining(now) { return Math.max(0, this.deadline - now); }
  finish(now, reason = 'timeout') {
    if (this.state !== 'playing') return;
    this.state = 'finished'; this.reason = reason; this.ended = now;
    this.hold = 0;
    this.message = messages[Math.floor(this.random() * messages.length)];
  }
  tick(now) { if (this.state === 'playing' && now >= this.deadline) this.finish(this.deadline); }
  observe(prediction, now) {
    this.tick(now);
    if (this.state !== 'playing') return false;
    const score = prediction?.scores?.[this.target] ?? 0;
    this.match = Number.isFinite(score) ? Math.max(0, Math.min(1, score)) : 0;
    this.peaks[this.completed] = Math.max(this.peaks[this.completed], this.match);
    const qualifies = this.match >= MATCH_THRESHOLD && prediction.label === this.target;
    if (!qualifies || this.lastSample === null || now - this.lastSample > 350 || now < this.lastSample) this.holdStart = null;
    this.lastSample = now;
    if (!qualifies) { this.hold = 0; return false; }
    this.holdStart ??= now;
    this.hold = Math.min(1, (now - this.holdStart) / HOLD_TIME);
    if (this.hold < 1) return false;
    this.completed++; this.holdStart = null; this.lastSample = null; this.hold = 0; this.match = 0;
    if (this.completed === 5) this.finish(now, 'win');
    return true;
  }
}
