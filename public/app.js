import {FilesetResolver,FaceLandmarker,HandLandmarker,PoseLandmarker} from '/vendor/vision_bundle.mjs';
import {memes,extract,classify} from './classifier.js';
import {MemeGame,MATCH_THRESHOLD} from './game.js';
const $=id=>document.getElementById(id);
const video=$('video'),canvas=$('overlay'),ctx=canvas.getContext('2d'),game=new MemeGame();
let tasks=null,stream=null,cameraRunning=false,busy=false,lastInference=-Infinity,lastVideo=-1;
let countdownAt=null,countdownNumber=null,displayedTarget=7,resultShown=false,matchSmooth=0,lastScoreAt=null;

function hideTarget() {
  $('targetImage').hidden=true;
  $('memeBadge').hidden=true;
  $('targetImage').removeAttribute('src');
  $('memePlaceholder').hidden=false;
  $('targetNumber').textContent='SẴN SÀNG KHÁM PHÁ';
  $('targetName').textContent='Thử thách đang chờ bạn';
  $('targetHint').textContent='Meme sẽ xuất hiện khi lượt chơi bắt đầu.';
}
function setTarget(id) {
  $('targetImage').hidden=false;
  $('memeBadge').hidden=false;
  $('memePlaceholder').hidden=true;
  displayedTarget=id;
  $('targetImage').src=`memes/${id}.jpg`;
  $('targetImage').alt=`Meme mục tiêu: ${memes[id][0]}`;
  $('targetName').textContent=memes[id][0];
  $('targetHint').textContent=memes[id][1];
  $('targetNumber').textContent=`MEME #${String(id).padStart(2,'0')}`;
  $('targetCount').textContent=`${String(Math.min(5,(game.completed||0)+1)).padStart(2,'0')} / 05`;
  matchSmooth=0;lastScoreAt=null;renderMatch(0,0,'Nhái theo meme<br>bên trái nhé');
}
function renderSteps() {
  $('steps').replaceChildren(...Array.from({length:5},(_,i)=>{
    const li=document.createElement('li'),done=i<(game.completed||0);
    li.className=done?'done':i===(game.completed||0)?'active':'';
    li.textContent=done?'✓':String(i+1).padStart(2,'0');
    li.setAttribute('aria-label',`Meme ${i+1}: ${done?'đã hoàn thành':i===(game.completed||0)?'hiện tại':'chưa chơi'}`);
    return li;
  }));
}
function renderMatch(score,hold,text) {
  const percent=Math.round(Math.max(0,Math.min(1,score))*100);
  $('matchValue').innerHTML=`${percent}<small>%</small>`;
  $('matchRing').style.setProperty('--match',`${percent}%`);
  $('holdBar').style.width=`${Math.round(hold*100)}%`;
  $('holdTrack').setAttribute('aria-valuenow',Math.round(hold*100));
  if($('matchStatus').innerHTML!==text)$('matchStatus').innerHTML=text;
}
async function loadModels() {
  if(tasks)return;
  const fs=await FilesetResolver.forVisionTasks('/vendor/wasm');
  const options=name=>({baseOptions:{modelAssetPath:`/models/${name}.task`,delegate:'CPU'},runningMode:'VIDEO'});
  // Close partially created models if loading fails, so the next attempt is clean.
  const loaded={};
  try {
    loaded.face=await FaceLandmarker.createFromOptions(fs,{...options('face'),numFaces:1,outputFaceBlendshapes:true,outputFacialTransformationMatrixes:true});
    loaded.hand=await HandLandmarker.createFromOptions(fs,{...options('hand'),numHands:2});
    loaded.pose=await PoseLandmarker.createFromOptions(fs,{...options('pose'),numPoses:1});
    tasks=loaded;
  } catch(error) {Object.values(loaded).forEach(model=>model.close());throw error;}
}
async function enableCamera() {
  if(cameraRunning)return true;
  if(busy)return false;
  busy=true;$('camera').disabled=true;$('play').disabled=true;
  $('cameraStatus').textContent='Đang tải bộ nhận diện trên máy…';
  try {
    await loadModels();
    stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:960},height:{ideal:720}},audio:false});
    video.srcObject=stream;await video.play();
    canvas.width=video.videoWidth;canvas.height=video.videoHeight;
    cameraRunning=true;lastVideo=-1;lastInference=-Infinity;
    for(const track of stream.getVideoTracks())track.addEventListener('ended',()=>stopCamera('Camera bị ngắt. Bật lại để tiếp tục thử thách.'),{once:true});
    $('placeholder').hidden=true;$('tracking').hidden=false;$('cameraBadge').textContent='LIVE · SKELETON';
    $('camera').textContent='Tắt camera';
    $('cameraStatus').textContent='Đứng lùi để thấy mặt, vai và hai tay.';
    return true;
  } catch(error) {
    stopCamera();
    $('cameraStatus').textContent=error.name==='NotAllowedError'?'Chưa được cấp quyền camera. Cho phép camera trong trình duyệt rồi thử lại.':`Không bật được camera: ${error.message}`;
    return false;
  } finally {busy=false;$('camera').disabled=false;$('play').disabled=false;}
}
function cancelCountdown() {countdownAt=null;countdownNumber=null;$('countdown').hidden=true;}
function stopCamera(message='Camera đã tắt. Hình ảnh không được lưu.') {
  if(game.state==='playing'){game.finish(performance.now(),'camera');showResults();}
  cancelCountdown();cameraRunning=false;
  stream?.getTracks().forEach(track=>track.stop());stream=null;video.srcObject=null;
  ctx.clearRect(0,0,canvas.width,canvas.height);$('placeholder').hidden=false;$('tracking').hidden=true;
  $('cameraBadge').textContent='CAMERA OFF';$('camera').textContent='Bật camera ↗';
  $('tracking').textContent='● Đang chờ camera';$('cameraStatus').textContent=message;
  $('play').hidden=false;$('stop').hidden=true;document.body.classList.remove('playing');
  renderMatch(0,0,'Chờ tín hiệu<br>từ bạn');
}
async function prepareGame() {
  if(busy||countdownAt!==null||game.state==='playing')return;
  hideTarget();
  if($('results').open)$('results').close();
  if(!await enableCamera())return;
  game.state='countdown';countdownAt=performance.now();countdownNumber=null;resultShown=false;
  $('play').hidden=true;$('stop').hidden=false;
  $('gameStatus').textContent='Vào vị trí! Đồng hồ bắt đầu sau đếm ngược.';
  $('countdown').hidden=false;
}
function startGame(now) {
  cancelCountdown();game.start(now);renderSteps();setTarget(game.target);
  document.body.classList.add('playing');
  $('gameStatus').textContent=`Khớp từ ${Math.round(MATCH_THRESHOLD*100)}% và giữ đúng dáng 0,65 giây để qua meme tiếp theo.`;
}
function showResults() {
  if(resultShown)return;resultShown=true;
  hideTarget();
  document.body.classList.remove('playing');$('play').hidden=false;$('stop').hidden=true;
  const win=game.reason==='win';
  $('resultTitle').textContent=win?'Chinh phục thiên hà meme!':'Mỗi lần thử, một lần tỏa sáng!';
  $('resultIcon').textContent=win?'✦':'✧';
  $('resultSubtitle').textContent=win?`Bạn hoàn thành 5 meme trong ${((game.ended-game.started)/1000).toFixed(1)} giây.`:game.reason==='camera'?'Camera đã ngắt. Chuyến đi mới luôn chờ bạn.':game.reason==='stopped'?'Đã kết thúc lượt chơi. Sẵn sàng cho chuyến tiếp theo?':'Hết 20 giây! Một chuyến khám phá thật đáng nhớ.';
  $('resultScore').innerHTML=`${game.completed}<small>/5</small>`;
  $('resultMessage').textContent=game.message;
  $('resultMemes').replaceChildren(...game.targets.map((id,i)=>{
    const item=document.createElement('div');item.className=i<game.completed?'done':'';
    const img=document.createElement('img');img.src=`memes/${id}.jpg`;img.alt=memes[id][0];
    const caption=document.createElement('span');caption.textContent=i<game.completed?'✓':'—';
    item.append(img,caption);return item;
  }));
  $('gameStatus').textContent=`Đã hoàn thành ${game.completed}/5 meme. Thử lại để khám phá bộ meme mới!`;
  renderSteps();$('results').showModal();
}

const poseEdges=[[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],[23,25],[25,27],[24,26],[26,28],[27,29],[29,31],[28,30],[30,32],[0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],[9,10]];
const handEdges=[[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
function drawSkeleton(result) {
  ctx.clearRect(0,0,canvas.width,canvas.height);
  function draw(points,edges,color) {
    if(!points)return;
    const visible=p=>p&&Number.isFinite(p.x)&&Number.isFinite(p.y)&&(p.visibility??1)>.35;
    ctx.lineCap='round';ctx.shadowColor=color;ctx.shadowBlur=9;
    for(const [a,b] of edges) {
      const p=points[a],q=points[b];if(!visible(p)||!visible(q))continue;
      const depth=Math.max(.5,Math.min(1.5,1-((p.z||0)+(q.z||0))/2));
      ctx.strokeStyle=color;ctx.globalAlpha=Math.min(1,.5+depth*.25);ctx.lineWidth=2.3*depth;
      ctx.beginPath();ctx.moveTo(p.x*canvas.width,p.y*canvas.height);ctx.lineTo(q.x*canvas.width,q.y*canvas.height);ctx.stroke();
    }
    for(const p of points) {
      if(!visible(p))continue;
      const radius=Math.max(2,Math.min(5,3-(p.z||0)*2));
      ctx.globalAlpha=.95;ctx.fillStyle='#eaffff';ctx.beginPath();ctx.arc(p.x*canvas.width,p.y*canvas.height,radius,0,Math.PI*2);ctx.fill();
    }
    ctx.shadowBlur=0;ctx.globalAlpha=1;
  }
  draw(result.pose.landmarks?.[0],poseEdges,'#65ece0');
  for(const points of result.hand.landmarks||[])draw(points,handEdges,'#c4a3ff');
  const face=result.face.faceLandmarks?.[0];
  if(face)draw(face.filter((_,i)=>[1,4,33,133,263,362,61,291,152].includes(i)),[],'#f1d4ff');
}
function infer(now) {
  const result={face:tasks.face.detectForVideo(video,now),hand:tasks.hand.detectForVideo(video,now),pose:tasks.pose.detectForVideo(video,now)};
  drawSkeleton(result);
  const f=extract(result.face,result.hand,result.pose,video.videoWidth,video.videoHeight),prediction=classify(f);
  $('tracking').textContent=`● MẶT ${f.face?'✓':'—'} · TAY ${f.hands}/2 · THÂN ${f.pose?'✓':'—'}`;
  if(game.state==='playing') {
    const advanced=game.observe(prediction,performance.now());
    if(advanced) {
      renderSteps();
      if(game.state==='playing') {
        setTarget(game.target);const panel=document.querySelector('.target-panel');panel.classList.remove('success-flash');void panel.offsetWidth;panel.classList.add('success-flash');
      }
    } else {
      const alpha=lastScoreAt===null?1:1-Math.exp(-(now-lastScoreAt)/120);
      matchSmooth=game.match===0?0:matchSmooth+alpha*(game.match-matchSmooth);lastScoreAt=now;
      const needsHands=[1,2,3,7,8,11,12,13].includes(game.target);
      const missingHands=needsHands && f.hands < (game.target===8?2:1);
      const text=game.hold>0?'Đúng rồi!<br>Giữ dáng nhé ✦':!f.face&&!f.pose&&!f.hands?'Chưa thấy bạn<br>Bước vào khung hình':missingHands?'Đưa tay vào khung<br>để thấy rõ các ngón':game.match>=MATCH_THRESHOLD?'Gần đúng rồi<br>Chỉnh thêm một chút':'Nhái theo meme<br>bên trái nhé';
      renderMatch(matchSmooth,game.hold,text);
    }
  }
}
function frame(now) {
  try {
    if(countdownAt!==null) {
      const elapsed=now-countdownAt;
      if(elapsed>=3000)startGame(now);
      else {const value=String(3-Math.floor(elapsed/1000));if(value!==countdownNumber){countdownNumber=value;$('countdown').textContent=value;}}
    }
    if(game.state==='playing') {
      game.tick(now);
      const remaining=game.remaining(now)/1000,whole=Math.floor(remaining),decimal=Math.floor((remaining-whole)*10);
      $('timer').innerHTML=`${String(whole).padStart(2,'0')}<small>.${decimal}s</small>`;
      $('timerBox').classList.toggle('urgent',remaining<=5);
    }
    if(cameraRunning&&video.readyState>=2&&now-lastInference>=90&&video.currentTime!==lastVideo) {
      lastInference=now;lastVideo=video.currentTime;infer(now);
    }
    if(game.state==='finished'&&!resultShown)showResults();
  } catch(error) {
    stopCamera(`Nhận diện bị gián đoạn: ${error.message}. Bật lại camera để thử tiếp.`);
  }
  requestAnimationFrame(frame);
}
$('camera').onclick=()=>{if(cameraRunning)stopCamera();else enableCamera();};
$('play').onclick=prepareGame;$('replay').onclick=prepareGame;
$('closeResults').onclick=()=>{$('results').close();$('play').focus();};
$('stop').onclick=()=>{
  if(countdownAt!==null){cancelCountdown();game.state='idle';$('play').hidden=false;$('stop').hidden=true;$('gameStatus').textContent='Đã hủy đếm ngược. Bắt đầu khi bạn sẵn sàng.';}
  else if(game.state==='playing'){game.finish(performance.now(),'stopped');showResults();}
};
$('fullscreen').onclick=async()=>{try {if(document.fullscreenElement)await document.exitFullscreen();else await document.documentElement.requestFullscreen();}catch{$('gameStatus').textContent='Trình duyệt chưa cho phép toàn màn hình. Bạn có thể dùng phím F11.';}};
document.addEventListener('visibilitychange',()=>{
  if(document.hidden){
    if(countdownAt!==null){cancelCountdown();$('play').hidden=false;$('stop').hidden=true;}
    if(game.state==='playing'){game.finish(performance.now(),'stopped');showResults();}
  }
});
window.addEventListener('pagehide',()=>{cameraRunning=false;stream?.getTracks().forEach(track=>track.stop());});
renderSteps();requestAnimationFrame(frame);
