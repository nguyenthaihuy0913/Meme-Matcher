import {test} from 'node:test';
import assert from 'node:assert/strict';
import {MemeGame,shuffledMemes,messages,DURATION,HOLD_TIME} from './public/game.js';
const prediction=(id,score=0.9,label=id)=>({label,scores:Array.from({length:14},(_,i)=>i===id?score:0)});
function hold(game,start) { for(let t=start;t<=start+700;t+=100)game.observe(prediction(game.target),t); }
test('draw five unique meme IDs in range',()=>{
 for(let n=0;n<100;n++){const ids=shuffledMemes();assert.equal(ids.length,5);assert.equal(new Set(ids).size,5);assert.ok(ids.every(id=>id>=1&&id<=13));}
});
test('timer expires at 20 seconds without needing camera frames',()=>{
 const game=new MemeGame();game.start(1000);game.tick(20999);assert.equal(game.state,'playing');game.tick(21000);
 assert.equal(game.state,'finished');assert.equal(game.reason,'timeout');assert.ok(messages.includes(game.message));
 assert.equal(game.remaining(22000),0);
});
test('brief matches, wrong label and low scores do not pass',()=>{
 const game=new MemeGame();game.start(0);game.observe(prediction(game.target),0);game.observe(prediction(game.target),100);
 game.observe(prediction(game.target,.9,0),200);assert.equal(game.hold,0);
 game.observe(prediction(game.target,.4),300);assert.equal(game.completed,0);
});
test('stable evidence passes once and resets hold for next meme',()=>{
 const game=new MemeGame();game.start(0);const first=game.target;hold(game,0);
 assert.equal(game.completed,1);assert.notEqual(game.target,first);assert.equal(game.hold,0);
 game.observe(prediction(first),800);assert.equal(game.completed,1);
});
test('completing a meme resets the 20-second timer for the next meme',()=>{
 const game=new MemeGame();game.start(0);hold(game,15000);
 assert.equal(game.completed,1);assert.equal(game.state,'playing');
 assert.equal(game.remaining(15700),DURATION);
 game.tick(30000);assert.equal(game.state,'playing');assert.equal(game.remaining(30000),5700);
 game.tick(35700);assert.equal(game.state,'finished');assert.equal(game.reason,'timeout');assert.equal(game.completed,1);
});
test('tracking gap does not count toward hold time',()=>{
 const game=new MemeGame();game.start(0);game.observe(prediction(game.target),0);game.observe(prediction(game.target),HOLD_TIME+1);
 assert.equal(game.completed,0);assert.equal(game.hold,0);
 game.observe(null,800);assert.equal(game.match,0);
});
test('all five matches finish with win, replay resets state',()=>{
 const game=new MemeGame();game.start(0);for(let i=0;i<5;i++)hold(game,i*900);
 assert.equal(game.completed,5);assert.equal(game.reason,'win');assert.equal(game.state,'finished');assert.ok(messages.includes(game.message));
 game.start(10000);assert.equal(game.completed,0);assert.equal(game.state,'playing');assert.equal(game.message,'');assert.equal(game.remaining(10000),DURATION);
});
test('a match at the deadline cannot win after timeout',()=>{
 const game=new MemeGame();game.start(0);hold(game,DURATION-700);assert.equal(game.state,'finished');assert.equal(game.completed,0);
});
test('nonfinite scores are rejected and results remain fixed after finish',()=>{
 const game=new MemeGame();game.start(0);game.observe(prediction(game.target,NaN),0);assert.equal(game.match,0);
 game.finish(10,'stopped');const message=game.message;game.finish(20);game.observe(prediction(game.target),30);
 assert.equal(game.reason,'stopped');assert.equal(game.message,message);assert.equal(game.completed,0);assert.equal(messages.length,20);
});
