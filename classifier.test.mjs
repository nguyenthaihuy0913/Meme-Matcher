import {test} from 'node:test';
import assert from 'node:assert/strict';
import {extract,classify,createPredictionTracker} from './public/classifier.js';
test('missing detections do not force a meme',()=>{assert.equal(classify(extract({}, {landmarks:[]}, {})).label,0);});
test('clear shaka and thumb gestures stay distinct',()=>{const base=extract({}, {landmarks:[]}, {});assert.equal(classify({...base,hands:1,shaka:1}).label,7);assert.equal(classify({...base,hands:1,thumb:1,jaw:1}).label,11);});
test('ambiguous gestures are rejected',()=>{const base=extract({}, {landmarks:[]}, {});assert.equal(classify({...base,face:1,hands:1,indexMouth:1,chin:1,turn:1}).label,0);});
test('two hands near head with open mouth match surprise',()=>{const base=extract({}, {landmarks:[]}, {});assert.equal(classify({...base,pose:1,oneHead:1,twoHead:1,jaw:1}).label,6);});

test('hand gestures require actual hand detection',()=>{
 const base=extract({}, {landmarks:[]}, {});
 for(const gesture of [{shaka:1},{thumb:1,jaw:1},{indexMouth:1}]) {
  const result=classify({...base,...gesture});
  assert.equal(result.label,0);
 }
});
test('tracker requires sustained evidence and clears missing detections',()=>{
 const tracker=createPredictionTracker(), base=extract({}, {landmarks:[]}, {});
 const shaka={...base,hands:1,shaka:1};
 assert.equal(tracker.update(shaka,0).label,0);
 assert.equal(tracker.update(shaka,100).label,0);
 assert.equal(tracker.update(shaka,250).label,7);
 assert.equal(tracker.update(base,350).label,0);
 assert.equal(tracker.update(shaka,450).label,0);
 tracker.reset();
 assert.equal(tracker.update(shaka,700).label,0);
});
test('tracker does not carry a gesture through a long camera gap',()=>{
 const tracker=createPredictionTracker(), f={...extract({}, {landmarks:[]}, {}),hands:1,shaka:1};
 tracker.update(f,0);assert.equal(tracker.update(f,250).label,7);
 assert.equal(tracker.update(f,1500).label,0);
});

test('a face alone cannot imply pointing an arm sideways',()=>{
 const f={...extract({}, {landmarks:[]}, {}),face:1,turn:0.548};
 const result=classify(f);
 assert.ok(!result.top.some(x=>[4,6,9,10].includes(x.id)&&x.score>0));
});
