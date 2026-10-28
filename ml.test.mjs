import {test} from 'node:test';
import assert from 'node:assert/strict';
import {predictModel} from './public/ml.js';
import {extract} from './public/features.js';

test('portable model applies standardization and arbitrary MLP layers',()=>{
  const model={model_type:'mlp',features:['x'],classes:[0,7],scaler:{mean:[10],scale:[2]},
    coefs:[[[1,-1]],[[1],[-1]],[[0,1]]],intercepts:[[0,0],[0],[0,0]]};
  const probs=predictModel(model,{x:14});
  assert.ok(Math.abs(probs[7]-1/(1+Math.exp(-2)))<1e-12);
  assert.equal(probs[1],0);
});
test('forest uses node thresholds and maps sparse class IDs',()=>{
  const model={model_type:'forest',features:['x'],classes:[0,11],trees:[{
    left:[1,-1,-1],right:[2,-1,-1],feature:[0,-2,-2],threshold:[.5,-2,-2],value:[[1,1],[4,0],[0,2]]}]};
  assert.equal(predictModel(model,{x:.5})[0],1);
  assert.equal(predictModel(model,{x:.6})[11],1);
});
test('logistic regression remains numerically stable for extreme logits',()=>{
  const probs=predictModel({model_type:'logistic_regression',features:['x'],classes:[0,3],coef:[[1000],[-1000]],intercept:[0,0]},{x:1});
  assert.equal(probs[0],1);assert.equal(probs[3],0);
});
test('missing tracking yields finite zeros for continuous geometry',()=>{
  const f=extract({}, {}, {});
  assert.ok(Object.values(f).every(Number.isFinite));
  assert.equal(f.armsVisible,0);assert.equal(f.indexExtension,0);assert.equal(f.headPitch,0);
});
test('invisible pose joints never create arm geometry',()=>{
  const points=Array.from({length:33},(_,i)=>({x:(i%5)/5,y:(i%7)/7,z:0,visibility:0}));
  const f=extract({}, {}, {landmarks:[points]});
  assert.equal(f.armsVisible,0);assert.equal(f.flex,0);assert.equal(f.oneHead,0);
});
test('finger extension features tolerate mirrored camera input',()=>{
  const points=Array.from({length:21},(_,i)=>({x:(i%5)/5,y:Math.floor(i/5)/5,z:0}));
  const a=extract({}, {landmarks:[points]}, {}),b=extract({}, {landmarks:[points.map(p=>({...p,x:1-p.x}))]}, {});
  for(const key of Object.keys(a).filter(k=>k.includes('Extension')))assert.ok(Math.abs(a[key]-b[key])<1e-12,key);
});
test('signed head pitch distinguishes looking down from looking up',()=>{
  const make=pitch=>{
    const data=[1,0,0,0,0,Math.cos(pitch),Math.sin(pitch),0,0,-Math.sin(pitch),Math.cos(pitch),0,0,0,0,1];
    return extract({facialTransformationMatrixes:[{data}]},{},{});
  };
  assert.equal(make(.45).up,0);
  assert.equal(make(-.45).up,1);
});
