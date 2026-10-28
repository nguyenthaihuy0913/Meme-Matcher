// Portable inference for the models exported by train_pipeline.py.
export function predictModel(model, features) {
  if (!model) return null;
  let x = model.features.map(k => Number.isFinite(features[k]) ? features[k] : 0);
  if (model.scaler) x = x.map((v,i)=>(v-model.scaler.mean[i])/model.scaler.scale[i]);
  let probabilities;
  if (model.model_type === 'forest') {
    probabilities=Array(model.classes.length).fill(0);
    for(const tree of model.trees) {
      let node=0;
      // sklearn tree prediction converts inputs to float32 before comparisons.
      while(tree.left[node]!==-1)node=Math.fround(x[tree.feature[node]])<=tree.threshold[node]?tree.left[node]:tree.right[node];
      const values=tree.value[node],total=values.reduce((a,b)=>a+b,0)||1;
      for(let k=0;k<values.length;k++)probabilities[k]+=values[k]/total/model.trees.length;
    }
  } else {
    let logits;
    if (model.model_type === 'logistic_regression') {
      logits=model.coef.map((row,k)=>row.reduce((s,w,i)=>s+w*x[i],model.intercept[k]));
    } else if(model.model_type === 'mlp') {
      let layer=x;
      for(let level=0;level<model.coefs.length;level++) {
        layer=model.intercepts[level].map((bias,j)=>{
          const value=layer.reduce((s,v,i)=>s+v*model.coefs[level][i][j],bias);
          return level===model.coefs.length-1?value:Math.max(0,value);
        });
      }
      logits=layer;
    } else return null;
    const peak=Math.max(...logits),exp=logits.map(v=>Math.exp(v-peak)),sum=exp.reduce((a,b)=>a+b,0);
    probabilities=exp.map(v=>v/sum);
  }
  const result=Array(14).fill(0);
  model.classes.forEach((id,k)=>{result[id]=probabilities[k];});
  return result;
}
