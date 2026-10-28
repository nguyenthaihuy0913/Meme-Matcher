"""Train portable classifiers using the exact same features as the webcam.

python train_pipeline.py --evaluate-only
python train_pipeline.py                  # evaluate, then export winning model
python train_pipeline.py --rebuild        # force landmark cache refresh
"""
import argparse
import hashlib
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scripts.prepare_dataset import prepare, ROOT

LEGACY_FEATURES = ['face','hands','pose','jaw','smile','up','turn','indexMouth','indexCheek','chin','shaka','thumb','index','fists','tipsTogether','oneHead','twoHead','flex','straight','chest']
DISCRETE = {'face','hands','pose','fists','index','shaka','thumb','flex','straight','armsVisible'}


def negatives(names, seed, count=100):
    """Training prior only. These do NOT substitute for real negative examples."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(count):
        f = dict.fromkeys(names, 0.0)
        f.update(face=1, pose=int(i % 3 != 0), hands=0,
                 jaw=float(rng.uniform(0,.18)), smile=float(rng.uniform(0,.65)),
                 turn=float(rng.uniform(0,.65)), up=float(rng.uniform(0,.15)),
                 headPitch=float(rng.uniform(-.02,.02)), headYaw=float(rng.uniform(0,.14)),
                 headRoll=float(rng.uniform(0,.08)), noseChinRatio=float(rng.uniform(.35,.44)))
        if f['pose']:
            f.update(armsVisible=1,bendMax=.9,bendMin=.8,wristHeightMax=-.7,wristHeightMin=-1.2,
                     elbowHeightMax=-.4,elbowHeightMin=-.6,wristSideMax=.2,wristSideMin=.1)
        if i % 10 == 0:
            f = dict.fromkeys(names, 0.0)
        rows.append([f.get(k,0) for k in names])
    return np.asarray(rows)


def balanced_training(x, y, names, seed):
    """Use every source image, and augment only the current training fold."""
    rng = np.random.default_rng(seed)
    xs, ys = [x], [y]
    for label in np.unique(y):
        group = x[y == label]
        need = max(0, 65-len(group))
        if need:
            aug = group[rng.integers(0,len(group),need)].copy()
            for i, key in enumerate(names):
                if key in DISCRETE:
                    continue
                # Preserve zeros (missing landmarks); never clamp count features to 1.
                mask = np.abs(aug[:,i]) > 1e-6
                aug[mask,i] += rng.normal(0,.018,int(mask.sum()))
                low, high = (-2,2) if 'Height' in key else (-1,1) if key == 'headPitch' else (0,1)
                aug[:,i] = np.clip(aug[:,i],low,high)
            xs.append(aug);ys.append(np.full(need,label))
    xs.append(negatives(names,seed));ys.append(np.zeros(100,dtype=int))
    return np.concatenate(xs), np.concatenate(ys)


def export_model(fitted, names, version):
    result = {'featureVersion':version,'features':names,'classes':None}
    if hasattr(fitted,'named_steps'):
        scaler = fitted.steps[0][1]
        result['scaler']={'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist()}
        fitted = fitted.steps[-1][1]
    result['classes'] = fitted.classes_.astype(int).tolist()
    if isinstance(fitted,MLPClassifier):
        result.update(model_type='mlp',coefs=[c.tolist() for c in fitted.coefs_],intercepts=[b.tolist() for b in fitted.intercepts_])
    else:
        result.update(model_type='forest',trees=[{
            'left':t.tree_.children_left.tolist(),'right':t.tree_.children_right.tolist(),
            'feature':t.tree_.feature.tolist(),'threshold':t.tree_.threshold.tolist(),
            'value':t.tree_.value[:,0,:].tolist()} for t in fitted.estimators_])
    return result


def run():
    parser=argparse.ArgumentParser()
    parser.add_argument('--evaluate-only',action='store_true')
    parser.add_argument('--rebuild',action='store_true')
    args=parser.parse_args()
    data=prepare(args.rebuild)
    names=list(data['rows'][0]['features'])
    original=[]; seen={}; unusable=[]; webcam=[]
    for row in data['rows']:
        if row['source']=='webcam':
            webcam.append(row);continue
        if row['group'] in seen:
            if seen[row['group']]!=row['label']:
                raise ValueError('Identical image has contradictory labels: '+row['file'])
            continue
        seen[row['group']]=row['label']
        if not any(row['features'][k] for k in ('face','hands','pose')):
            unusable.append(row['file']);continue
        original.append(row)
    x=np.asarray([[r['features'][k] for k in names] for r in original]);y=np.asarray([r['label'] for r in original])
    groups=np.asarray([r['group'] for r in original])
    counts=Counter(y.tolist());folds=min(3,min(counts.values()))
    if len(counts)!=13 or folds<2:raise ValueError('Need >=2 distinct usable images for each meme.')
    print(f'{len(x)} unique usable images; {len(names)} features; {folds} folds; {len(webcam)} webcam frames',flush=True)
    print('Per class:',dict(sorted(counts.items())),flush=True)
    specs={
        'legacy_mlp_20':(MLPClassifier(hidden_layer_sizes=(32,),max_iter=1000,alpha=.01,random_state=42),LEGACY_FEATURES),
        'scaled_mlp_32':(make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(32,),solver='lbfgs',max_iter=1200,alpha=1.,random_state=42)),names),
        'scaled_mlp_64':(make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(64,),solver='lbfgs',max_iter=1200,alpha=1.,random_state=42)),names),
        'extra_trees':(ExtraTreesClassifier(n_estimators=120,max_depth=14,max_features=.8,class_weight='balanced',random_state=42,n_jobs=-1),names),
        'random_forest':(RandomForestClassifier(n_estimators=120,max_depth=14,max_features=.8,class_weight='balanced',random_state=42,n_jobs=-1),names)
    }
    splits=list(StratifiedGroupKFold(n_splits=folds,shuffle=True,random_state=42).split(x,y,groups))
    # Stratification with rare groups is approximate; validate class coverage explicitly.
    if any(len(set(y[train]))<13 for train,_ in splits):
        raise ValueError('A fold is missing a meme class; collect more independent images.')
    results={};oof={}
    for name,(estimator,keys) in specs.items():
        indices=[names.index(k) for k in keys];xx=x[:,indices]
        probs=np.zeros((len(x),14));negative_fp=[];convergence=[]
        for fold,(train,test) in enumerate(splits):
            tx,ty=balanced_training(xx[train],y[train],keys,42+fold)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always',ConvergenceWarning)
                model=clone(estimator).fit(tx,ty)
            convergence.extend(str(w.message) for w in caught if issubclass(w.category,ConvergenceWarning))
            p=model.predict_proba(xx[test]);probs[np.ix_(test,model.classes_.astype(int))]=p
            negative_fp.append(float(np.mean(model.predict(negatives(keys,9000+fold))!=0)))
        pred=probs.argmax(axis=1)
        accuracy=float(accuracy_score(y,pred));balanced=float(recall_score(y,pred,labels=list(range(1,14)),average='macro',zero_division=0))
        results[name]={'accuracy':accuracy,'balanced_accuracy':balanced,
                       'per_class_recall':dict(zip(range(1,14),recall_score(y,pred,labels=list(range(1,14)),average=None,zero_division=0).tolist())),
                       'confusion_matrix':confusion_matrix(y,pred,labels=list(range(14))).tolist(),
                       'synthetic_negative_false_positive_rate':float(np.mean(negative_fp)),
                       'convergence_warnings':convergence}
        oof[name]=probs
        print(f'{name}: accuracy={accuracy:.4f}, macro recall={balanced:.4f}, synthetic negative FP={np.mean(negative_fp):.4f}',flush=True)
    # Macro recall weights every meme equally. Accuracy breaks ties.
    winner=max(results,key=lambda k:(results[k]['balanced_accuracy'],results[k]['accuracy']))
    report={'featureVersion':data['featureVersion'],'images':len(x),'features':len(names),'folds':folds,
            'class_counts':dict(sorted(counts.items())),'excluded_no_tracking':unusable,
            'evaluation':'Grouped by exact image hash; fold-local augmentation; label 0 synthetic in training only; image validation labels 1-13',
            'limitations':['Model-selection cross-validation, not an independent final test.',
                           'People and capture sessions are unknown in image folders; near-duplicate frames and same people may cross folds.',
                           'Class 0 has no real negative examples; synthetic negative scores do not measure real-world rejection.',
                           'Some classes have only 2-4 independent images. Webcam accuracy is not established.'],
            'results':results,'selected_model':winner,'webcam_frames_available':len(webcam)}
    (ROOT/'scratch/evaluation_v2.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Selected:',winner,flush=True)
    if args.evaluate_only:return
    model,keys=specs[winner];indices=[names.index(k) for k in keys]
    final_x,final_y=x[:,indices],y
    # Cap each confirmed webcam session/person to one representative vector: a long
    # recording cannot outweigh an entire image class. All its frames remain together.
    by_group={}
    for row in webcam:by_group.setdefault((row['group'],row['label']),[]).append(row)
    for (_,label),rows in by_group.items():
        vector=np.median([[row['features'][k] for k in keys] for row in rows],axis=0)
        final_x=np.vstack([final_x,vector]);final_y=np.append(final_y,label)
    tx,ty=balanced_training(final_x,final_y,keys,42)
    fitted=clone(model).fit(tx,ty)
    weights=export_model(fitted,keys,data['featureVersion'])
    weights['training']={'selected_model':winner,'images':len(x),'webcam_groups':len(by_group),'evaluation':'scratch/evaluation_v2.json'}
    # Save candidate before installation. Browser/Python inference parity is verified next.
    (ROOT/'scratch/candidate_weights.json').write_text(json.dumps(weights),encoding='utf-8')
    (ROOT/'scratch/inference_parity.json').write_text(json.dumps({'rows':[dict(zip(keys,row)) for row in final_x[:50]],
                                                               'probabilities':fitted.predict_proba(final_x[:50]).tolist(),
                                                               'classes':fitted.classes_.astype(int).tolist()}),encoding='utf-8')
    print('Candidate saved to scratch/candidate_weights.json',flush=True)


if __name__=='__main__':
    run()
