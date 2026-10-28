import os
import sys
import math
import json
import random
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score

FEATURE_NAMES = [
    'face', 'hands', 'pose', 'jaw', 'smile', 'up', 'turn',
    'indexMouth', 'indexCheek', 'chin', 'shaka', 'thumb', 'index',
    'fists', 'tipsTogether', 'oneHead', 'twoHead', 'flex', 'straight', 'chest'
]

def clamp(x):
    return max(0.0, min(1.0, float(x)))

def dist(a, b):
    if a and b:
        return math.hypot(a['x'] - b['x'], a['y'] - b['y'])
    return 99.0

def angle(a, b, c):
    u = [a['x'] - b['x'], a['y'] - b['y']]
    v = [c['x'] - b['x'], c['y'] - b['y']]
    dot = u[0]*v[0] + u[1]*v[1]
    mag = math.hypot(*u) * math.hypot(*v) or 1.0
    return math.acos(max(-1.0, min(1.0, dot / mag))) * 180.0 / math.pi

def extract_features(face_res, hand_res, pose_res, width=640, height=480):
    aspect = width / height
    def correct(p):
        return {'x': p.x * aspect, 'y': p.y, 'z': getattr(p, 'z', 0), 'visibility': getattr(p, 'visibility', 1.0)} if p else None

    f = [correct(p) for p in face_res.face_landmarks[0]] if face_res.face_landmarks else None
    p = [correct(p) for p in pose_res.pose_landmarks[0]] if pose_res.pose_landmarks else None
    blends = {c.category_name: c.score for c in face_res.face_blendshapes[0]} if face_res.face_blendshapes else {}

    scale = dist(f[234], f[454]) if f else (dist(p[11], p[12]) if p else 0.25)
    def near(a, b, r=0.7):
        return clamp(1.0 - dist(a, b) / (max(scale, 0.03) * r))

    up, turn = 0.0, 0.0
    if face_res.facial_transformation_matrixes:
        mat = np.array(face_res.facial_transformation_matrixes[0]).flatten()
        pitch = max(math.atan2(mat[6], mat[10]), math.atan2(-mat[6], mat[10]))
        up = clamp(pitch / 0.40)
        turn = clamp(abs(math.atan2(mat[8], mat[10])) / 0.65)
    
    # Landmark fallback for head pitch (up)
    if f:
        d_chin = f[152]['y'] - f[1]['y']
        d_face = dist(f[10], f[152]) or 1.0
        up_pts = clamp((d_chin / d_face - 0.42) / 0.22)
        up = max(up, up_pts)

    hands_list = []
    if hand_res.hand_landmarks:
        for raw in hand_res.hand_landmarks:
            l = [correct(pt) for pt in raw]
            extended = [angle(l[1], l[2], l[4]) > 145] + [angle(l[i], l[i+1], l[i+3]) > 155 for i in (5, 9, 13, 17)]
            index = extended[1] and not extended[2] and not extended[3] and not extended[4]
            hands_list.append({
                'extended': extended,
                'index': float(index),
                'fist': float(not any(extended[1:])),
                'shaka': float(extended[0] and extended[4] and not extended[1] and not extended[2] and not extended[3]),
                'thumb': float(extended[0] and not any(extended[1:])),
                'mouth': near(l[8], f[13]) if f else 0.0,
                'chin': near(l[8], f[152], 1.0) if f else 0.0,
                'cheek': max(near(l[8], f[234]), near(l[8], f[454])) if f else 0.0,
                'vertical': clamp(1.0 - abs(l[8]['x'] - l[5]['x']) / (dist(l[8], l[5]) or 1.0)),
                'tip': l[8],
                'wrist': l[0]
            })

    def visible(i, min_v=0.15):
        return p and i < len(p) and p[i] and (p[i].get('visibility', 1.0) or 1.0) > min_v

    arms = []
    if p:
        for s, e, w in [[11, 13, 15], [12, 14, 16]]:
            if not (visible(s, 0.20) and visible(e, 0.20) and visible(w, 0.10)):
                arms.append({'head': 0.0, 'flex': 0.0, 'straight': 0.0, 'chest': 0.0})
            else:
                a = angle(p[s], p[e], p[w])
                head = max(near(p[w], f[234], 2.2), near(p[w], f[454], 2.2), near(p[w], f[10], 2.0)) if f else near(p[w], p[0], 2.2)
                arms.append({
                    'head': head,
                    'flex': float(40 < a < 120 and p[e]['y'] < p[s]['y'] + 0.13 and abs(p[e]['x'] - p[s]['x']) > scale * 0.35),
                    'straight': float(a > 145 and abs(p[w]['x'] - p[s]['x']) > scale),
                    'chest': near(p[w], {'x': (p[11]['x'] + p[12]['x']) / 2, 'y': (p[11]['y'] + p[12]['y']) / 2 + 0.15}, 1.4)
                })

    def max_h(fn):
        return max([0.0] + [fn(x) for x in hands_list])

    jaw = clamp((blends.get('jawOpen', 0.0) or 0.0) * 2)
    smile = max(blends.get('mouthSmileLeft', 0.0) or 0.0, blends.get('mouthSmileRight', 0.0) or 0.0)

    tipsTogether = 0.0
    if len(hands_list) == 2:
        tipsTogether = hands_list[0]['index'] * hands_list[1]['index'] * near(hands_list[0]['tip'], hands_list[1]['tip'], 0.8)

    handNearHead = 0.0
    if f:
        for h in hands_list:
            for pt in [h['tip'], h['wrist']]:
                h_prox = max(near(pt, f[234], 2.2), near(pt, f[454], 2.2), near(pt, f[10], 2.0))
                if h_prox > handNearHead:
                    handNearHead = h_prox

    arm_heads = [x['head'] for x in arms if x['head'] > 0.05]
    oneHead = max([0.0] + [x['head'] for x in arms] + [handNearHead])
    twoHead = min(arm_heads) if len(arm_heads) >= 2 else 0.0
    flex = sum(x['flex'] for x in arms) / 2.0 if arms else 0.0
    straight = max([0.0] + [x['straight'] for x in arms])
    chest = max([0.0] + [x['chest'] for x in arms])

    feats = [
        float(bool(f)),
        float(len(hands_list)),
        float(bool(p)),
        jaw,
        smile,
        up,
        turn,
        max_h(lambda x: x['index'] * x['mouth'] * x['vertical']),
        max_h(lambda x: x['index'] * x['cheek']),
        max_h(lambda x: x['chin']),
        max_h(lambda x: x['shaka']),
        max_h(lambda x: x['thumb']),
        max_h(lambda x: x['index']),
        float(sum(1 for x in hands_list if x['fist'])),
        tipsTogether,
        oneHead,
        twoHead,
        flex,
        straight,
        chest
    ]
    return feats

def generate_normal_samples(n=70):
    samples = []
    for _ in range(n):
        face = 1.0
        pose = 1.0
        hands = float(random.choice([0, 1]))
        jaw = clamp(random.gauss(0.03, 0.02))
        smile = clamp(random.gauss(0.05, 0.04))
        up = clamp(random.gauss(0.02, 0.02))
        turn = clamp(random.gauss(0.04, 0.03))
        indexMouth = 0.0
        indexCheek = 0.0
        chin = 0.0
        shaka = 0.0
        thumb = 0.0
        index = float(random.choice([0, 0, 0, 1]) if hands > 0 else 0)
        fists = 0.0
        tipsTogether = 0.0
        oneHead = clamp(random.gauss(0.02, 0.02))
        twoHead = 0.0
        flex = 0.0
        straight = 0.0
        chest = clamp(random.gauss(0.05, 0.04)) if hands > 0 else 0.0

        vec = [face, hands, pose, jaw, smile, up, turn,
               indexMouth, indexCheek, chin, shaka, thumb, index,
               fists, tipsTogether, oneHead, twoHead, flex, straight, chest]
        samples.append(vec)
    return samples

def augment_sample(vec):
    aug = list(vec)
    for i in range(3, len(aug)):
        if aug[i] > 0.02:
            aug[i] = clamp(aug[i] + random.gauss(0, 0.03))
    return aug

def main():
    cache_path = Path('scratch/features_cache.json')
    cached_data = {}

    if cache_path.exists():
        print(f"Đọc dữ liệu đặc trưng đã lưu cache từ {cache_path}...")
        with open(cache_path, 'r', encoding='utf-8') as fp:
            cached_data = json.load(fp)
    else:
        print("Khởi tạo MediaPipe Landmarkers...")
        f_opts = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path='public/models/face.task'),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1
        )
        h_opts = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path='public/models/hand.task'),
            num_hands=2
        )
        p_opts = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path='public/models/pose.task'),
            num_poses=1
        )

        face_landmarker = vision.FaceLandmarker.create_from_options(f_opts)
        hand_landmarker = vision.HandLandmarker.create_from_options(h_opts)
        pose_landmarker = vision.PoseLandmarker.create_from_options(p_opts)

        dataset_root = Path('data/data image meme/meme AI cv project')
        folders = sorted([d for d in os.listdir(dataset_root) if (dataset_root / d).is_dir()])

        print("\n--- Bắt đầu trích xuất đặc trưng từ ảnh người thật ---")
        for folder_name in folders:
            label = int(folder_name.split('_')[0])
            folder_path = dataset_root / folder_name
            files = [f for f in os.listdir(folder_path) if not f.startswith('meme_goc') and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            cached_data[label] = []

            for f in files:
                img_path = str(folder_path / f)
                try:
                    bgr = cv2.imdecode(np.frombuffer(open(img_path, 'rb').read(), np.uint8), cv2.IMREAD_COLOR)
                    if bgr is None:
                        continue
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                    face_res = face_landmarker.detect(mp_img)
                    hand_res = hand_landmarker.detect(mp_img)
                    pose_res = pose_landmarker.detect(mp_img)

                    features = extract_features(face_res, hand_res, pose_res, mp_img.width, mp_img.height)
                    if features[0] > 0 or features[2] > 0:
                        cached_data[label].append({'file': f, 'features': features})
                except Exception:
                    pass

            print(f"Nhãn {label:2d} ({folder_name}): {len(cached_data[label])}/{len(files)} ảnh thành công")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as fp:
            json.dump(cached_data, fp)

    # Keep evaluation on original images. Augmented copies belong only in final
    # training; splitting them across folds would leak the source image.
    eval_x, eval_y = [], []
    for label in range(1, 14):
        items = cached_data.get(str(label), []) or cached_data.get(label, [])
        for item in items:
            eval_x.append(item['features'])
            eval_y.append(label)
    class_counts = [eval_y.count(label) for label in range(1, 14)]
    if min(class_counts) < 2:
        raise ValueError('Need at least two original images for each of the 13 classes.')
    eval_x, eval_y = np.asarray(eval_x), np.asarray(eval_y)

    # Chuẩn bị dữ liệu cân bằng
    random.seed(42)
    np.random.seed(42)

    X_raw = []
    y_raw = []

    # Nhãn 0: Tư thế bình thường
    print("\nTạo 70 mẫu đặc trưng cho Nhãn 0 (Tư thế bình thường / không khớp)...")
    for s in generate_normal_samples(70):
        X_raw.append(s)
        y_raw.append(0)

    TARGET_PER_CLASS = 45

    for label in range(1, 14):
        items = cached_data.get(str(label), []) or cached_data.get(label, [])
        feats = [item['features'] for item in items]

        if len(feats) > TARGET_PER_CLASS:
            chosen = random.sample(feats, TARGET_PER_CLASS)
        else:
            chosen = list(feats)
            if len(chosen) > 0:
                need = TARGET_PER_CLASS - len(chosen)
                for _ in range(need):
                    base = random.choice(feats)
                    chosen.append(augment_sample(base))

        for s in chosen:
            X_raw.append(s)
            y_raw.append(label)

    X = np.array(X_raw)
    y = np.array(y_raw)
    unique_labels = sorted(list(set(y)))

    print(f"\nTổng dataset huấn luyện: {len(X)} mẫu, {X.shape[1]} đặc trưng, {len(unique_labels)} nhãn.")

    cv = StratifiedKFold(n_splits=min(5, min(class_counts)), shuffle=True, random_state=42)

    lr = LogisticRegression(max_iter=1000, C=2.0, random_state=42)
    lr_scores = cross_val_score(lr, eval_x, eval_y, cv=cv, scoring='accuracy')
    print(f"Logistic Regression original-image Cross-Val Accuracy: {lr_scores.mean()*100:.2f}% (+/- {lr_scores.std()*100:.2f}%)")

    mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=1000, alpha=0.01, random_state=42)
    mlp_scores = cross_val_score(mlp, eval_x, eval_y, cv=cv, scoring='accuracy')
    print(f"MLP (32 hidden units) original-image Cross-Val Accuracy: {mlp_scores.mean()*100:.2f}% (+/- {mlp_scores.std()*100:.2f}%)")

    best_model_name = "MLP" if mlp_scores.mean() >= lr_scores.mean() else "Logistic Regression"
    print(f"\n-> Chọn mô hình: {best_model_name}")

    report = {
        'evaluation': 'stratified original-image folds; no augmented images; labels 1-13 only',
        'limitation': 'People are not identified in image folders; this is not an unseen-person evaluation. Class 0 is synthetic and excluded from validation.',
        'folds': cv.n_splits,
        'original_images': len(eval_y),
        'class_counts': dict(zip(range(1, 14), class_counts)),
        'logistic_regression_accuracy': float(lr_scores.mean()),
        'mlp_accuracy': float(mlp_scores.mean()),
        'selected_model': best_model_name
    }
    Path('scratch').mkdir(exist_ok=True)
    Path('scratch/evaluation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if '--evaluate-only' in sys.argv:
        print('Evaluation saved to scratch/evaluation.json; existing weights preserved.')
        return

    if best_model_name == "Logistic Regression":
        final_model = lr.fit(X, y)
        y_pred = final_model.predict(X)
        print("\nBáo cáo chi tiết trên tập dữ liệu cân bằng:")
        print(classification_report(y, y_pred, digits=3))

        weights_data = {
            "model_type": "logistic_regression",
            "features": FEATURE_NAMES,
            "classes": final_model.classes_.tolist(),
            "coef": final_model.coef_.tolist(),
            "intercept": final_model.intercept_.tolist()
        }
    else:
        final_model = mlp.fit(X, y)
        y_pred = final_model.predict(X)
        print("\nBáo cáo chi tiết trên tập dữ liệu cân bằng:")
        print(classification_report(y, y_pred, digits=3))

        weights_data = {
            "model_type": "mlp",
            "features": FEATURE_NAMES,
            "classes": final_model.classes_.tolist(),
            "coefs": [c.tolist() for c in final_model.coefs_],
            "intercepts": [i.tolist() for i in final_model.intercepts_]
        }

    # Đánh giá lại trực tiếp trên toàn bộ ảnh gốc của người dùng
    print("\n--- Đánh giá độ chính xác trên toàn bộ ảnh gốc người dùng ---")
    for label in range(1, 14):
        items = cached_data.get(str(label), []) or cached_data.get(label, [])
        feats = [item['features'] for item in items]
        if not feats:
            continue
        preds = final_model.predict(feats)
        correct = sum(1 for p in preds if p == label)
        acc = correct / len(feats) * 100
        print(f"Nhãn {label:2d}: {correct:3d}/{len(feats):3d} đúng ({acc:5.1f}%)")

    out_json = Path('public/model_weights.json')
    with open(out_json, 'w', encoding='utf-8') as fp:
        json.dump(weights_data, fp, indent=2)

    out_js = Path('public/model_weights.js')
    with open(out_js, 'w', encoding='utf-8') as fp:
        fp.write('export const modelWeights = ' + json.dumps(weights_data) + ';\n')

    print(f"\n=> Đã xuất weights thành công ra: {out_json} và {out_js}")

if __name__ == '__main__':
    main()
