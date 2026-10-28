"""Cache MediaPipe landmarks, then use the browser's exact JS feature extractor."""
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parents[1]
IMAGE_ROOT = ROOT / 'data/data image meme/meme AI cv project'
RAW = ROOT / 'scratch/landmarks_v2.json'
FEATURES = ROOT / 'scratch/features_v2.json'


def points(sets):
    return [[{'x': p.x, 'y': p.y, 'z': p.z,
              'visibility': p.visibility if p.visibility is not None else 1.0}
             for p in group] for group in sets]


def prepare(rebuild=False):
    paths = sorted(p for p in IMAGE_ROOT.glob('*/*')
                   if p.suffix.lower() in ('.jpg', '.jpeg', '.png') and not p.name.startswith('meme_goc'))
    # Invalidate when inputs or extraction logic change, not just when a cache is missing.
    manifest = hashlib.sha256(json.dumps([(str(p.relative_to(ROOT)), p.stat().st_size, p.stat().st_mtime_ns)
                                         for p in paths]).encode()).hexdigest()
    meta_path = RAW.with_suffix('.meta.json')
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if rebuild or not RAW.exists() or meta.get('manifest') != manifest:
        rows, errors = [], []
        face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(ROOT/'public/models/face.task')),
            output_face_blendshapes=True, output_facial_transformation_matrixes=True))
        hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(ROOT/'public/models/hand.task')), num_hands=2))
        pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(ROOT/'public/models/pose.task'))))
        try:
            for index, path in enumerate(paths):
                try:
                    content = path.read_bytes()
                    bgr = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
                    if bgr is None:
                        raise ValueError('Cannot decode image')
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                    f, h, p = face.detect(image), hand.detect(image), pose.detect(image)
                    rows.append({
                        'metadata': {'label': int(path.parent.name.split('_')[0]),
                                     'file': str(path.relative_to(IMAGE_ROOT)),
                                     'group': hashlib.sha256(content).hexdigest(), 'source': 'image'},
                        'width': image.width, 'height': image.height,
                        'face': {'faceLandmarks': points(f.face_landmarks),
                                 'faceBlendshapes': [{'categories': [{'categoryName': c.category_name, 'score': c.score}
                                                                    for c in cats]} for cats in f.face_blendshapes],
                                 # NumPy matrices are row-major; browser MatrixData is column-major.
                                 'facialTransformationMatrixes': [{'data': np.asarray(m).flatten(order='F').tolist()}
                                                                  for m in f.facial_transformation_matrixes]},
                        'hand': {'landmarks': points(h.hand_landmarks)},
                        'pose': {'landmarks': points(p.pose_landmarks)}})
                except Exception as error:
                    errors.append({'file': str(path), 'error': str(error)})
                if index % 25 == 0:
                    print(f'Landmarks {index+1}/{len(paths)}', flush=True)
        finally:
            face.close(); hand.close(); pose.close()
        RAW.parent.mkdir(exist_ok=True)
        RAW.write_text(json.dumps(rows), encoding='utf-8')
        meta_path.write_text(json.dumps({'manifest': manifest, 'images': len(rows), 'errors': errors}, indent=2))
    # Webcam turns retain person/session grouping and are never treated as independent frames.
    rows = json.loads(RAW.read_text(encoding='utf-8'))
    for path in sorted((ROOT/'data').glob('*.json')):
        sample = json.loads(path.read_text(encoding='utf-8'))
        if not sample.get('labelConfirmed') or not isinstance(sample.get('label'), int):
            continue
        if not 0 <= sample['label'] <= 13:
            continue
        for index, frame in enumerate(sample.get('frames', [])):
            lm = frame.get('landmarks', {})
            rows.append({'metadata': {'label': sample['label'], 'source': 'webcam',
                                     'file': f'{path.name}:{index}', 'person': sample.get('person'),
                                     'group': 'webcam:'+sample.get('person', sample.get('session', path.stem))},
                         'width': sample.get('camera', {}).get('width', 640),
                         'height': sample.get('camera', {}).get('height', 480),
                         'face': {'faceLandmarks': lm.get('face', []), 'faceBlendshapes': frame.get('blendshapes', []),
                                  'facialTransformationMatrixes': [{'data': m} for m in frame.get('headMatrix', [])]},
                         'hand': {'landmarks': lm.get('hand', [])}, 'pose': {'landmarks': lm.get('pose', [])}})
    combined = ROOT/'scratch/training_landmarks.json'
    combined.write_text(json.dumps(rows), encoding='utf-8')
    subprocess.run(['node', 'scripts/extract_features.mjs', str(combined), str(FEATURES)], cwd=ROOT, check=True)
    return json.loads(FEATURES.read_text(encoding='utf-8'))


if __name__ == '__main__':
    prepare()
