$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
npm ci
if ($LASTEXITCODE -ne 0) { throw 'Không cài được MediaPipe.' }
New-Item -ItemType Directory -Force public/models | Out-Null
$models = @{face='face_landmarker/face_landmarker/float16/1/face_landmarker.task'; hand='hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'; pose='pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'}
foreach ($entry in $models.GetEnumerator()) {
    Invoke-WebRequest ('https://storage.googleapis.com/mediapipe-models/' + $entry.Value) -OutFile ('public/models/' + $entry.Key + '.task')
}
Write-Output 'Sẵn sàng. Chạy npm start rồi mở http://localhost:5173.'
