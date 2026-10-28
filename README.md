# Meme Matcher · 9.5AI Club

Game showcase chủ đề ngân hà cho 9.5AI Club, FPT University campus Hồ Chí Minh.

## Chơi showcase

Chạy `npm start` và mở http://localhost:5173 bằng Chrome hoặc Edge. Bấm **Bắt đầu thử thách**, cấp quyền camera, vào vị trí trong 3 giây đếm ngược.

- Mỗi lượt chọn 5 meme khác nhau từ 13 meme hiện có. Thời gian cho mỗi meme: **20 giây**.
- Nhái meme bên trái. Camera bên phải có skeleton mặt/tay/thân; tọa độ z điều chỉnh độ dày khớp và đường nối để gợi chiều sâu trên lớp phủ 2D, không phải mô hình 3D xoay tự do.
- Điểm khớp (%) là điểm của bộ phân loại dành cho **meme mục tiêu**, không phải xác suất hay độ giống ảnh theo pixel. Đạt ít nhất 62%, đúng nhãn mục tiêu và giữ liên tục 650 ms để qua meme. Ngưỡng nằm trong `public/game.js`.
- Vượt cả 5 meme là thắng. Hết giờ, dừng lượt hoặc camera ngắt sẽ hiện kết quả cùng 1 trong **20 thông điệp** do ban tổ chức cung cấp.
- Có nút toàn màn hình, chơi lại, bật/tắt camera. Tab bị ẩn sẽ kết thúc lượt để tránh giữ dự đoán cũ.
- Game xử lý tại máy, không ghi hình và không gửi/lưu lượt chơi. Màn thu dữ liệu riêng vẫn ở http://localhost:5173/lab.html.
- Kiểm tra luật chơi và classifier: `npm test`. Cần thử webcam với người thật tại gian hàng để chỉnh ngưỡng phù hợp ánh sáng, khoảng cách và tốc độ máy.

## Meme Lab: thu dữ liệu và training


Ứng dụng webcam local: MediaPipe nhận diện mặt/tay/thân, kết hợp Extra Trees đã train và luật để phân loại 13 meme, thu lượt ghi 3 giây có nhãn do người dùng xác nhận. Model hiện train từ bộ ảnh trong dự án; chưa tự học từ các lượt webcam mới. Điểm kết hợp không phải xác suất.

## Chạy trên Windows

Đã tải thư viện và model vào máy. Trong thư mục project:

```powershell
npm start
```

Mở http://localhost:5173/lab.html bằng Chrome hoặc Edge để thu dữ liệu. Bấm **Bật camera** và cho phép camera. Sau khi cài thư viện và tải model, không cần Internet để chạy. Server chỉ nghe trên loopback. Bấm **Tắt** để giải phóng camera; Ctrl+C ở terminal để tắt server.

Nếu chuyển sang máy khác hoặc mất thư viện/model, chạy `powershell -ExecutionPolicy Bypass -File .\setup.ps1` một lần khi có Internet.

## Thu data

1. Nhập ID người tham gia, dùng cùng ID cho cùng người.
2. Tạo dáng, bấm **Ghi mẫu 3 giây**, giữ tư thế.
3. Chọn nhãn thật (click ảnh hoặc dùng dropdown). Nhãn mặc định là “Không khớp”; app không tự chọn nhãn theo dự đoán.
4. Bấm **Xác nhận nhãn & lưu**, hoặc bỏ lượt để ghi lại.

Mỗi lượt tạo một file UUID trong `data/`. File gồm ID người, ID lượt, nhãn xác nhận, kích thước camera, chuỗi landmarks và visibility nếu model cung cấp, handedness, blendshapes, ma trận đầu, đặc trưng và top 3 dự đoán. Không ghi video, ảnh hay âm thanh. Không tự train khi lưu. File lỗi vẫn giữ trong giao diện để thử lại; tải lại trang sẽ mất lượt chưa lưu.

Thu cả đứng bình thường, chuyển tư thế, chỉnh tóc và mẫu gần đúng nhưng sai, gán nhãn 0. Sau này tách train/test theo người và lượt; không chia ngẫu nhiên frame của cùng lượt.

## Quy ước và giới hạn

- Nhận diện một người, tối đa hai tay. Tư thế trái/phải đối xứng được chấp nhận. Webcam hiển thị lật gương nhưng đầu vào model không lật.
- Hình 1 quy ước ngón trỏ gần má. Lè lưỡi, phụ kiện, vóc dáng và đồ vật không chấm.
- Góc ngón và đầu là ước lượng; che tay, mặt nghiêng, ánh sáng kém và khớp bị che làm giảm độ đúng. Nếu top 1 thấp hoặc sát top 2 sẽ trả “Không khớp / chưa chắc”.
- Luật chưa được hiệu chỉnh bằng người thật. Cần kiểm thử webcam, đặc biệt hướng ngẩng đầu và các cặp suỵt/suy nghĩ. Không cam kết độ chính xác.
- Ba model chạy CPU trên luồng giao diện ở tối đa khoảng 10 lượt/giây, tốc độ thực tế tùy máy. Chuyển sang worker là bước tối ưu tiếp theo nếu máy chậm.

## Kiểm tra

```powershell
npm test
```

Các test kiểm tra không ép nhãn khi mất detection, phân biệt ngón cái/shaka, loại dự đoán mơ hồ và nhận tư thế ôm đầu. Chúng không thay thế đánh giá trên người thật.

Model chính thức: https://ai.google.dev/edge/mediapipe/solutions/vision


## Model v2: đặc trưng ngón tay liên tục

- `public/features.js` là bộ trích đặc trưng duy nhất cho webcam lẫn training: 46 đặc trưng gồm độ duỗi từng ngón, góc đầu có dấu, góc khuỷu, vị trí cổ tay và độ gần mặt/thân. Cache landmark dùng ma trận theo đúng thứ tự cột của trình duyệt.
- `python train_pipeline.py` tạo cache khi cần, so sánh 5 cấu hình, rồi ghi model ứng viên vào `scratch/candidate_weights.json`. Lệnh này chưa thay model đang chạy. `--evaluate-only` chỉ đánh giá; `--rebuild` tạo lại landmark từ ảnh.
- `node scripts/install_model.mjs` kiểm tra ứng viên không kém baseline theo macro recall, đối chiếu inference JS/Python (sai số tối đa 1e-6), sao lưu model hiện tại rồi cài model mới.
- Training dùng toàn bộ ảnh khác nhau, cân bằng bằng augmentation **bên trong từng training fold**. Ảnh trùng byte được gộp theo SHA-256. Mẫu webcam có xác nhận nhãn được nhóm theo người và nhãn, chỉ thêm một vector trung vị mỗi nhóm vào lượt train cuối; không đưa frame cùng người vào đánh giá ảnh.
- Báo cáo `scratch/evaluation_v2.json`: 471 ảnh dùng được, 2 fold; MLP 20 đặc trưng macro recall 75,42%, Extra Trees 83,35%. Accuracy theo số ảnh là 94,69% và 95,97%. Đây là kết quả chọn model; chưa có test độc lập theo người, ảnh gần trùng có thể vẫn nằm ở hai fold.
- Các lớp 2, 11, 12 có rất ít ảnh và recall cross-validation lần này lần lượt 0%, 0%, 100%. Cần thêm ảnh nhiều người, nhiều góc tay và mẫu âm thật. Không suy ra độ đúng webcam từ điểm training hay điểm trên ảnh đã học.
- Model hiện tại: Extra Trees 120 cây. `public/ml.js` vẫn hỗ trợ MLP nhiều lớp, scaler và Logistic Regression. Ngưỡng và thời gian giữ dáng trong game được giữ nguyên.
- Camera ưu tiên 960×720 khi thiết bị hỗ trợ; nhắc đưa tay vào khung nếu động tác cần ngón tay nhưng tracking chưa thấy tay.
