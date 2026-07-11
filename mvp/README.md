# Ceramic Tile Defect Detection - MVP Demo

Đây là phiên bản MVP (Minimum Viable Product) cho hệ thống phát hiện lỗi gạch men sử dụng Computer Vision truyền thống. Ứng dụng sử dụng Streamlit để cung cấp một giao diện trực quan, cho phép bạn điều chỉnh và khảo sát các tham số của Pipeline xử lý ảnh theo thời gian thực.

## Yêu cầu hệ thống

- Python 3.7 trở lên
- Các thư viện Python cần thiết: `streamlit`, `opencv-python`, `numpy`

## Cài đặt

Bạn có thể cài đặt các thư viện yêu cầu thông qua `pip`. Mở terminal và chạy lệnh:

```bash
pip install streamlit opencv-python numpy
```

## Hướng dẫn chạy ứng dụng

**Bước 1: Tạo ảnh mẫu**

Trước khi chạy giao diện chính, bạn cần khởi tạo ảnh gạch men chứa các lỗi giả lập (vết nứt, lỗ kim). Hãy di chuyển vào thư mục `mvp` và chạy script tạo ảnh:

```bash
cd mvp
python generate_sample.py
```

Sau khi chạy thành công, một file ảnh mới tên là `sample_tile.jpg` sẽ xuất hiện trong cùng thư mục.

**Bước 2: Khởi chạy giao diện Streamlit**

Khi đã có ảnh mẫu, bạn chạy lệnh sau để khởi động ứng dụng:

```bash
streamlit run app.py
```

Trình duyệt web của bạn sẽ tự động mở lên với đường dẫn ứng dụng (thường là `http://localhost:8501`). Tại đây, bạn có thể dùng thanh công cụ bên trái (Sidebar) để tuỳ chỉnh các tham số:
1. Kích thước bộ lọc nhiễu (Median Filter Kernel Size).
2. Ngưỡng phát hiện biên (Canny Min/Max Value).
3. Kích thước kernel cho phép toán hình thái học (Morphology Closing).
4. Các quy tắc phân loại kích thước và tỷ lệ khung hình.

Mỗi khi thay đổi tham số, ứng dụng sẽ ngay lập tức chạy lại toàn bộ Pipeline và hiển thị kết quả cho bạn theo dõi.
