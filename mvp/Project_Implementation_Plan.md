# Kế Hoạch Triển Khai Đồ Án Cuối Kỳ: Hệ Thống Phát Hiện Lỗi Gạch Men

*(Automated Ceramic Tiles Surface Defect Detection)*

Kế hoạch này được ánh xạ 1-1 với các yêu cầu chấm điểm trong file hướng dẫn đồ án (`Computer_Vision 25.md`).

## 1. Phát Biểu Mục Tiêu & Giả Thuyết (Phần Bắt Buộc)

- **Vấn đề:** Phát hiện và phân loại tự động các lỗi trên bề mặt gạch men trơn (như vết nứt lớn, nứt mảnh, lỗ kim) trên dây chuyền sản xuất để thay thế khâu kiểm tra thủ công.
- **Giả thuyết:**
  - *Giả thuyết 1:* Việc sử dụng bộ lọc hình thái học (Morphological Closing) kết hợp thuật toán tìm biên (Canny Edge) sẽ phân đoạn vết nứt chính xác hơn so với chỉ dùng phương pháp phân ngưỡng (Thresholding) đơn thuần.
  - *Giả thuyết 2:* Kích thước Kernel (Structuring Element) của phép Closing càng tăng thì khả năng lấp đầy đứt gãy của vết nứt càng tốt, nhưng nếu tăng vượt ngưỡng $7\times7$ sẽ làm dính các nhiễu rác lân cận vào vùng lỗi.
- **Tiêu chí thành công:** Hệ thống trích xuất thành công đặc trưng hình học, thuật toán ML phân loại đúng $\ge 85\%$ trên tập dữ liệu thử nghiệm, đáp ứng thời gian thực thi nhanh.

## 2. Thiết Kế Pipeline & Ràng Buộc Kỹ Thuật

Đồ án sử dụng 4 kỹ thuật (đáp ứng điều kiện *ít nhất 3 kỹ thuật, trong đó $\ge 2$ kỹ thuật từ Chương 3, 4, 5*):

1. **Tiền xử lý (Chương 2):** Chuyển Grayscale, dùng Lọc phi tuyến (Median Filter) để giảm nhiễu mà không làm nhòe cạnh viền.
2. **Phát hiện cạnh (Chương 3):** Dùng Canny q  (để so sánh với RIMLV của bài báo gốc).
3. **Phân đoạn ảnh (Chương 4):** Áp dụng Toán tử hình thái học (Dilation & Closing) để làm liền mạch các biên nứt lởm chởm.
4. **Nhận dạng ảnh (Chương 3 & 5):** Tìm đường viền (Contours) để rút trích Đặc trưng hình học (Diện tích, Tỷ lệ khung hình, Độ tròn). Huấn luyện Machine Learning (SVM hoặc KNN) để phân loại thành các class lỗi.

## 3. Khảo Sát Tham Số & Ảnh Trung Gian (Parameter Sweeps)

Pipeline sẽ được code trên **Jupyter Notebook** để in thẳng các bước ảnh trung gian (Grayscale $\rightarrow$ Lọc nhiễu $\rightarrow$ Viền nhị phân $\rightarrow$ Bounding box màu) phục vụ báo cáo.

Các khảo sát tham số bắt buộc phải thực hiện (chạy nghiệm với $\ge 3$ giá trị):

- **Tham số `ksize` của Median Filter:** Thử nghiệm $ksize = 3, 5, 7, 9$. (Mục tiêu: Cho thấy nếu lọc quá tay sẽ làm mờ đi vết nứt cực nhỏ).
- **Tham số `minVal` / `maxVal` của Canny:** Thử nghiệm 3 cặp ngưỡng Threshold để tìm ra cặp tối ưu nhất không bị lọt nhiễu nền.
- **Tham số `Kernel Shape` của Hình thái học:** Thử nghiệm Kernel Hình vuông ($5\times5$), Hình chữ nhật ($1\times7$), Hình Elip. (Mục tiêu: Đánh giá xem hình dáng cấu trúc nào gom vết nứt dọc/ngang tốt nhất).

## 4. Dữ Liệu (Dataset)

Vì đồ án nghiêm cấm dùng ảnh tổng hợp (synthetic) để báo cáo chính thức, chúng ta sẽ cần phải tải bộ dữ liệu bề mặt công nghiệp thực tế.

- **Đề xuất:** Sử dụng **Kolektor Surface Defect Dataset (KSDD)** hoặc **Magnetic Tile Defect Dataset (MT_Free)**, cả hai đều public và có cấu trúc lỗi nứt, lỗ kim trên bề mặt nền giống hệt gạch men.

## 5. Tổ Chức Báo Cáo & Phân Công (Template)

Dựa theo giới hạn số trang (6-15 trang), chia việc như sau:

1. **Thành viên A (Data & Tiền xử lý):** Phụ trách load dữ liệu, Code Chương 2, viết phần *Đặt vấn đề*, *Giả thuyết*, và *Công trình liên quan* (Reference).
2. **Thành viên B (Phân đoạn & Thực nghiệm):** Phụ trách Code Chương 3-4 (Canny, Morphology), thiết kế các ô Jupyter Notebook hiển thị song song ảnh *Khảo sát tham số*. Viết phần *Phương pháp*.
3. **Thành viên C (Đặc trưng & Học máy):** Phụ trách tính Toán đặc trưng hình học, Code ML (Train/Test SVM hoặc KNN), vẽ Confusion Matrix, vẽ biểu đồ Độ chính xác. Viết phần *Kết quả* và *Kết luận*.
