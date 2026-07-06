import streamlit as st
import cv2
import numpy as np
# Import các hàm từ pipeline có sẵn của bạn
from scanner_mvp import (
    resize_image,
    find_corner_markers,
    four_point_transform,
    order_points,
    preprocess_image_otsu,
    whiten_page,
    image_to_pdf_bytes
)

st.set_page_config(layout="wide", page_title="Smart Document Scanner Visualizer")
st.title("📸 Smart Document Scanner Pipeline Visualizer")
st.write("Tải ảnh chụp bài trắc nghiệm lên để xem trực quan từng bước xử lý của thuật toán.")

# 1. Bộ tải ảnh lên (File Uploader)
uploaded_file = st.file_uploader("Chọn ảnh chụp bài trắc nghiệm...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Đọc file ảnh từ bytes sang OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    orig = cv2.imdecode(file_bytes, 1) # BGR image

    # Tạo các cột hiển thị thông tin ảnh gốc
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ảnh chụp gốc")
        st.image(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB), use_container_width=True)

    # Chạy pipeline xử lý
    ratio = orig.shape[0] / 500.0
    image_resized = resize_image(orig, height=500)

    # Bước 3: Tìm corner markers
    screenCnt = find_corner_markers(image_resized)

    # Dùng tabs để visualize từng bước của pipeline
    st.markdown("---")
    st.subheader("🔍 Chi tiết từng bước trong Pipeline")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1. Ảnh Thu nhỏ (Resized)",
        "2. Nhị phân hóa (Otsu)",
        "3. Dò tìm góc (Corners)",
        "4. Duỗi phẳng (Warped)",
        "5. Tẩy trắng & Xuất PDF (Final)"
    ])

    with tab1:
        st.write(f"Ảnh được resize về chiều cao 500px (Kích thước hiện tại: {image_resized.
shape[1]}x{image_resized.shape[0]}).")
        st.image(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB), use_container_width=True)

    with tab2:
        st.write("Nhị phân hóa bằng phương pháp Otsu + Gaussian Blur để làm nổi bật các ô vuông góc.")
        thresh = preprocess_image_otsu(image_resized)
        st.image(thresh, caption="Ảnh nhị phân (Otsu)", use_container_width=True)

    with tab3:
        st.write("Tọa độ 4 góc tìm được (được vẽ đè lên ảnh để kiểm tra độ chính xác):")
        img_corners = image_resized.copy()
        if screenCnt is not None:
            # Vẽ các chấm tròn đỏ tại 4 tâm ô vuông phát hiện được
            for pt in screenCnt:
                cv2.circle(img_corners, (int(pt[0]), int(pt[1])), 8, (0, 0, 255), -1)
            st.image(cv2.cvtColor(img_corners, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.success("Đã tìm thấy 4 ô định vị thành công!")
        else:
            st.error("Không tìm thấy 4 ô vuông góc định vị. Vui lòng kiểm tra lại chất lượng ảnh.")

    with tab4:
        if screenCnt is not None:
            # Áp dụng tỷ lệ ratio để nhân tọa độ lên kích thước gốc và thực hiện biến đổi phối cảnh
            warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
            st.image(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), use_container_width=True)
        else:
            st.warning("Không có dữ liệu 4 góc để thực hiện Warp.")

    with tab5:
        if screenCnt is not None:
            # Tẩy trắng nền
            T = whiten_page(warped)

            col_final, col_download = st.columns([2, 1])
            with col_final:
                st.image(T, caption="Kết quả quét cuối cùng (Chữ đen nền trắng)", use_container_width=True)

            with col_download:
                # Đóng gói ảnh thành PDF trong bộ nhớ (In-memory bytes)
                pdf_bytes = image_to_pdf_bytes(T)

                # Nút tải file PDF về máy tính
                st.download_button(
                    label="📥 Tải xuống file PDF kết quả",
                    data=pdf_bytes,
                    file_name="scanned_omr.pdf",
                    mime="application/pdf"
                )
        else:
            st.warning("Vui lòng sửa lỗi nhận diện góc để xem kết quả cuối cùng.")