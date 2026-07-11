import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Ceramic Tile Defect Detection", layout="wide")

st.title("Phát hiện Lỗi Gạch Men - MVP Demo")
st.markdown("Pipeline: **Grayscale -> Median Filter -> Canny Edge -> Morphology -> Contours -> Classification**")

# Load image
@st.cache_data
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

image_path = 'sample_tile.jpg'
img = load_image(image_path)

if img is None:
    st.error("Không tìm thấy file `sample_tile.jpg`. Hãy chạy file `generate_sample.py` trước!")
    st.stop()

# --- SIDEBAR (Parameter Sweep) ---
st.sidebar.header("Khảo sát tham số (Parameter Sweeps)")

st.sidebar.subheader("1. Lọc nhiễu (Tiền xử lý)")
median_ksize = st.sidebar.slider("Median Filter Kernel Size", min_value=1, max_value=15, step=2, value=5)

st.sidebar.subheader("2. Phát hiện biên (Canny)")
canny_min = st.sidebar.slider("Canny Min Value", 0, 255, 30)
canny_max = st.sidebar.slider("Canny Max Value", 0, 255, 100)

st.sidebar.subheader("3. Toán tử hình thái học (Morphology)")
morph_ksize_x = st.sidebar.slider("Closing Kernel Size X", 1, 31, step=2, value=9)
morph_ksize_y = st.sidebar.slider("Closing Kernel Size Y", 1, 31, step=2, value=9)

st.sidebar.subheader("4. Phân loại (Classification Rules)")
min_area = st.sidebar.slider("Min Area (Loại bỏ nhiễu)", 1, 100, value=10)
aspect_ratio_thresh = st.sidebar.slider("Aspect Ratio Threshold (Phân biệt Nứt/Lỗ)", 1.0, 5.0, value=2.0)

# --- PIPELINE THỰC THI ---

# Bước 1: Tiền xử lý
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
blurred = cv2.medianBlur(gray, median_ksize)

# Bước 2: Edge Detection
edges = cv2.Canny(blurred, canny_min, canny_max)

# Bước 3: Morphology
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_ksize_x, morph_ksize_y))
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# Bước 4: Trích xuất đặc trưng & Phân loại
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output_img = img.copy()

results = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < min_area:
        continue
        
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h
    # Lật ngược aspect ratio nếu h > w để luôn có tỷ lệ >= 1
    if aspect_ratio < 1:
        aspect_ratio = 1 / aspect_ratio
        
    # Phân loại đơn giản bằng Rule-based (thay thế SVM cho bản MVP)
    if aspect_ratio > aspect_ratio_thresh:
        label = "Crack"
        color = (255, 0, 0) # Red
    else:
        label = "Pin-hole"
        color = (0, 0, 255) # Blue
        
    cv2.rectangle(output_img, (x, y), (x+w, y+h), color, 2)
    cv2.putText(output_img, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    results.append({"Label": label, "Area": area, "Aspect Ratio": round(aspect_ratio, 2)})

# --- HIỂN THỊ TRỰC QUAN (VISUALIZATION) ---

col1, col2 = st.columns(2)
with col1:
    st.subheader("1. Ảnh gốc")
    st.image(img, use_column_width=True)
    
    st.subheader("3. Canny Edge Detection")
    st.image(edges, use_column_width=True, caption=f"Threshold: {canny_min}-{canny_max}")

with col2:
    st.subheader("2. Lọc nhiễu (Median Blur)")
    st.image(blurred, use_column_width=True, caption=f"Kernel: {median_ksize}x{median_ksize}", clamp=True)
    
    st.subheader("4. Hình thái học (Closing)")
    st.image(closed, use_column_width=True, caption=f"Kernel: {morph_ksize_x}x{morph_ksize_y}", clamp=True)

st.markdown("---")
st.subheader("5. Kết quả Phân loại cuối cùng (Result)")
col3, col4 = st.columns([2, 1])
with col3:
    st.image(output_img, use_column_width=True)
with col4:
    st.write("**Danh sách lỗi phát hiện:**")
    st.table(results)
