import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Ceramic Tile Defect Detection", layout="wide")

st.title("Phát hiện Lỗi Gạch Men - MVP Demo")
st.markdown("""
Pipeline song song (Parallel Pipeline):
- **Nhánh Grout**: Grayscale -> Grout Filter -> Grout Canny -> Grout Closing -> Grout Mask
- **Nhánh Lỗi**: Grayscale -> Defect Filter -> Defect Canny
- **Kết hợp**: Defect Edges - Grout Mask -> Defect Morphology -> Contours -> Classification
""")

def adaptive_median_filter(img, max_s=7):
    """
    Vectorized Adaptive Median Filter for 2D Grayscale images.
    """
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
        
    output = np.zeros_like(gray)
    undecided = np.ones_like(gray, dtype=bool)
    
    for s in range(3, max_s + 1, 2):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (s, s))
        z_min = cv2.erode(gray, kernel)
        z_max = cv2.dilate(gray, kernel)
        z_med = cv2.medianBlur(gray, s)
        
        passed_a = (z_med > z_min) & (z_med < z_max)
        decide_now = undecided & passed_a
        
        if np.any(decide_now):
            passed_b = (gray > z_min) & (gray < z_max)
            output[decide_now] = np.where(passed_b[decide_now], gray[decide_now], z_med[decide_now])
            undecided[decide_now] = False
            
        if not np.any(undecided):
            break
            
    if np.any(undecided):
        output[undecided] = z_med[undecided]
        
    return output


# Load image
@st.cache_data
def load_image(path):
    import os
    if not os.path.exists(path):
        return None
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

uploaded_file = st.file_uploader("Tải lên ảnh gạch men cần kiểm tra (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

img = None
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if opencv_img is not None:
        img = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)
else:
    # Thử load ảnh mẫu mặc định làm fallback
    img = load_image('sample_tile.jpg')
    if img is not None:
        st.info("Đang hiển thị ảnh mẫu mặc định (`sample_tile.jpg`). Bạn có thể tải lên ảnh khác ở trên để kiểm tra.")

if img is None:
    st.warning("Vui lòng tải lên một ảnh gạch men để bắt đầu phân tích.")
    st.stop()

# --- SIDEBAR (Parameter Sweep) ---
st.sidebar.header("Khảo sát tham số (Parameter Sweeps)")

# Nhánh 1: Cấu hình Grout Lines
st.sidebar.subheader("1. Cấu hình Grout Lines")
use_grout_removal = st.sidebar.checkbox("Kích hoạt Grout Line Removal", value=True)
if use_grout_removal:
    st.sidebar.markdown("**[Grout] Bộ lọc nhiễu**")
    grout_filter_type = st.sidebar.selectbox(
        "Chọn bộ lọc Grout", 
        ["Median Filter", "Adaptive Median Filter", "Gaussian Filter", "Bilateral Filter"],
        key="grout_filter_type"
    )
    if grout_filter_type == "Median Filter":
        grout_median_ksize = st.sidebar.slider("Grout Median Kernel Size", min_value=1, max_value=15, step=2, value=5, key="grout_median")
    elif grout_filter_type == "Adaptive Median Filter":
        grout_adaptive_max_ksize = st.sidebar.slider("Grout Adaptive Median Max Kernel Size", min_value=3, max_value=15, step=2, value=7, key="grout_adaptive_max_ksize")
    elif grout_filter_type == "Gaussian Filter":
        grout_gaussian_ksize = st.sidebar.slider("Grout Gaussian Kernel Size", min_value=1, max_value=15, step=2, value=5, key="grout_gaussian")
        grout_gaussian_sigma = st.sidebar.slider("Grout Gaussian Sigma X", min_value=0.0, max_value=5.0, step=0.1, value=1.0, key="grout_gaussian_sig")
    else:
        grout_bilateral_d = st.sidebar.slider("Grout Bilateral Diameter (d)", min_value=1, max_value=15, step=1, value=9, key="grout_bilateral")
        grout_bilateral_sigma_color = st.sidebar.slider("Grout Bilateral Sigma Color", min_value=10, max_value=150, step=5, value=75, key="grout_bil_color")
        grout_bilateral_sigma_space = st.sidebar.slider("Grout Bilateral Sigma Space", min_value=10, max_value=150, step=5, value=75, key="grout_bil_space")

    st.sidebar.markdown("**[Grout] Phát hiện biên (Edge Detection)**")
    grout_edge_type = st.sidebar.selectbox(
        "Phương pháp phát hiện biên Grout",
        ["Canny", "Hessian Filter"],
        key="grout_edge_type"
    )
    if grout_edge_type == "Canny":
        grout_canny_min = st.sidebar.slider("Grout Canny Min Value", 0, 255, 30, key="grout_canny_min")
        grout_canny_max = st.sidebar.slider("Grout Canny Max Value", 0, 255, 100, key="grout_canny_max")
    else:
        grout_hessian_sigma = st.sidebar.slider("Grout Hessian Sigma", min_value=0.5, max_value=5.0, step=0.5, value=1.0, key="grout_hessian_sigma")
        grout_hessian_threshold = st.sidebar.slider("Grout Hessian Threshold", min_value=0.001, max_value=0.500, step=0.001, value=0.020, key="grout_hessian_threshold", format="%.3f")
        grout_hessian_black_ridges = st.sidebar.checkbox("Phát hiện đường mạch tối (Black ridges)", value=True, key="grout_hessian_black")

    use_grout_closing = st.sidebar.checkbox("Kích hoạt Closing cho Grout Lines", value=True, key="use_grout_closing")
    if use_grout_closing:
        grout_closing_shape_str = st.sidebar.selectbox(
            "Hình dạng Kernel Closing Grout", 
            ["RECT (Hình chữ nhật)", "ELLIPSE (Hình elip)", "CROSS (Chữ thập)"],
            index=0,
            key="grout_closing_shape"
        )
        grout_closing_ksize = st.sidebar.slider(
            "Grout Closing Kernel Size", 
            min_value=1, max_value=21, step=2, value=5,
            key="grout_closing_ksize"
        )
    
    st.sidebar.markdown("**[Grout] Hough Lines & Mask**")
    hough_threshold = st.sidebar.slider(
        "Hough Threshold (Ngưỡng vote)", 
        min_value=30, max_value=200, value=80,
        key="hough_thresh"
    )
    min_line_length = st.sidebar.slider(
        "Min Line Length (px)", 
        min_value=20, max_value=500, value=100,
        key="min_line_len"
    )
    max_line_gap = st.sidebar.slider(
        "Max Line Gap (px)", 
        min_value=1, max_value=50, value=10,
        key="max_line_gap"
    )
    grout_line_width = st.sidebar.slider(
        "Grout Mask Width (px)", 
        min_value=1, max_value=30, value=10,
        key="grout_mask_w"
    )

# Nhánh 2: Cấu hình Phát hiện Lỗi
st.sidebar.subheader("2. Cấu hình Phát hiện Lỗi (Cracks/Pin-holes)")
st.sidebar.markdown("**[Lỗi] Bộ lọc nhiễu**")
defect_filter_type = st.sidebar.selectbox(
    "Chọn bộ lọc Defect", 
    ["Median Filter", "Adaptive Median Filter", "Gaussian Filter", "Bilateral Filter"],
    key="defect_filter_type"
)
if defect_filter_type == "Median Filter":
    defect_median_ksize = st.sidebar.slider("Defect Median Kernel Size", min_value=1, max_value=15, step=2, value=3, key="defect_median")
elif defect_filter_type == "Adaptive Median Filter":
    defect_adaptive_max_ksize = st.sidebar.slider("Defect Adaptive Median Max Kernel Size", min_value=3, max_value=15, step=2, value=7, key="defect_adaptive_max_ksize")
elif defect_filter_type == "Gaussian Filter":
    defect_gaussian_ksize = st.sidebar.slider("Defect Gaussian Kernel Size", min_value=1, max_value=15, step=2, value=3, key="defect_gaussian")
    defect_gaussian_sigma = st.sidebar.slider("Defect Gaussian Sigma X", min_value=0.0, max_value=5.0, step=0.1, value=1.0, key="defect_gaussian_sig")
else:
    defect_bilateral_d = st.sidebar.slider("Defect Bilateral Diameter (d)", min_value=1, max_value=15, step=1, value=5, key="defect_bilateral")
    defect_bilateral_sigma_color = st.sidebar.slider("Defect Bilateral Sigma Color", min_value=10, max_value=150, step=5, value=75, key="defect_bil_color")
    defect_bilateral_sigma_space = st.sidebar.slider("Defect Bilateral Sigma Space", min_value=10, max_value=150, step=5, value=75, key="defect_bil_space")

st.sidebar.markdown("**[Lỗi] Phát hiện biên (Edge Detection)**")
defect_edge_type = st.sidebar.selectbox(
    "Phương pháp phát hiện biên Defect",
    ["Canny", "Hessian Filter"],
    key="defect_edge_type"
)
if defect_edge_type == "Canny":
    defect_canny_min = st.sidebar.slider("Defect Canny Min Value", 0, 255, 30, key="defect_canny_min")
    defect_canny_max = st.sidebar.slider("Defect Canny Max Value", 0, 255, 100, key="defect_canny_max")
else:
    defect_hessian_sigma = st.sidebar.slider("Defect Hessian Sigma", min_value=0.5, max_value=5.0, step=0.5, value=1.0, key="defect_hessian_sigma")
    defect_hessian_threshold = st.sidebar.slider("Defect Hessian Threshold", min_value=0.001, max_value=0.500, step=0.001, value=0.020, key="defect_hessian_threshold", format="%.3f")
    defect_hessian_black_ridges = st.sidebar.checkbox("Phát hiện đường nứt tối (Black ridges)", value=True, key="defect_hessian_black")

st.sidebar.markdown("**[Lỗi] Toán tử hình thái học**")
use_morphology = st.sidebar.checkbox("Kích hoạt Morphology cho Lỗi", value=True, key="use_morphology")
if use_morphology:
    morph_op = st.sidebar.selectbox(
        "Chọn phép toán", 
        ["Closing (Đóng)", "Opening (Mở)", "Dilation (Giãn)", "Erosion (Co)", "Gradient"],
        key="morph_op"
    )
    morph_shape_str = st.sidebar.selectbox(
        "Hình dạng Kernel", 
        ["RECT (Hình chữ nhật)", "ELLIPSE (Hình elip)", "CROSS (Chữ thập)"],
        key="morph_shape"
    )
    morph_ksize_x = st.sidebar.slider("Kernel Size X", 1, 31, step=2, value=9, key="morph_ksize_x")
    morph_ksize_y = st.sidebar.slider("Kernel Size Y", 1, 31, step=2, value=9, key="morph_ksize_y")

# Nhánh 3: Quy tắc phân loại
st.sidebar.subheader("3. Quy tắc phân loại")
min_area = st.sidebar.slider("Min Area (Loại bỏ nhiễu)", 1, 100, value=10, key="min_area")
aspect_ratio_thresh = st.sidebar.slider("Aspect Ratio Threshold (Phân biệt Nứt/Lỗ)", 1.0, 5.0, value=2.0, key="aspect_ratio_thresh")

# --- PIPELINE THỰC THI ---

# --- PIPELINE THỰC THI ---

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# ==================== NHÁNH 1: XỬ LÝ GROUT LINES ====================
if use_grout_removal:
    # 1.1. Lọc nhiễu riêng cho Grout
    if grout_filter_type == "Median Filter":
        blurred_grout = cv2.medianBlur(gray, grout_median_ksize)
        grout_filter_caption = f"Grout Median: {grout_median_ksize}x{grout_median_ksize}"
    elif grout_filter_type == "Adaptive Median Filter":
        blurred_grout = adaptive_median_filter(gray, grout_adaptive_max_ksize)
        grout_filter_caption = f"Grout Adaptive Median (Max Kernel: {grout_adaptive_max_ksize}x{grout_adaptive_max_ksize})"
    elif grout_filter_type == "Gaussian Filter":
        blurred_grout = cv2.GaussianBlur(gray, (grout_gaussian_ksize, grout_gaussian_ksize), grout_gaussian_sigma)
        grout_filter_caption = f"Grout Gaussian: {grout_gaussian_ksize}x{grout_gaussian_ksize}, Sigma: {grout_gaussian_sigma}"
    else:
        blurred_grout = cv2.bilateralFilter(gray, grout_bilateral_d, grout_bilateral_sigma_color, grout_bilateral_sigma_space)
        grout_filter_caption = f"Grout Bilateral d: {grout_bilateral_d}, SigmaColor: {grout_bilateral_sigma_color}, SigmaSpace: {grout_bilateral_sigma_space}"

    # 1.2. Phát hiện biên riêng cho Grout
    if grout_edge_type == "Canny":
        edges_grout = cv2.Canny(blurred_grout, grout_canny_min, grout_canny_max)
    else:
        from skimage.filters import hessian
        h_out = hessian(blurred_grout, sigmas=[grout_hessian_sigma], black_ridges=grout_hessian_black_ridges)
        edges_grout = (h_out > grout_hessian_threshold).astype(np.uint8) * 255

    # 1.3. Áp dụng Morphological Closing cho Grout
    if use_grout_closing:
        grout_closing_shape_map = {
            "RECT (Hình chữ nhật)": cv2.MORPH_RECT,
            "ELLIPSE (Hình elip)": cv2.MORPH_ELLIPSE,
            "CROSS (Chữ thập)": cv2.MORPH_CROSS
        }
        grout_closing_kernel = cv2.getStructuringElement(
            grout_closing_shape_map[grout_closing_shape_str], 
            (grout_closing_ksize, grout_closing_ksize)
        )
        edges_for_hough = cv2.morphologyEx(edges_grout, cv2.MORPH_CLOSE, grout_closing_kernel)
        grout_closing_caption = f"Closing: {grout_closing_shape_str} {grout_closing_ksize}x{grout_closing_ksize}"
    else:
        edges_for_hough = edges_grout.copy()
        grout_closing_caption = "Pre-closing: Đã tắt"
        
    # 1.4. Tìm Grout Lines
    lines = cv2.HoughLinesP(
        edges_for_hough, 
        rho=1,                          # Độ phân giải khoảng cách (pixel)
        theta=np.pi / 180,              # Độ phân giải góc (radian)
        threshold=hough_threshold, 
        minLineLength=min_line_length, 
        maxLineGap=max_line_gap
    )
    
    # Tạo ảnh trực quan: vẽ grout lines lên ảnh gốc
    grout_viz_img = img.copy()
    grout_mask = np.zeros_like(gray)
    
    num_grout_lines = 0
    if lines is not None:
        num_grout_lines = len(lines)
        for line in lines:
            x1, y1, x2, y2 = line.flatten()[:4]
            cv2.line(grout_viz_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.line(grout_mask, (x1, y1), (x2, y2), 255, thickness=grout_line_width)
    
    # Dilate mask thêm 1 chút để đảm bảo xóa sạch vùng grout
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grout_mask_dilated = cv2.dilate(grout_mask, dilate_kernel, iterations=1)
    
    grout_caption = f"Detected {num_grout_lines} grout lines (Threshold: {hough_threshold}, MinLen: {min_line_length})"
else:
    blurred_grout = gray.copy()
    grout_filter_caption = "Grout Filter: Đã tắt"
    edges_grout = np.zeros_like(gray)
    edges_for_hough = np.zeros_like(gray)
    grout_closing_caption = "Pre-closing: Đã tắt"
    grout_viz_img = img.copy()
    grout_mask_dilated = np.zeros_like(gray)
    num_grout_lines = 0
    grout_caption = "Grout Removal: Đã tắt"


# ==================== NHÁNH 2: PHÁT HIỆN LỖI (CRACK/PIN-HOLE) ====================
# 2.1. Lọc nhiễu riêng cho Lỗi
if defect_filter_type == "Median Filter":
    blurred_defect = cv2.medianBlur(gray, defect_median_ksize)
    defect_filter_caption = f"Defect Median: {defect_median_ksize}x{defect_median_ksize}"
elif defect_filter_type == "Adaptive Median Filter":
    blurred_defect = adaptive_median_filter(gray, defect_adaptive_max_ksize)
    defect_filter_caption = f"Defect Adaptive Median (Max Kernel: {defect_adaptive_max_ksize}x{defect_adaptive_max_ksize})"
elif defect_filter_type == "Gaussian Filter":
    blurred_defect = cv2.GaussianBlur(gray, (defect_gaussian_ksize, defect_gaussian_ksize), defect_gaussian_sigma)
    defect_filter_caption = f"Defect Gaussian: {defect_gaussian_ksize}x{defect_gaussian_ksize}, Sigma: {defect_gaussian_sigma}"
else:
    blurred_defect = cv2.bilateralFilter(gray, defect_bilateral_d, defect_bilateral_sigma_color, defect_bilateral_sigma_space)
    defect_filter_caption = f"Defect Bilateral d: {defect_bilateral_d}, SigmaColor: {defect_bilateral_sigma_color}, SigmaSpace: {defect_bilateral_sigma_space}"

# 2.2. Phát hiện biên riêng cho Lỗi
if defect_edge_type == "Canny":
    edges_defect = cv2.Canny(blurred_defect, defect_canny_min, defect_canny_max)
else:
    from skimage.filters import hessian
    h_out = hessian(blurred_defect, sigmas=[defect_hessian_sigma], black_ridges=defect_hessian_black_ridges)
    edges_defect = (h_out > defect_hessian_threshold).astype(np.uint8) * 255

# 2.3. Loại bỏ grout line khỏi bản đồ biên của Lỗi
clean_edges = cv2.bitwise_and(edges_defect, cv2.bitwise_not(grout_mask_dilated))

# 2.4. Áp dụng Morphology cho bản đồ biên đã làm sạch
if use_morphology:
    shape_map = {
        "RECT (Hình chữ nhật)": cv2.MORPH_RECT,
        "ELLIPSE (Hình elip)": cv2.MORPH_ELLIPSE,
        "CROSS (Chữ thập)": cv2.MORPH_CROSS
    }
    kernel_shape = shape_map[morph_shape_str]
    kernel = cv2.getStructuringElement(kernel_shape, (morph_ksize_x, morph_ksize_y))
    
    op_map = {
        "Closing (Đóng)": cv2.MORPH_CLOSE,
        "Opening (Mở)": cv2.MORPH_OPEN,
        "Dilation (Giãn)": cv2.MORPH_DILATE,
        "Erosion (Co)": cv2.MORPH_ERODE,
        "Gradient": cv2.MORPH_GRADIENT
    }
    morph_op_val = op_map[morph_op]
    morph_output = cv2.morphologyEx(clean_edges, morph_op_val, kernel)
    morph_caption = f"Op: {morph_op}, Kernel: {morph_shape_str} {morph_ksize_x}x{morph_ksize_y}"
else:
    morph_output = clean_edges.copy()
    morph_caption = "Morphology: Đã tắt (Không áp dụng)"

# Bước 5: Trích xuất đặc trưng & Phân loại
contours, _ = cv2.findContours(morph_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output_img = img.copy()

# Vẽ tất cả contour thô để trực quan hóa bước trung gian
raw_contours_img = img.copy()
cv2.drawContours(raw_contours_img, contours, -1, (0, 255, 0), 2)

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

# Hàng 1: Nhánh Grout Lines
st.markdown("### 🔬 Nhánh 1: Xử lý Grout Lines (Tìm & Xóa đường mạch)")
col1_g, col2_g, col3_g, col4_g, col5_g = st.columns(5)
with col1_g:
    st.subheader("1. Ảnh gốc")
    st.image(img, width='stretch')

with col2_g:
    st.subheader("2. Grout Filtered")
    st.image(blurred_grout, width='stretch', caption=grout_filter_caption, clamp=True)

with col3_g:
    if use_grout_removal:
        if grout_edge_type == "Canny":
            st.subheader("3. Grout Canny")
            grout_edge_caption = f"Canny Threshold: {grout_canny_min}-{grout_canny_max}"
        else:
            st.subheader("3. Grout Hessian")
            grout_edge_caption = f"Hessian Sigma: {grout_hessian_sigma}, Thresh: {grout_hessian_threshold:.3f}"
    else:
        st.subheader("3. Grout Edges")
        grout_edge_caption = "Grout Removal: Đã tắt"
    st.image(edges_grout, width='stretch', caption=grout_edge_caption)

with col4_g:
    st.subheader("4. Grout Closing")
    st.image(edges_for_hough, width='stretch', caption=grout_closing_caption, clamp=True)

with col5_g:
    st.subheader("5. Grout Lines")
    st.image(grout_viz_img, width='stretch', caption=grout_caption)

# Hàng 2: Nhánh Defect Detection
st.markdown("---")
st.markdown("### 🔬 Nhánh 2: Phát hiện Lỗi (Cracks/Pin-holes)")
col1_d, col2_d, col3_d, col4_d, col5_d = st.columns(5)
with col1_d:
    st.subheader("6. Grout Mask")
    st.image(grout_mask_dilated, width='stretch', 
             caption=f"Mask width: {grout_line_width if use_grout_removal else 0}px")

with col2_d:
    st.subheader("7. Defect Filtered")
    st.image(blurred_defect, width='stretch', caption=defect_filter_caption, clamp=True)

with col3_d:
    if defect_edge_type == "Canny":
        st.subheader("8. Defect Canny")
        defect_edge_caption = f"Canny Threshold: {defect_canny_min}-{defect_canny_max}"
    else:
        st.subheader("8. Defect Hessian")
        defect_edge_caption = f"Hessian Sigma: {defect_hessian_sigma}, Thresh: {defect_hessian_threshold:.3f}"
    st.image(edges_defect, width='stretch', caption=defect_edge_caption)

with col4_d:
    st.subheader("9. Clean Edges")
    st.image(clean_edges, width='stretch', 
             caption="Edges sau khi loại bỏ Grout Mask")

with col5_d:
    st.subheader("10. Defect Morph")
    st.image(morph_output, width='stretch', caption=morph_caption, clamp=True)

# Hàng 3: Kết quả cuối cùng
st.markdown("---")
st.subheader("🎯 Kết quả Phân loại cuối cùng")
col_res1, col_res2, col_res3 = st.columns([1.5, 1.5, 1])
with col_res1:
    st.subheader("Contours thô phát hiện")
    st.image(raw_contours_img, width='stretch', 
             caption=f"Tổng contour phát hiện: {len(contours)}")
with col_res2:
    st.subheader("Kết quả phân loại và khoanh vùng")
    st.image(output_img, width='stretch')
with col_res3:
    st.metric("Tổng lỗi phát hiện", len(results))
    num_cracks = sum(1 for r in results if r["Label"] == "Crack")
    num_pinholes = sum(1 for r in results if r["Label"] == "Pin-hole")
    st.metric("🔴 Vết nứt (Crack)", num_cracks)
    st.metric("🔵 Lỗ kim (Pin-hole)", num_pinholes)
    if use_grout_removal:
        st.info(f"✅ Đã loại bỏ {num_grout_lines} grout lines khỏi edge map")
    st.write("**Danh sách lỗi phát hiện:**")
    st.table(results)
