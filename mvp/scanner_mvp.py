import cv2
import numpy as np
from PIL import Image
import os
import itertools
import io

def resize_image(image, height=None, width=None):
    """Giữ nguyên tỷ lệ khung hình khi resize ảnh"""
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

def order_points(pts):
    """Sắp xếp 4 tọa độ theo thứ tự: Trái-trên, Phải-trên, Phải-dưới, Trái-dưới"""
    rect = np.zeros((4, 2), dtype="float32")
    
    # Trái-trên có tổng (x+y) nhỏ nhất, Phải-dưới có tổng (x+y) lớn nhất
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Trái-dưới có hiệu (y-x) lớn nhất, Phải-trên có hiệu nhỏ nhất
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def four_point_transform(image, pts):
    """Thực hiện biến đổi phối cảnh (Perspective Transform) để làm phẳng ảnh"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Tính toán chiều rộng mới của ảnh
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # Tính toán chiều cao mới của ảnh
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Lập ma trận tọa độ đích để ánh xạ ảnh sang
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    # Tính ma trận chuyển đổi và thực hiện wrap
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped

def preprocess_image_otsu(image):
    """Nhị phân hóa bằng phương pháp Otsu + Gaussian Blur để làm nổi bật các ô vuông góc"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

def whiten_page(image):
    """Tẩy trắng trang bằng phương pháp Adaptive Thresholding (kiểu CamScanner)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    T = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
    return T

def image_to_pdf_bytes(image_np):
    """Chuyển đổi ảnh (Numpy Array) thành dữ liệu PDF (bytes) trong bộ nhớ"""
    pil_img = Image.fromarray(image_np)
    pdf_buffer = io.BytesIO()
    pil_img.save(pdf_buffer, format="PDF", resolution=100.0)
    return pdf_buffer.getvalue()

def save_image_as_pdf(image_np, output_path):
    """Lưu ảnh (Numpy Array) thành file PDF"""
    pil_img = Image.fromarray(image_np)
    pil_img.save(output_path, "PDF", resolution=100.0)

def find_corner_markers(image):
    """Tìm 4 ô vuông ở 4 góc tờ giấy và trả về tọa độ tâm của chúng"""
    thresh = preprocess_image_otsu(image)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        # Lọc các contour có diện tích phù hợp với ô vuông góc trên ảnh resize (chiều cao 500px)
        if area < 15 or area > 300:
            continue
            
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w) / h if h > 0 else 0
        if not (0.5 <= aspect_ratio <= 2.0):
            continue
            
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            candidates.append((cx, cy))
            
    if len(candidates) < 4:
        return None
        
    # Sử dụng Convex Hull để tìm các điểm cực trị ngoài cùng
    pts_arr = np.array(candidates, dtype=np.int32).reshape(-1, 1, 2)
    hull = cv2.convexHull(pts_arr)
    hull_pts = hull.reshape(-1, 2)
    
    if len(hull_pts) < 4:
        return None
        
    if len(hull_pts) == 4:
        return hull_pts.astype("float32")
        
    # Nếu convex hull có nhiều hơn 4 điểm, tìm bộ 4 điểm tạo ra tứ giác có diện tích lớn nhất
    best_pts = None
    max_area = 0
    for pts in itertools.combinations(hull_pts, 4):
        pts_arr = np.array(pts, dtype="float32")
        hull_quad = cv2.convexHull(pts_arr)
        if len(hull_quad) == 4:
            area = cv2.contourArea(hull_quad)
            if area > max_area:
                max_area = area
                best_pts = pts_arr
                
    return best_pts

def scan_document_to_pdf(image_path, output_pdf_path):
    if not os.path.exists(image_path):
        print(f"Lỗi: Không tìm thấy ảnh tại {image_path}")
        return
        
    image = cv2.imread(image_path)
    if image is None:
        print("Lỗi: Không thể đọc ảnh.")
        return
        
    # --- BƯỚC 1: CHUẨN HÓA ĐỘ PHÂN GIẢI ---
    # Tính tỷ lệ scale và resize về chiều cao 500px
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = resize_image(image, height=500)
    
    # --- BƯỚC 3: PHÁT HIỆN BIÊN VÀ TÌM TỨ GIÁC (CHƯƠNG 3, 4) ---
    # Thử phát hiện tài liệu qua 4 ô vuông góc trước
    screenCnt = find_corner_markers(image)
    
    if screenCnt is not None:
        screenCnt = screenCnt.reshape(4, 1, 2)
    else:
        # Nếu không tìm thấy, sử dụng phương pháp tìm contour lớn nhất mặc định làm fallback
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 75, 200)
        
        cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
        
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                screenCnt = approx
                break
            
    if screenCnt is None:
        print("Không thể tìm thấy khung viền tài liệu. Vui lòng thử lại với nền tương phản hơn.")
        return
        
    # --- BƯỚC 4: LÀM PHẲNG TRÊN ẢNH GỐC (CHƯƠNG 2) ---
    # Áp dụng tỷ lệ ratio để nhân tọa độ 4 góc lên kích thước của ảnh gốc
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    
    # --- BƯỚC 5: TẨY TRẮNG VÀ XUẤT PDF ---
    # Sử dụng Adaptive Thresholding để làm trắng nền (đặc trưng của CamScanner)
    T = whiten_page(warped)
    
    # Xuất ra PDF sử dụng PIL
    save_image_as_pdf(T, output_pdf_path)
    print(f"Thành công! Đã lưu tài liệu thành file {output_pdf_path}")

if __name__ == "__main__":
    print("--- Document Scanner MVP Pipeline ---")
    print("Sử dụng hàm: scan_document_to_pdf('input.jpg', 'output.pdf') để chạy thử.")
    # VD: scan_document_to_pdf("image.png", "scanned_result.pdf")
    scan_document_to_pdf("image.png", "scanned_result.pdf")
