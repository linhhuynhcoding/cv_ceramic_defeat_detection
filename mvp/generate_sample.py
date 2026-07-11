import cv2
import numpy as np
import os

def create_synthetic_tile(output_path):
    # Tạo một nền gạch trơn màu xám nhạt (500x500)
    img = np.full((500, 500, 3), 200, dtype=np.uint8)
    
    # Thêm một chút nhiễu Gaussian nhẹ để mô phỏng bề mặt thực tế
    noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)

    # 1. Vẽ một vết nứt (Crack) ngoằn ngoèo, đứt đoạn
    points = [(100, 100), (120, 150), (110, 200), (140, 250), (130, 300), (170, 350)]
    for i in range(len(points) - 1):
        pt1 = points[i]
        pt2 = points[i+1]
        # Thêm chút ngẫu nhiên để nét vẽ không hoàn hảo
        thickness = np.random.randint(1, 3)
        cv2.line(img, pt1, pt2, (50, 50, 50), thickness)
        
    # Thêm vài đốm đen nhỏ li ti dọc theo vết nứt để mô phỏng việc đứt đoạn (Capillary crack)
    for _ in range(10):
        x = np.random.randint(100, 170)
        y = np.random.randint(100, 350)
        cv2.circle(img, (x, y), 1, (70, 70, 70), -1)

    # 2. Vẽ 2 cái Lỗ kim (Pin-hole)
    cv2.circle(img, (350, 200), 3, (30, 30, 30), -1)
    cv2.circle(img, (400, 400), 4, (40, 40, 40), -1)

    # 3. Vẽ một nhiễu lớn không phải là lỗi (ví dụ một vệt ố màu xám nhạt không có cạnh sắc nét)
    cv2.circle(img, (100, 400), 40, (180, 180, 180), -1)
    img = cv2.GaussianBlur(img, (5, 5), 0) # Blur lại một lần cuối
    
    cv2.imwrite(output_path, img)
    print(f"Đã tạo ảnh mẫu tại: {output_path}")

if __name__ == '__main__':
    create_synthetic_tile('sample_tile.jpg')
