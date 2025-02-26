import cv2

# Đọc ảnh
img = cv2.imread("D:\SourceCode\ProjectOJT\mug2.jpg")

# Kiểm tra nếu ảnh bị lỗi
if img is None:
    print("⚠️ Ảnh bị lỗi hoặc không hợp lệ!")
else:
    # Lưu lại ảnh với định dạng chuẩn
    cv2.imwrite("D:/SourceCode/ProjectOJT/OJT_TASK3_DEPLOY/mug2_fixed.jpg", img)
    print("✅ Đã lưu ảnh mới: mug2_fixed.jpg")
