import cv2

cap = cv2.VideoCapture(0)

while True:
    # Đọc từng khung hình từ camera
    ret, frame = cap.read()
    
    # Kiểm tra xem việc đọc khung hình có thành công hay không
    if not ret:
        print("Không thể đọc khung hình từ camera")
        break
    
    # Hiển thị khung hình
    cv2.imshow('Camera', frame)
    
    # Kiểm tra phím nhấn 'q' để thoát vòng lặp
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
