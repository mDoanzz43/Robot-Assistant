# sử dụng esp32 để điều khiển 2 servo SG90 làm 2 hand gesture cho robot và màn hình lcd làm biểu cảm 
Esp32 tôi đã nạp flash code trước rồi 

## Đối với phần Servo 
Left -- Right 

User có thể demo bằng cách user tương tác (voice hoặc text) với robot, cụ thể như sau:
+ Dơ/giơ/rơ/ di chuyển... tay trái -> Truyền qua serial tới esp32 nhận lệnh -> Điều khiển tay trái di chuyển lên 
+ Dơ/giơ/rơ/ di chuyển... tay phải -> Truyền qua serial tới esp32 nhận lệnh -> Điều khiển tay phải di chuyển lên 
+ Dơ/giơ/rơ/ di chuyển... cả hai tay -> Truyền qua serial tới esp32 nhận lệnh -> Cả hai tay sẽ di chuyển lên 
+ Bắt tay nào -> Truyền qua serial tới esp32 nhận lệnh -> Robot bắt tay

(các hành động hoạt động theo function đã code ở esp32 trong file esp32_lcd_sero.txt tôi copy vào sẵn rồi). Mỗi khi servo di chuyển hãy đồng thời nói: "Bạn xem tay tôi di chuyển này", ngoại trừ khi bắt tay thì nói: "Bạn hãy giơ tay ra để bắt tay với tôi nào".

## Đối với phần biểu cảm LCD
LCD tôi đã code theo các biểu cảm sau:
+ Heart: Dùng để đánh giá, tổng kết khi học bài xong 
+ Listening: Chờ user nghĩ, nói, nhập lệnh
+ Speaking: Robot nói 
+ Laughing: Khi user mở robot lên (ban đầu, trước khi nhập tên và tuổi)
+ Thinking: Khi robot đang trong quá trình suy nghĩ (trước khi speaking ra)
...

## Yêu cẩu bổ sung: Các mã code phải cần được tối ưu hóa, không được dùng hàm ngắt, nghỉ (chỉ được dùng bộ timer để đếm) 
## Ngoài ra (Thêm print status để check xem LCD và servo nếu gọi đến có hoạt động hay không. Ví dụ: INFO... esp32: Smile, Handshake.... )