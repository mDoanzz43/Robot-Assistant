import serial
import time

ser = serial.Serial('COM5', 115200, timeout=1)
time.sleep(2) # Đợi kết nối ổn định

def send_cmd(cmd):
    print(f"Gửi lệnh: {cmd}")
    ser.write(cmd.encode())
    time.sleep(0.1)

try:
    # Kịch bản test:
    print("Bắt đầu test robot...")
    
    # 1. Test biểu cảm
    send_cmd('S') # Speaking
    time.sleep(3)
    
    # 2. Test hành động servo
    send_cmd('r') # Right wave
    time.sleep(2)
    
    send_cmd('S') # Speaking
    send_cmd('b') # Both swing
    time.sleep(2)

    send_cmd('H') # Trở về mặc định Heart Eyes

except KeyboardInterrupt:
    ser.close()
    print("Dừng test.")