import ollama
import os
import sys

# Thiết lập env var để giữ model trong VRAM lâu (giá trị -1: giữ mãi, tối ưu realtime, tránh reload latency)
# Chỉnh sửa: Nếu muốn thời gian cụ thể (phút), thay -1 bằng số phút (ví dụ: '5' cho 5 phút)
os.environ['OLLAMA_KEEP_ALIVE'] = '-1'
MODEL = 'qwen2.5:1.5b'

# Host Ollama server: Mặc định localhost, thay nếu remote
# Chỉnh sửa: Nếu chạy Docker/remote, thay bằng 'http://ip:11434'
OLLAMA_HOST = 'http://localhost:11434'

# System prompt bằng tiếng Việt: Tập trung suy nghĩ nhanh, trả lời ngắn gọn, chính xác
# Chỉnh sửa: Thay đổi nội dung nếu cần tùy chỉnh hành vi
SYSTEM_PROMPT = """
Bạn tên là Kitty một trợ lý ảo AI siêu tốc. Mục tiêu: Suy nghĩ nhanh, trả lời ngắn gọn, chính xác 100%, chỉ được trả lời bằng Tiếng Việt. 
- Trước khi trả lời: Phân tích nhanh vấn đề, loại bỏ thông tin thừa.
- Trả lời: Chỉ dùng từ ngữ cần thiết, không giải thích thừa trừ khi hỏi.
- Nếu không chắc chắn: Nói 'Kitty không biết' thay vì đoán.
"""

# Options cho inference: Tối ưu tốc độ (temperature thấp: ít sáng tạo, nhanh chính xác; num_ctx nhỏ: giảm memory/latency; num_predict giới hạn: trả lời ngắn)
# Chỉnh sửa: Tăng num_ctx nếu cần context dài hơn (nhưng chậm hơn); temperature 0.0-1.0 (thấp hơn = chính xác hơn)
OPTIONS = {
    'temperature': 0.1,  # Thấp để suy nghĩ nhanh, chính xác, ít ngẫu nhiên
    'num_ctx': 512,      # Context size nhỏ để tiết kiệm VRAM và tăng tốc trên Jetson
    'num_predict': 100   # Giới hạn max tokens output để trả lời ngắn gọn, realtime
}

# Lịch sử chat: Để giữ context, nhưng trim nếu dài để tránh latency
conversation_history = [{'role': 'system', 'content': SYSTEM_PROMPT}]

def trim_history(messages, max_length=5):
    """Trim lịch sử để giữ ngắn, tối ưu memory/speed"""
    # Giữ system prompt + max_length tin nhắn gần nhất
    if len(messages) > max_length + 1:  # +1 cho system
        return [messages[0]] + messages[-max_length:]
    return messages

def generate_response(user_input):
    """Generate response với stream cho realtime"""
    global conversation_history
    conversation_history.append({'role': 'user', 'content': user_input})
    conversation_history = trim_history(conversation_history)  # Trim để tối ưu

    stream = ollama.chat(
        model=MODEL,
        messages=conversation_history,
        stream=True,
        options=OPTIONS
    )

    response = ''
    for chunk in stream:
        content = chunk['message']['content']
        print(content, end='', flush=True)  # Print realtime cho robot output
        response += content

    print()  # Newline sau response
    conversation_history.append({'role': 'assistant', 'content': response})

if __name__ == '__main__':
    print("Chat realtime với AI (nhập 'exit' để dừng)")
    while True:
        user_input = input("Bạn: ")
        if user_input.lower() == 'exit':
            break
        generate_response(user_input)