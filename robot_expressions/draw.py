# #!/usr/bin/env python3
# """
# Robot Face Expression Generator
# Generates high-quality robot facial expressions for animation
# Size: 320x240 pixels with smooth anti-aliased rendering
# """

# from PIL import Image, ImageDraw
# import os
# import math


# class RobotFaceGenerator:
#     def __init__(self, width=320, height=240, theme="default"):
#         self.width = width
#         self.height = height
#         self.theme = theme
        
#         # Color themes
#         self.themes = {
#             "default": {
#                 "background": (195, 241, 208),  # Light mint green
#                 "primary": (0, 0, 0),            # Black
#                 "accent": (76, 175, 80)          # Dark green
#             },
#             "blue": {
#                 "background": (187, 222, 251),
#                 "primary": (13, 71, 161),
#                 "accent": (33, 150, 243)
#             },
#             "pink": {
#                 "background": (248, 187, 208),
#                 "primary": (136, 14, 79),
#                 "accent": (233, 30, 99)
#             },
#             "orange": {
#                 "background": (255, 224, 178),
#                 "primary": (230, 81, 0),
#                 "accent": (255, 152, 0)
#             },
#             "purple": {
#                 "background": (225, 190, 231),
#                 "primary": (74, 20, 140),
#                 "accent": (156, 39, 176)
#             }
#         }
        
#         self.colors = self.themes.get(theme, self.themes["default"])
        
#         # Feature positions (relative to canvas)
#         self.left_eye_x = int(width * 0.3)
#         self.right_eye_x = int(width * 0.7)
#         self.eye_y = int(height * 0.35)
#         self.mouth_y = int(height * 0.65)
        
#     def create_canvas(self):
#         """Create a new canvas with background color"""
#         img = Image.new('RGB', (self.width, self.height), self.colors["background"])
#         draw = ImageDraw.Draw(img, 'RGBA')
#         return img, draw
    
#     def draw_eye_sleepy(self, draw, x, y, width=60):
#         """Draw sleepy/half-closed eye (arc)"""
#         bbox = [x - width//2, y - 15, x + width//2, y + 15]
#         draw.arc(bbox, start=180, end=0, fill=self.colors["primary"], width=8)
    
#     def draw_eye_closed(self, draw, x, y, width=50):
#         """Draw closed eye (curved line)"""
#         # Draw a smooth arc for closed eye
#         bbox = [x - width//2, y - 10, x + width//2, y + 10]
#         draw.arc(bbox, start=180, end=0, fill=self.colors["primary"], width=6)
    
#     def draw_eye_normal_with_brow(self, draw, x, y, brow_height=0):
#         """Draw normal eye (circle) with eyebrow"""
#         # Eyebrow
#         if brow_height != 0:
#             brow_y = y - 40 + brow_height
#             draw.line([(x - 35, brow_y), (x + 35, brow_y)], 
#                      fill=self.colors["primary"], width=8)
        
#         # Eye
#         draw.ellipse([x - 15, y - 15, x + 15, y + 15], 
#                     fill=self.colors["primary"])
    
#     def draw_eye_surprised(self, draw, x, y):
#         """Draw surprised eye (large circle with highlight)"""
#         # Outer circle
#         draw.ellipse([x - 35, y - 35, x + 35, y + 35], 
#                     fill=self.colors["primary"], 
#                     outline=self.colors["primary"], width=3)
        
#         # Large white highlight
#         draw.ellipse([x - 20, y - 25, x + 8, y - 3], 
#                     fill=(255, 255, 255))
        
#         # Small white highlight
#         draw.ellipse([x - 5, y + 8, x + 5, y + 18], 
#                     fill=(255, 255, 255))
    
#     def draw_eye_happy_closed(self, draw, x, y):
#         """Draw happy closed eye (curved up arc)"""
#         bbox = [x - 25, y - 10, x + 25, y + 10]
#         draw.arc(bbox, start=0, end=180, fill=self.colors["primary"], width=6)
    
#     def draw_mouth_neutral(self, draw, x, y, width=100):
#         """Draw neutral mouth (horizontal line)"""
#         draw.line([(x - width//2, y), (x + width//2, y)], 
#                  fill=self.colors["primary"], width=8)
    
#     def draw_mouth_smile(self, draw, x, y, intensity=1.0):
#         """Draw smile (curved line)"""
#         width = int(100 * intensity)
#         height = int(30 * intensity)
#         bbox = [x - width//2, y - height//2, x + width//2, y + height//2]
#         draw.arc(bbox, start=0, end=180, fill=self.colors["primary"], width=8)
    
#     def draw_mouth_open_speaking(self, draw, x, y, openness=1.0):
#         """Draw open mouth for speaking"""
#         width = int(80 * openness)
#         height = int(40 * openness)
        
#         # Outer mouth
#         draw.ellipse([x - width//2, y - height//2, x + width//2, y + height//2],
#                     fill=self.colors["primary"],
#                     outline=self.colors["primary"], width=3)
        
#         # Inner (light area)
#         inner_scale = 0.7
#         draw.ellipse([x - int(width//2 * inner_scale), y - int(height//2 * inner_scale), 
#                      x + int(width//2 * inner_scale), y + int(height//2 * inner_scale)],
#                     fill=(255, 255, 255))
        
#         # Tongue/accent
#         if openness > 0.7:
#             tongue_width = int(width * 0.4)
#             tongue_height = int(height * 0.3)
#             draw.ellipse([x - tongue_width//2, y + int(height * 0.1), 
#                          x + tongue_width//2, y + int(height * 0.1) + tongue_height],
#                         fill=self.colors["accent"])
    
#     def draw_mouth_wide_open(self, draw, x, y):
#         """Draw wide open mouth (rounded rectangle)"""
#         width = 100
#         height = 45
        
#         # Outer rounded rectangle
#         draw.rounded_rectangle([x - width//2, y - height//2, x + width//2, y + height//2],
#                               radius=20, fill=self.colors["primary"], 
#                               outline=self.colors["primary"], width=3)
        
#         # Inner white area
#         inner_scale = 0.75
#         draw.rounded_rectangle([x - int(width//2 * inner_scale), y - int(height//2 * inner_scale), 
#                                x + int(width//2 * inner_scale), y + int(height//2 * inner_scale)],
#                               radius=15, fill=(255, 255, 255))
        
#         # Accent color at bottom
#         accent_height = int(height * 0.35)
#         draw.rounded_rectangle([x - int(width//2 * inner_scale * 0.6), 
#                                y + int(height//2 * inner_scale) - accent_height,
#                                x + int(width//2 * inner_scale * 0.6), 
#                                y + int(height//2 * inner_scale)],
#                               radius=10, fill=self.colors["accent"])
    
#     def draw_mouth_small_o(self, draw, x, y):
#         """Draw small O mouth"""
#         radius = 15
#         draw.ellipse([x - radius, y - radius, x + radius, y + radius],
#                     fill=self.colors["primary"])
    
#     # ============ EXPRESSION GENERATORS ============
    
#     def generate_idle(self):
#         """Idle/neutral expression"""
#         img, draw = self.create_canvas()
        
#         # Eyes with slight brows
#         self.draw_eye_normal_with_brow(draw, self.left_eye_x, self.eye_y, brow_height=0)
#         self.draw_eye_normal_with_brow(draw, self.right_eye_x, self.eye_y, brow_height=0)
        
#         # Neutral mouth
#         self.draw_mouth_neutral(draw, self.width//2, self.mouth_y)
        
#         return img
    
#     def generate_thinking(self, variant=1):
#         """Thinking expressions with raised eyebrows"""
#         img, draw = self.create_canvas()
        
#         # Eyes with raised eyebrows
#         brow_raise = -10 if variant <= 2 else -5
#         self.draw_eye_normal_with_brow(draw, self.left_eye_x, self.eye_y, brow_height=brow_raise)
#         self.draw_eye_normal_with_brow(draw, self.right_eye_x, self.eye_y, brow_height=brow_raise)
        
#         # Mouth variations
#         if variant in [1, 3]:
#             self.draw_mouth_neutral(draw, self.width//2, self.mouth_y, width=100)
#         else:
#             self.draw_mouth_smile(draw, self.width//2, self.mouth_y, intensity=0.5)
        
#         return img
    
#     def generate_speaking(self, variant=1):
#         """Speaking expressions with open mouth"""
#         img, draw = self.create_canvas()
        
#         # Normal eyes
#         self.draw_eye_normal_with_brow(draw, self.left_eye_x, self.eye_y, brow_height=0)
#         self.draw_eye_normal_with_brow(draw, self.right_eye_x, self.eye_y, brow_height=0)
        
#         # Speaking mouth variations
#         if variant == 1:
#             self.draw_mouth_open_speaking(draw, self.width//2, self.mouth_y, openness=0.8)
#         elif variant == 2:
#             self.draw_mouth_smile(draw, self.width//2, self.mouth_y, intensity=0.7)
#         else:
#             self.draw_mouth_wide_open(draw, self.width//2, self.mouth_y)
        
#         return img
    
#     def generate_listening(self, variant=1):
#         """Listening expressions with closed/happy eyes"""
#         img, draw = self.create_canvas()
        
#         # Closed/happy eyes
#         if variant == 1:
#             self.draw_eye_happy_closed(draw, self.left_eye_x, self.eye_y)
#             self.draw_eye_happy_closed(draw, self.right_eye_x, self.eye_y)
#         else:
#             self.draw_eye_closed(draw, self.left_eye_x, self.eye_y)
#             self.draw_eye_closed(draw, self.right_eye_x, self.eye_y)
        
#         # Smile
#         self.draw_mouth_smile(draw, self.width//2, self.mouth_y, intensity=0.7)
        
#         return img
    
#     def generate_capturing(self):
#         """Surprised/capturing expression"""
#         img, draw = self.create_canvas()
        
#         # Surprised eyes
#         self.draw_eye_surprised(draw, self.left_eye_x, self.eye_y)
#         self.draw_eye_surprised(draw, self.right_eye_x, self.eye_y)
        
#         # Small O mouth
#         self.draw_mouth_small_o(draw, self.width//2, self.mouth_y)
        
#         return img
    
#     def generate_happy(self, variant=1):
#         """Happy expressions"""
#         img, draw = self.create_canvas()
        
#         if variant == 1:
#             # Happy with open eyes
#             self.draw_eye_normal_with_brow(draw, self.left_eye_x, self.eye_y, brow_height=0)
#             self.draw_eye_normal_with_brow(draw, self.right_eye_x, self.eye_y, brow_height=0)
#         else:
#             # Happy with closed eyes
#             self.draw_eye_happy_closed(draw, self.left_eye_x, self.eye_y)
#             self.draw_eye_happy_closed(draw, self.right_eye_x, self.eye_y)
        
#         # Big smile
#         self.draw_mouth_smile(draw, self.width//2, self.mouth_y, intensity=1.2)
        
#         return img
    
#     def generate_sleepy(self):
#         """Sleepy expression"""
#         img, draw = self.create_canvas()
        
#         # Sleepy eyes
#         self.draw_eye_sleepy(draw, self.left_eye_x, self.eye_y)
#         self.draw_eye_sleepy(draw, self.right_eye_x, self.eye_y)
        
#         # Neutral mouth
#         self.draw_mouth_neutral(draw, self.width//2, self.mouth_y, width=80)
        
#         return img
    
#     def generate_confused(self):
#         """Confused expression"""
#         img, draw = self.create_canvas()
        
#         # One eyebrow up, one normal
#         self.draw_eye_normal_with_brow(draw, self.left_eye_x, self.eye_y, brow_height=-10)
#         self.draw_eye_normal_with_brow(draw, self.right_eye_x, self.eye_y, brow_height=5)
        
#         # Neutral or slightly curved mouth
#         self.draw_mouth_neutral(draw, self.width//2, self.mouth_y, width=70)
        
#         return img
    
#     def generate_sad(self):
#         """Sad expression"""
#         img, draw = self.create_canvas()
        
#         # Eyes with lowered brows
#         self.draw_eye_normal_with_brow(draw, self.left_eye_x, self.eye_y, brow_height=10)
#         self.draw_eye_normal_with_brow(draw, self.right_eye_x, self.eye_y, brow_height=10)
        
#         # Downturned mouth (inverted smile)
#         width = 80
#         height = 25
#         bbox = [self.width//2 - width//2, self.mouth_y - height//2, 
#                self.width//2 + width//2, self.mouth_y + height//2]
#         draw.arc(bbox, start=180, end=0, fill=self.colors["primary"], width=8)
        
#         return img
    
#     def generate_excited(self):
#         """Excited expression"""
#         img, draw = self.create_canvas()
        
#         # Wide surprised eyes
#         self.draw_eye_surprised(draw, self.left_eye_x, self.eye_y)
#         self.draw_eye_surprised(draw, self.right_eye_x, self.eye_y)
        
#         # Wide open smiling mouth
#         self.draw_mouth_open_speaking(draw, self.width//2, self.mouth_y, openness=1.2)
        
#         return img
    
#     def generate_processing(self, variant=1):
#         """Processing/computing expression"""
#         img, draw = self.create_canvas()
        
#         # One eye closed, one open (or both with different states)
#         if variant == 1:
#             self.draw_eye_normal_with_brow(draw, self.left_eye_x, self.eye_y, brow_height=-5)
#             self.draw_eye_closed(draw, self.right_eye_x, self.eye_y)
#         else:
#             self.draw_eye_sleepy(draw, self.left_eye_x, self.eye_y, width=50)
#             self.draw_eye_sleepy(draw, self.right_eye_x, self.eye_y, width=50)
        
#         # Neutral mouth
#         self.draw_mouth_neutral(draw, self.width//2, self.mouth_y, width=90)
        
#         return img
    
#     def generate_all_expressions(self, output_dir="robot_expressions"):
#         """Generate all expression variations"""
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)
        
#         expressions = {
#             "idle_01": self.generate_idle,
            
#             "thinking_01": lambda: self.generate_thinking(1),
#             "thinking_02": lambda: self.generate_thinking(2),
#             "thinking_03": lambda: self.generate_thinking(3),
#             "thinking_04": lambda: self.generate_thinking(4),
            
#             "speaking_01": lambda: self.generate_speaking(1),
#             "speaking_02": lambda: self.generate_speaking(2),
#             "speaking_03": lambda: self.generate_speaking(3),
            
#             "listening_01": lambda: self.generate_listening(1),
#             "listening_02": lambda: self.generate_listening(2),
            
#             "capturing_01": self.generate_capturing,
            
#             "happy_01": lambda: self.generate_happy(1),
#             "happy_02": lambda: self.generate_happy(2),
            
#             "sleepy_01": self.generate_sleepy,
            
#             "confused_01": self.generate_confused,
            
#             "sad_01": self.generate_sad,
            
#             "excited_01": self.generate_excited,
            
#             "processing_01": lambda: self.generate_processing(1),
#             "processing_02": lambda: self.generate_processing(2),
#         }
        
#         print(f"Generating {len(expressions)} expressions with theme '{self.theme}'...")
#         print(f"Output directory: {output_dir}/")
#         print("-" * 50)
        
#         for name, generator in expressions.items():
#             img = generator()
#             filename = f"{output_dir}/{name}.png"
#             img.save(filename, "PNG", quality=100)
#             print(f"✓ Generated: {filename}")
        
#         print("-" * 50)
#         print(f"✅ Complete! Generated {len(expressions)} expressions")
#         print(f"📁 Location: {os.path.abspath(output_dir)}/")


# def main():
#     """Main function with example usage"""
#     import argparse
    
#     parser = argparse.ArgumentParser(description="Generate robot face expressions")
#     parser.add_argument("--theme", default="default", 
#                        choices=["default", "blue", "pink", "orange", "purple"],
#                        help="Color theme for expressions")
#     parser.add_argument("--output", default="robot_expressions",
#                        help="Output directory")
#     parser.add_argument("--width", type=int, default=320,
#                        help="Image width")
#     parser.add_argument("--height", type=int, default=240,
#                        help="Image height")
    
#     args = parser.parse_args()
    
#     # Create generator
#     generator = RobotFaceGenerator(
#         width=args.width, 
#         height=args.height,
#         theme=args.theme
#     )
    
#     # Generate all expressions
#     generator.generate_all_expressions(output_dir=args.output)
    
#     print("\n🎨 Available themes:")
#     for theme in generator.themes.keys():
#         print(f"   - {theme}")
    
#     print("\n💡 Usage examples:")
#     print("   python robot_expressions_generator.py --theme blue")
#     print("   python robot_expressions_generator.py --theme pink --output my_robot_faces")
#     print("   python robot_expressions_generator.py --width 640 --height 480")


# if __name__ == "__main__":
#     main()

import cairo
import os
import math

# --- Cài đặt Hằng số ---
WIDTH = 320
HEIGHT = 240
# Màu nền: Xanh mint nhạt (#C1F0C1)
BG_COLOR = (193/255, 240/255, 193/255)
# Màu mắt và miệng (Đen chuẩn)
FACE_COLOR = (0, 0, 0)
OUTPUT_DIR = "robot_expressions"
MATRIX_OUTPUT_FILE = "robot_expression_matrix.png"

# Đảm bảo thư mục đầu ra tồn tại
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- Các Hàm Vẽ vector ---

def set_background(ctx, color):
    ctx.set_source_rgb(*color)
    ctx.paint()

def draw_round_eye(ctx, x, y, size=18, color=FACE_COLOR, angle=0):
    ctx.save()
    ctx.set_source_rgb(*color)
    ctx.translate(x, y)
    ctx.rotate(angle)
    ctx.arc(0, 0, size, 0, 2 * math.pi)
    ctx.fill()
    ctx.restore()

def draw_deadpan_eye(ctx, x, y, width=40, height=18, color=FACE_COLOR, angle=0):
    ctx.save()
    ctx.set_source_rgb(*color)
    ctx.set_line_width(3) 
    ctx.translate(x, y)
    ctx.rotate(angle)
    ctx.rectangle(-width/2, -height/2, width, height)
    ctx.stroke()
    ctx.arc(0, 0, height/2.5, 0, 2 * math.pi)
    ctx.fill()
    ctx.restore()

def draw_closed_eye_down_curve(ctx, x, y, width=36, height=12, color=FACE_COLOR):
    ctx.save()
    ctx.set_source_rgb(*color)
    ctx.set_line_width(4)
    ctx.move_to(x - width/2, y)
    ctx.curve_to(x - width/4, y + height, x + width/4, y + height, x + width/2, y)
    ctx.stroke()
    ctx.restore()

def draw_closed_eye_up_curve(ctx, x, y, width=36, height=12, color=FACE_COLOR):
    ctx.save()
    ctx.set_source_rgb(*color)
    ctx.set_line_width(4)
    ctx.move_to(x - width/2, y)
    ctx.curve_to(x - width/4, y - height, x + width/4, y - height, x + width/2, y)
    ctx.stroke()
    ctx.restore()

def draw_smile_mouth_small(ctx, x, y, width=60, height=20, color=FACE_COLOR):
    ctx.save()
    ctx.set_source_rgb(*color)
    ctx.set_line_width(4)
    ctx.move_to(x - width/2, y)
    ctx.curve_to(x - width/4, y + height, x + width/4, y + height, x + width/2, y)
    ctx.stroke()
    ctx.restore()

def draw_smile_mouth_wide(ctx, x, y, width=100, height=35, color=FACE_COLOR):
    ctx.save()
    ctx.set_source_rgb(*color)
    ctx.set_line_width(4)
    ctx.move_to(x - width/2, y)
    ctx.curve_to(x - width/4, y + height, x + width/4, y + height, x + width/2, y)
    ctx.stroke()
    ctx.restore()

def draw_talk_mouth_open(ctx, x, y, width=80, height=35, color=FACE_COLOR):
    ctx.save()
    ctx.set_source_rgb(*color)
    ctx.set_line_width(3)
    corner_radius = height/2
    ctx.new_sub_path()
    ctx.arc(x - width/2 + corner_radius, y - height/2 + corner_radius, corner_radius, math.pi, 3*math.pi/2)
    ctx.arc(x + width/2 - corner_radius, y - height/2 + corner_radius, corner_radius, 3*math.pi/2, 2*math.pi)
    ctx.arc(x + width/2 - corner_radius, y + height/2 - corner_radius, corner_radius, 0, math.pi/2)
    ctx.arc(x - width/2 + corner_radius, y + height/2 - corner_radius, corner_radius, math.pi/2, math.pi)
    ctx.close_path()
    ctx.stroke()
    ctx.set_source_rgb(1, 1, 1)
    ctx.rectangle(x - width/2.5, y - height/2 + 2, width/1.25, 4)
    ctx.fill()
    ctx.restore()

def draw_straight_mouth(ctx, x, y, width=60, color=FACE_COLOR):
    ctx.save()
    ctx.set_source_rgb(*color)
    ctx.set_line_width(3)
    ctx.move_to(x - width/2, y)
    ctx.line_to(x + width/2, y)
    ctx.stroke()
    ctx.restore()

def draw_o_mouth(ctx, x, y, radius=20, color=FACE_COLOR):
    ctx.save()
    ctx.set_source_rgb(*color)
    ctx.set_line_width(3)
    ctx.arc(x, y, radius, 0, 2 * math.pi)
    ctx.stroke()
    ctx.restore()

def draw_pixelated_smile(ctx, x, y, width=60, size=5, color=FACE_COLOR):
    ctx.save()
    ctx.set_source_rgb(*color)
    for i in range(-5, 6, 2):
        ctx.rectangle(x + i*size, y - size/2, size, size)
    ctx.fill()
    ctx.restore()

# Khai báo spacing ở đây để các hàm bên dưới nhận diện được
spacing_between_eyes = 120

def draw_blush_marks(ctx, x, y, spacing=60, size=15, color=(1, 0.7, 0.7)):
    ctx.save()
    ctx.set_source_rgb(*color)
    draw_round_eye(ctx, x - spacing/2, y, size=size, color=color)
    draw_round_eye(ctx, x + spacing/2, y, size=size, color=color)
    ctx.restore()

def draw_heart_eyes(ctx, x, y, size=20, color=(1, 0.1, 0.1)):
    ctx.save()
    ctx.set_source_rgb(*color)
    
    def draw_heart(ctx, cx, cy, sz):
        ctx.save()
        ctx.translate(cx, cy)
        ctx.set_source_rgb(*color)
        ctx.move_to(0, 1.2 * sz)
        ctx.curve_to(-2.5 * sz, -1.2 * sz, sz, -1.2 * sz, 0, sz)
        ctx.curve_to(-sz, -1.2 * sz, 2.5 * sz, -1.2 * sz, 0, 1.2 * sz)
        ctx.fill()
        ctx.restore()

    draw_heart(ctx, x - spacing_between_eyes/2, y, size)
    draw_heart(ctx, x + spacing_between_eyes/2, y, size)
    ctx.restore()

def draw_text(ctx, text, x, y, font_size=16, color=(0.1, 0.1, 0.1)):
    ctx.save()
    ctx.select_font_face("Monospace", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(font_size)
    ctx.set_source_rgb(*color)
    (x_bearing, y_bearing, width, height, x_advance, y_advance) = ctx.text_extents(text)
    ctx.move_to(x - width/2, y + height/2)
    ctx.show_text(text)
    ctx.restore()

# --- Định nghĩa ma trận các biểu cảm ---
expressions = [
    ("idle_neutral.png", lambda ctx: (draw_deadpan_eye(ctx, WIDTH/2 - 60, HEIGHT/2 - 20), draw_deadpan_eye(ctx, WIDTH/2 + 60, HEIGHT/2 - 20), draw_straight_mouth(ctx, WIDTH/2, HEIGHT/2 + 30))),
    ("happy_small.png", lambda ctx: (draw_round_eye(ctx, WIDTH/2 - 60, HEIGHT/2 - 20), draw_round_eye(ctx, WIDTH/2 + 60, HEIGHT/2 - 20), draw_smile_mouth_small(ctx, WIDTH/2, HEIGHT/2 + 30))),
    ("excited_wide.png", lambda ctx: (draw_round_eye(ctx, WIDTH/2 - 60, HEIGHT/2 - 20), draw_round_eye(ctx, WIDTH/2 + 60, HEIGHT/2 - 20), draw_smile_mouth_wide(ctx, WIDTH/2, HEIGHT/2 + 30))),
    ("talk_open_wide.png", lambda ctx: (draw_round_eye(ctx, WIDTH/2 - 60, HEIGHT/2 - 20), draw_round_eye(ctx, WIDTH/2 + 60, HEIGHT/2 - 20), draw_talk_mouth_open(ctx, WIDTH/2, HEIGHT/2 + 30))),
    
    ("sad.png", lambda ctx: (draw_deadpan_eye(ctx, WIDTH/2 - 60, HEIGHT/2 - 20), draw_deadpan_eye(ctx, WIDTH/2 + 60, HEIGHT/2 - 20), draw_closed_eye_up_curve(ctx, WIDTH/2, HEIGHT/2 + 30))),
    ("surprised.png", lambda ctx: (draw_round_eye(ctx, WIDTH/2 - 65, HEIGHT/2 - 20, size=22), draw_round_eye(ctx, WIDTH/2 + 65, HEIGHT/2 - 20, size=22), draw_o_mouth(ctx, WIDTH/2, HEIGHT/2 + 30))),
    ("wink_friendly.png", lambda ctx: (draw_round_eye(ctx, WIDTH/2 - 60, HEIGHT/2 - 20), draw_closed_eye_down_curve(ctx, WIDTH/2 + 60, HEIGHT/2 - 20), draw_smile_mouth_small(ctx, WIDTH/2, HEIGHT/2 + 30))),
    ("friendly_blink_both.png", lambda ctx: (draw_closed_eye_down_curve(ctx, WIDTH/2 - 60, HEIGHT/2 - 20), draw_closed_eye_down_curve(ctx, WIDTH/2 + 60, HEIGHT/2 - 20), draw_smile_mouth_small(ctx, WIDTH/2, HEIGHT/2 + 30))),
    
    ("maintenance_mode.png", lambda ctx: (draw_deadpan_eye(ctx, WIDTH/2 - 60, HEIGHT/2 - 20, width=30, height=12, angle=0.1), draw_deadpan_eye(ctx, WIDTH/2 + 60, HEIGHT/2 - 20, width=30, height=12, angle=-0.1), draw_pixelated_smile(ctx, WIDTH/2, HEIGHT/2 + 30), draw_text(ctx, "SERVICE", WIDTH/2, HEIGHT/2 + 55, font_size=12))),
    ("blushing.png", lambda ctx: (draw_round_eye(ctx, WIDTH/2 - 60, HEIGHT/2 - 20), draw_round_eye(ctx, WIDTH/2 + 60, HEIGHT/2 - 20), draw_smile_mouth_small(ctx, WIDTH/2, HEIGHT/2 + 30), draw_blush_marks(ctx, WIDTH/2, HEIGHT/2 + 10))),
    ("heart_eyes.png", lambda ctx: (draw_heart_eyes(ctx, WIDTH/2, HEIGHT/2 - 20), draw_smile_mouth_small(ctx, WIDTH/2, HEIGHT/2 + 30))),
    ("tired_sleepy.png", lambda ctx: (draw_closed_eye_up_curve(ctx, WIDTH/2 - 60, HEIGHT/2 - 20), draw_closed_eye_up_curve(ctx, WIDTH/2 + 60, HEIGHT/2 - 20), draw_straight_mouth(ctx, WIDTH/2, HEIGHT/2 + 30))),
    
    ("confused.png", lambda ctx: (draw_deadpan_eye(ctx, WIDTH/2 - 60, HEIGHT/2 - 20, angle=-0.1), draw_deadpan_eye(ctx, WIDTH/2 + 60, HEIGHT/2 - 20, angle=0.15), draw_straight_mouth(ctx, WIDTH/2, HEIGHT/2 + 30))),
    ("angry.png", lambda ctx: (draw_round_eye(ctx, WIDTH/2 - 60, HEIGHT/2 - 20), draw_round_eye(ctx, WIDTH/2 + 60, HEIGHT/2 - 20), draw_closed_eye_up_curve(ctx, WIDTH/2, HEIGHT/2 + 25, width=50, height=8))),
    ("laughing.png", lambda ctx: (draw_round_eye(ctx, WIDTH/2 - 65, HEIGHT/2 - 20), draw_round_eye(ctx, WIDTH/2 + 65, HEIGHT/2 - 20), draw_smile_mouth_wide(ctx, WIDTH/2, HEIGHT/2 + 35))),
    ("wink_deadpan.png", lambda ctx: (draw_round_eye(ctx, WIDTH/2 - 60, HEIGHT/2 - 20), draw_deadpan_eye(ctx, WIDTH/2 + 60, HEIGHT/2 - 20, width=30, height=12), draw_straight_mouth(ctx, WIDTH/2, HEIGHT/2 + 30)))
]

# --- Chạy script tạo ra các hình ảnh ---

print(f"Bắt đầu tạo {len(expressions)} biểu cảm...")

matrix_cols = 4
matrix_rows = 4
matrix_width = matrix_cols * WIDTH
matrix_height = matrix_rows * (HEIGHT + 30) 

matrix_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, matrix_width, matrix_height)
matrix_ctx = cairo.Context(matrix_surface) # <-- Đã sửa ở đây
set_background(matrix_ctx, BG_COLOR)

for idx, (filename, draw_func) in enumerate(expressions):
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
    ctx = cairo.Context(surface) # <-- Đã sửa ở đây
    set_background(ctx, BG_COLOR)
    draw_func(ctx) 
    
    full_path = os.path.join(OUTPUT_DIR, filename)
    surface.write_to_png(full_path)
    print(f"Đã tạo: {full_path}")
    
    col = idx % matrix_cols
    row = idx // matrix_cols
    x_offset = col * WIDTH
    y_offset = row * (HEIGHT + 30)
    
    matrix_ctx.set_source_rgb(*BG_COLOR)
    matrix_ctx.rectangle(x_offset, y_offset, WIDTH, HEIGHT)
    matrix_ctx.fill()
    
    matrix_ctx.set_source_surface(surface, x_offset, y_offset)
    matrix_ctx.paint()
    
    label = filename.replace(".png", "")
    draw_text(matrix_ctx, label, x_offset + WIDTH/2, y_offset + HEIGHT + 15, font_size=12, color=(0.2, 0.2, 0.2))

matrix_surface.write_to_png(MATRIX_OUTPUT_FILE)
print(f"--- Hoàn tất! Đã tạo ma trận biểu cảm tại: {MATRIX_OUTPUT_FILE} ---")