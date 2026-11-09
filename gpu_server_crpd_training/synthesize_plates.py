#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合成仿真中国车牌 (用于识别测试, 不是训练数据)
GB 标准: 小型车牌 440mm x 140mm
布局: 省(大字) + · + 5位字母数字 (共 7 字符 + 1 分隔点)
"""
import os
import cv2
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ===== 字符集 =====
PROVINCES = ["皖","沪","津","渝","冀","晋","蒙","辽","吉","黑","苏","浙","京","闽","赣","鲁","豫","鄂","湘","粤","桂","琼","川","贵","云","藏","陕","甘","青","宁","新"]
ALPHAS = ["A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z"]
DIGITS = "0123456789"
ALNUM = ALPHAS + list(DIGITS)

# ===== 输出 =====
OUT_DIR = "E:/xuhrWorkProject/floating_ocr_ai_paddle/docs/synthetic_plates_100"
os.makedirs(OUT_DIR, exist_ok=True)


# ===== 字体 =====
def load_font(size):
    """加载字体 (优先微软雅黑, 渲染更干净)"""
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",       # 微软雅黑 (优先)
        "C:/Windows/Fonts/simhei.ttf",     # 黑体
        "C:/Windows/Fonts/simsun.ttc",     # 宋体
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size)
    return ImageFont.load_default()


# ===== 车牌号生成 =====
def rand_str(charset, n):
    return "".join(random.choice(charset) for _ in range(n))

def gen_plate(kind):
    """生成车牌号 + 颜色类型"""
    if kind == "蓝":
        plate = random.choice(PROVINCES) + random.choice(ALPHAS) + rand_str(ALNUM, 5)
        return plate, "蓝"
    elif kind == "黄":
        plate = random.choice(PROVINCES) + random.choice(ALPHAS) + rand_str(ALNUM, 5)
        return plate, "黄"
    elif kind == "绿":
        prov = random.choice(PROVINCES)
        alpha = random.choice(ALPHAS)
        df = random.choice("DF")
        if random.random() < 0.5:
            return prov + alpha + df + rand_str(ALNUM, 4), "绿"
        else:
            return prov + alpha + rand_str(ALNUM, 4) + df, "绿"
    elif kind == "警":
        plate = random.choice(PROVINCES) + random.choice(ALPHAS) + rand_str(DIGITS, 4) + "警"
        return plate, "白"
    elif kind == "挂":
        plate = random.choice(PROVINCES) + random.choice(ALPHAS) + rand_str(ALNUM, 5) + "挂"
        return plate, "黄"
    elif kind == "学":
        plate = random.choice(PROVINCES) + random.choice(ALPHAS) + rand_str(ALNUM, 4) + "学"
        return plate, "黄"
    elif kind == "使":
        plate = random.choice(PROVINCES) + random.choice(ALPHAS) + rand_str(ALNUM, 4) + "使"
        return plate, "黑"
    elif kind == "领":
        plate = random.choice(PROVINCES) + random.choice(ALPHAS) + rand_str(ALNUM, 4) + "领"
        return plate, "黑"
    elif kind == "港澳":
        alpha = random.choice(ALPHAS)
        return "粤" + alpha + "Z" + rand_str(DIGITS, 4), "黑"
    else:
        return gen_plate("蓝")


# ===== 绘图工具 =====
def rounded_rect(draw, xy, radius, fill, outline=None, width=1):
    """画圆角矩形"""
    x0, y0, x1, y1 = xy
    r = radius
    # 四个圆角
    draw.pieslice([x0, y0, x0+2*r, y0+2*r], 180, 270, fill=fill)
    draw.pieslice([x1-2*r, y0, x1, y0+2*r], 270, 360, fill=fill)
    draw.pieslice([x0, y1-2*r, x0+2*r, y1], 90, 180, fill=fill)
    draw.pieslice([x1-2*r, y1-2*r, x1, y1], 0, 90, fill=fill)
    # 四个矩形
    draw.rectangle([x0+r, y0, x1-r, y1], fill=fill)
    draw.rectangle([x0, y0+r, x0+r, y1-r], fill=fill)
    draw.rectangle([x1-r, y0+r, x1, y1-r], fill=fill)
    if outline:
        draw.arc([x0, y0, x0+2*r, y0+2*r], 180, 270, fill=outline, width=width)
        draw.arc([x1-2*r, y0, x1, y0+2*r], 270, 360, fill=outline, width=width)
        draw.arc([x0, y1-2*r, x0+2*r, y1], 90, 180, fill=outline, width=width)
        draw.arc([x1-2*r, y1-2*r, x1, y1], 0, 90, fill=outline, width=width)
        draw.line([x0+r, y0, x1-r, y0], fill=outline, width=width)
        draw.line([x0+r, y1, x1-r, y1], fill=outline, width=width)
        draw.line([x0, y0+r, x0, y1-r], fill=outline, width=width)
        draw.line([x1, y0+r, x1, y1-r], fill=outline, width=width)


def make_gradient_bg(w, h, color_type):
    """制作带金属渐变的底板"""
    # 基础颜色 (RGB)
    color_map = {
        "蓝": (0, 92, 175),    # #005CAF
        "黄": (255, 206, 0),   # #FFCE00
        "绿": (0, 106, 55),    # #006A37
        "白": (240, 240, 240), # 白色
        "黑": (20, 20, 20),    # 黑色
    }
    base = color_map[color_type]

    # 创建带垂直渐变的图像 (中间亮两边暗)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        # 垂直渐变: 顶部略暗, 中间最亮, 底部略暗
        dy = abs(y - h/2) / (h/2)
        factor = 1.0 - dy * 0.15  # 边缘减暗15%
        img[y, :] = [min(255, int(c * factor)) for c in base]

    # 轻微水平渐变 (模拟金属反光)
    for x in range(w):
        dx = abs(x - w/2) / (w/2)
        factor = 1.0 - dx * 0.08
        img[:, x] = (img[:, x].astype(np.float32) * factor).astype(np.uint8)

    return img


def draw_screw(draw, cx, cy, r, dark_color):
    """画螺丝孔"""
    # 外圈
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=dark_color, outline=dark_color)
    # 内圈
    r2 = int(r * 0.4)
    draw.ellipse([cx-r2, cy-r2, cx+r2, cy+r2], fill=(60, 60, 60))


def draw_plate_image(plate_text, color_type, W=440, H=140):
    """绘制真实感车牌图像

    真实车牌布局 (GB/T 36045-2018, 440×140mm):
    ─────────────────────────────────────────┐
    │ ○  省  ·  A  B  C  D  E  F           ○  │
    │                                         │
    │ ○                                      ○  │
    └─────────────────────────────────────────┘
    省字符: 约55px宽, 居中于 [20, 75]
    分隔点: 圆点, 直径约10px, 居中于 [80, 105]
    后6字符: 每个约55px宽, 居中于各自区间
    字符高度: 约70px, 垂直居中
    """
    # 1. 金属渐变底板
    bg = make_gradient_bg(W, H, color_type)
    img = Image.fromarray(bg)
    draw = ImageDraw.Draw(img)

    # 2. 文字颜色
    text_colors = {
        "蓝": (255, 255, 255),
        "黄": (0, 0, 0),
        "绿": (255, 255, 255),
        "白": (0, 0, 0),
        "黑": (255, 255, 255),
    }
    text_color = text_colors[color_type]
    dark_color = (30, 30, 30) if color_type in ("黄", "白") else (180, 180, 180)

    # 3. 边框
    rounded_rect(draw, (2, 2, W-3, H-3), radius=8, fill=None,
                 outline=dark_color, width=3)

    # 4. 字体尺寸
    # 字符高度目标约 68px (留出上下边距)
    font_prov = load_font(62)   # 省份
    font_char = load_font(60)   # 后续字符

    base_color = make_gradient_bg(W, H, color_type).astype(np.uint8)

    # 5. 字符布局 (绝对定位)
    # 省字符: 区间 [15, 72], 宽57px
    # 分隔点: 区间 [78, 100], 宽22px
    # 后6字符: 每个区间约 55px
    # 总宽: 57 + 22 + 6*55 = 57 + 22 + 330 = 409 (两边留 15px)

    y_center = H // 2  # 70
    char_height = 68

    def render_char_to_canvas(text, font, bg_color, text_color):
        """把单个字符渲染到独立画布, 裁掉透明区域, 返回干净的字符像素"""
        # 用一个大画布渲染
        canvas = Image.new('RGBA', (200, 200), (0, 0, 0, 0))
        cd = ImageDraw.Draw(canvas)
        bbox = cd.textbbox((0, 0), text, font=font)
        ox, oy = bbox[0], bbox[1]
        cd.text((0, 0), text, fill=text_color, font=font)
        # 裁掉透明区域 (保留非透明像素)
        bbox2 = canvas.getbbox()  # (x0,y0,x1,y1) 非透明区域
        if bbox2 is None:
            return canvas.convert('RGB')
        cropped = canvas.crop(bbox2)
        # 加 4px 内边距
        pad = 4
        w2, h2 = cropped.size
        result = Image.new('RGBA', (w2 + pad*2, h2 + pad*2), (0, 0, 0, 0))
        result.paste(cropped, (pad, pad))
        return result.convert('RGB')

    # ===== 逐个字符渲染并粘贴 =====
    # 省字符
    char_canvas = render_char_to_canvas(plate_text[0], font_prov,
                                         base_color, text_color)
    cw, ch_ = char_canvas.size
    x = int((10 + 65 - cw) / 2)
    y = int(y_center - ch_ / 2)
    img.paste(char_canvas, (x, y), char_canvas.convert('RGBA'))

    # 分隔点
    x_dot = 82
    dot_r = 5
    draw.ellipse([x_dot - dot_r, y_center - dot_r,
                  x_dot + dot_r, y_center + dot_r], fill=text_color)

    # 后续字符
    rest = plate_text[1:]
    n = len(rest)
    x_start = 100
    x_end = W - 15
    slot_w = (x_end - x_start) / n
    for i, ch in enumerate(rest):
        f = load_font(50) if ch in "警学挂使领" else font_char
        char_canvas = render_char_to_canvas(ch, f, base_color, text_color)
        cw, ch_ = char_canvas.size
        x_left = int(x_start + i * slot_w)
        x_right = int(x_start + (i + 1) * slot_w)
        x = int((x_left + x_right - cw) / 2)
        y = int(y_center - ch_ / 2)
        img.paste(char_canvas, (x, y), char_canvas.convert('RGBA'))

    # 6. 四角螺丝孔
    screw_r = 4
    for sx, sy in [(16, 16), (W-16, 16), (16, H-16), (W-16, H-16)]:
        draw.ellipse([sx-screw_r, sy-screw_r, sx+screw_r, sy+screw_r],
                     fill=dark_color)
        draw.ellipse([sx-2, sy-2, sx+2, sy+2], fill=(80, 80, 80))

    # 7. 后处理: 噪点 + 轻微老化
    arr = np.array(img).astype(np.float32)
    noise = np.random.randint(-6, 6, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.3))

    return img


# ===== 主流程 =====
def main():
    random.seed(123)
    np.random.seed(123)

    tasks = [
        ("绿", 20),
        ("蓝", 20),
        ("黄", 20),
        ("警", 10),
        # 特殊 30 张: 挂5 + 学5 + 使5 + 领5 + 港澳10
        ("挂", 5),
        ("学", 5),
        ("使", 5),
        ("领", 5),
        ("港澳", 10),
    ]

    idx = 0
    records = []

    for kind, count in tasks:
        for i in range(count):
            plate_text, color_type = gen_plate(kind)
            img = draw_plate_image(plate_text, color_type)
            fn = f"syn_{idx:03d}.jpg"
            img.save(os.path.join(OUT_DIR, fn), quality=95)
            records.append((fn, plate_text, color_type, kind))
            idx += 1

    # 统计
    from collections import defaultdict
    stats = defaultdict(int)
    for _, _, _, k in records:
        stats[k] += 1

    print(f"合成 {len(records)} 张完成")
    print("\n=== 类型分布 ===")
    for k in ["绿", "蓝", "黄", "警", "挂", "学", "使", "领", "港澳"]:
        if k in stats:
            print(f"  {k}: {stats[k]}")

    # 详细列表
    with open(os.path.join(OUT_DIR, "stat.txt"), "w", encoding="utf-8") as f:
        f.write(f"=== 合成 {len(records)} 张仿真车牌 ===\n")
        f.write("编号 | 文件名 | 车牌号 | 颜色 | 类型\n")
        f.write("-" * 75 + "\n")
        for i, (fn, plate, color, kind) in enumerate(records):
            f.write(f"{i+1:3d} | {fn} | {plate:10s} | {color} | {kind}\n")
        f.write(f"\n=== 按类型统计 ===\n")
        for k in ["绿", "蓝", "黄", "警", "挂", "学", "使", "领", "港澳"]:
            if k in stats:
                f.write(f"  {k}: {stats[k]}\n")

    print(f"\n保存到: {OUT_DIR}")


if __name__ == "__main__":
    main()
