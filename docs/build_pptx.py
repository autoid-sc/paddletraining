import collections.abc
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
import os

pptx_path = r"e:\xuhrWorkProject\floating_ocr_ai_paddle\docs\OCR软件整体流程及多场景易用性设计.pptx"
# Writing to slides_v6.pptx
output_path = r"e:\xuhrWorkProject\floating_ocr_ai_paddle\docs\slides_v7.pptx"

prs = Presentation(pptx_path)

# Keep Slide 1 (index 0) intact, clear the rest of the template slides
for i in range(len(prs.slides)-1, 0, -1):
    rId = prs.slides._sldIdLst[i].rId
    prs.part.drop_rel(rId)
    del prs.slides._sldIdLst[i]

# Helper to add a custom slide title (for Blank layout index 6)
def add_slide_title(slide, title_text):
    txBox = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.8))
    tf = txBox.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.0)
    tf.margin_top = Inches(0.0)
    tf.margin_bottom = Inches(0.0)
    p = tf.paragraphs[0]
    p.text = title_text
    p.font.name = 'Microsoft YaHei'
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = RGBColor(3, 105, 161) # 蓝色

# Helper to add structured bullets inside a custom text box
def add_slide_bullets(slide, bullets, left=Inches(0.8), top=Inches(1.5), width=Inches(11.7), height=Inches(4.8)):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.0)
    tf.margin_top = Inches(0.0)
    
    for i, item in enumerate(bullets):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
            
        if isinstance(item, tuple):
            text, level = item
            p.text = text
            p.level = level
        else:
            p.text = item
            p.level = 0
            
        p.font.name = 'Microsoft YaHei'
        p.font.size = Pt(14)
        p.line_spacing = 1.3
        p.space_after = Pt(6)
        
        # Color coding: Level 0 (Bold Blue), Level 1 (Regular slate grey)
        if level == 0:
            p.font.bold = True
            p.font.color.rgb = RGBColor(3, 105, 161) # Deep Blue
            p.space_before = Pt(8)
        else:
            p.font.bold = False
            p.font.color.rgb = RGBColor(71, 85, 105) # Slate Grey

# Helper to add custom text box (completely editable)
def add_custom_text(slide, text, left, top, width, height, font_size=Pt(13), bold=False, color=RGBColor(51, 65, 85), align=PP_ALIGN.CENTER):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    tf.margin_top = Inches(0.01)
    tf.margin_bottom = Inches(0.01)
    tf.margin_left = Inches(0.01)
    tf.margin_right = Inches(0.01)
    p = tf.paragraphs[0]
    p.text = text
    p.alignment = align
    p.font.name = 'Microsoft YaHei'
    p.font.size = font_size
    p.font.bold = bold
    p.font.color.rgb = color
    return txBox

# Helper to add dark code panels (editable)
def add_code_panel(slide, code_text, left, top, width, height, font_size=Pt(10)):
    # Background panel shape
    panel = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    panel.fill.solid()
    panel.fill.fore_color.rgb = RGBColor(30, 41, 59) # #1e293b
    panel.line.color.rgb = RGBColor(71, 85, 105) # #475569
    panel.line.width = Pt(1.0)
    
    # Overlay editable textbox
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    tf.margin_top = Inches(0.08)
    tf.margin_bottom = Inches(0.08)
    tf.margin_left = Inches(0.08)
    tf.margin_right = Inches(0.08)
    
    lines = code_text.split('\n')
    for idx, line in enumerate(lines):
        if idx == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = line
        p.font.name = 'Consolas'
        p.font.size = font_size
        p.font.color.rgb = RGBColor(248, 250, 252) # #f8fafc
        p.line_spacing = 1.1

# Helper to add a clean visual placeholder box for user to drop images/videos
def add_media_placeholder(slide, text, left, top, width, height):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    box.fill.solid()
    box.fill.fore_color.rgb = RGBColor(241, 245, 249) # #f1f5f9
    box.line.color.rgb = RGBColor(148, 163, 184) # #94a3b8
    box.line.width = Pt(1.5)
    
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.alignment = PP_ALIGN.CENTER
    p.font.name = 'Microsoft YaHei'
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = RGBColor(100, 116, 139) # #64748b

# ----------------- SLIDE 1: 封面 (替换标题, 继承原样式) -----------------
slide1 = prs.slides[0]

tf1 = slide1.shapes[1].text_frame
tf1.text = "Paddle模型训练和场景应用"
p1 = tf1.paragraphs[0]
p1.font.name = 'Microsoft YaHei'
p1.font.size = Pt(36)
p1.font.bold = True
p1.font.color.rgb = RGBColor(3, 105, 161)
p1.alignment = PP_ALIGN.CENTER

tf0 = slide1.shapes[0].text_frame
tf0.text = "应用开发：徐浩然\nAndroid 移动端 OCR 落地与多模型对比实战"
p0 = tf0.paragraphs[0]
p0.font.name = 'Microsoft YaHei'
p0.font.size = Pt(14)
p0.font.color.rgb = RGBColor(71, 85, 105)
p0.alignment = PP_ALIGN.CENTER

# ----------------- SLIDE 2: 目录 -----------------
slide2 = prs.slides.add_slide(prs.slide_layouts[6])
add_slide_title(slide2, "目录")
add_slide_bullets(slide2, [
    ("第一部分：引言 & 核心原理解析", 0),
    ("第二部分：PPOCRv6 特性 & 模型差异对比", 0),
    ("第三部分：模型微调与训练实战", 0),
    ("第四部分：端侧转换、部署与实测效果展示", 0),
    ("第五部分：总结 & 迭代优化方向", 0)
])

# ----------------- SLIDE 3: 一、引言 —— 移动端 OCR 核心挑战与诉求 -----------------
slide3 = prs.slides.add_slide(prs.slide_layouts[6])
add_slide_title(slide3, "一、引言 —— 移动端 OCR 核心挑战与诉求")
add_slide_bullets(slide3, [
    ("1. 移动端部署的三大“拦路虎”", 0),
    ("包体积限制：常规模型体积过大，严重影响 App 安装包瘦身和下载转化率", 1),
    ("内存与 OOM 风险：端侧可用内存受限，OCR 推理时易发生内存抖动导致系统强杀", 1),
    ("持续扫码的发热降频：扫描枪高频连续扫码时，CPU 持续满载，导致发热并触发降频卡顿", 1),
    ("2. 精度与业务的冲突", 0),
    ("对于 VIN 码、反光钢印、特殊工业标签等垂直格式，通用 OCR 的默认识别率普遍不标", 1),
    ("解决方案：构建一套在端侧兼顾极速与高精度，且支持低成本小样本微调 of OCR 落地管道", 1)
])

# ----------------- SLIDE 4: 一、引言 —— 深度学习 OCR 原理通俗化 -----------------
slide4 = prs.slides.add_slide(prs.slide_layouts[6])
add_slide_title(slide4, "一、引言 —— 深度学习 OCR 原理通俗化")
add_slide_bullets(slide4, [
    ("1. 文本检测（DBNet）：找位置的“智能魔棒”", 0),
    ("在像素级寻找文字轮廓边界，就像 Photoshop 的魔棒或套索工具一样", 1),
    ("完美适应斜角、侧视和弯曲文本，能精准贴合边缘把文本“抠出来”并做拉直切片", 1),
    ("2. 文本识别（SVTR_LCNet）：读文字的“阅读专家”", 0),
    ("拿到切片小纸条后，像看书一样，一字句读取并将其翻译成电子文本字符串", 1),
    ("3. 移动端优化核心：移除了无法并行计算的 RNN 算子", 0),
    ("传统 RNN 像“排队买票”，后字依赖前字，在多核 CPU 上无法并发计算，延迟极高", 1),
    ("PP-OCRv4/v6 采用视觉 Transformer（SVTR）和轻量卷积（LCNet），支持多线程并行加速", 1)
])

# ----------------- SLIDE 5: 一、引言 —— 主流 OCR 方案对比 -----------------
slide5 = prs.slides.add_slide(prs.slide_layouts[6])
add_slide_title(slide5, "一、引言 —— 主流 OCR 方案对比")

# Column 1: 开箱即用方案
c1 = slide5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(5.6), Inches(4.8))
c1.fill.solid()
c1.fill.fore_color.rgb = RGBColor(255, 255, 255)
c1.line.color.rgb = RGBColor(203, 213, 225)
c1.line.width = Pt(1.5)
add_custom_text(slide5, "开箱即用方案", Inches(1.0), Inches(1.7), Inches(5.2), Inches(0.4), font_size=Pt(15), bold=True, color=RGBColor(3, 105, 161), align=PP_ALIGN.LEFT)
add_custom_text(slide5, 
    "• Google ML Kit (On-device)\n"
    "  - 优势：开箱即用免训练，通过 Play 服务分发包体积几近 0MB\n"
    "  - 劣势：黑盒状态无法自定义微调，反光钢印或异形字符识别差\n\n"
    "• Tesseract (Tessdata)\n"
    "  - 优势：老牌开源引擎，易跨平台，简单排版支持好\n"
    "  - 劣势：不支持现代 GPU，暗光、倾斜或污损环境极易出错", 
    Inches(1.0), Inches(2.3), Inches(5.2), Inches(3.8), font_size=Pt(13), align=PP_ALIGN.LEFT)

# Column 2: 深度学习定制方案 (高亮)
c2 = slide5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.9), Inches(1.5), Inches(5.6), Inches(4.8))
c2.fill.solid()
c2.fill.fore_color.rgb = RGBColor(255, 255, 255)
c2.line.color.rgb = RGBColor(2, 132, 199)
c2.line.width = Pt(2.0)
add_custom_text(slide5, "深度学习定制方案", Inches(7.1), Inches(1.7), Inches(5.2), Inches(0.4), font_size=Pt(15), bold=True, color=RGBColor(2, 132, 199), align=PP_ALIGN.LEFT)
add_custom_text(slide5, 
    "• EasyOCR (PyTorch)\n"
    "  - 优势：Python 社区极活跃，学术界算法更新快，识别精度高\n"
    "  - 劣势：模型普遍在 100MB 以上，移动端极易发生内存溢出(OOM)\n\n"
    "• PaddleOCR (PP-OCR) [本案推荐]\n"
    "  - 优势：大模型底座加持，仅 50~100 张图小样本微调即可达 98% 精度\n"
    "  - 优势：轻量至 6MB~30MB，针对 ARM 汇编与 Vulkan 进行深度优化\n"
    "  - 劣势：需自行搭建 Python 环境微调并导出 nb/ncnn 格式", 
    Inches(7.1), Inches(2.3), Inches(5.2), Inches(3.8), font_size=Pt(13), align=PP_ALIGN.LEFT)

# ----------------- SLIDE 6: 二、PPOCRv6 特性 —— 极致轻量化核心选型 -----------------
slide6 = prs.slides.add_slide(prs.slide_layouts[6])
add_slide_title(slide6, "二、PPOCRv6 特性 —— 极致轻量化核心选型")
add_slide_bullets(slide6, [
    ("1. 蒸馏算法升级 (Distillation)", 0),
    ("通过高精度大模型引导轻量级小模型学习，大幅度削减模型体积，同时保留极高的识别准确度", 1),
    ("2. 骨干网络 PP-LCNetV2 升级", 0),
    ("采用更适合 ARM 架构的骨干网络，对基本卷积算子进行汇编级深度重构，前向速度极快", 1),
    ("3. 方向分类器（Cls）与多角度校正", 0),
    ("自动纠正 0°/90°/180°/270° 翻转图像，DBNet 多边形边界点配合 OpenCV 透视仿射拉直", 1),
    ("4. 零字符对齐 CTC 解码", 0),
    ("重塑 CTC 解码机制，彻底改善了紧凑粘连字符（如印刷 VIN 码）多字、漏字问题", 1)
])

# ----------------- SLIDE 7: 二、PPOCRv6 特性 —— Tiny, Small, Medium 区别与对比 -----------------
slide7 = prs.slides.add_slide(prs.slide_layouts[6])
add_slide_title(slide7, "二、PPOCRv6 特性 —— Tiny, Small, Medium 区别与对比")

# Tiny Card
t_card = slide7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(3.6), Inches(4.8))
t_card.fill.solid()
t_card.fill.fore_color.rgb = RGBColor(255, 255, 255)
t_card.line.color.rgb = RGBColor(203, 213, 225)
t_card.line.width = Pt(1.5)
add_custom_text(slide7, "PP-OCRv6 Tiny (极速版)", Inches(0.9), Inches(1.7), Inches(3.4), Inches(0.4), font_size=Pt(14), bold=True, color=RGBColor(3, 105, 161))
add_custom_text(slide7, 
    "• 体积：检测 1.9MB + 识别 4.6MB = 6.5MB\n\n"
    "• 负载：内存与 CPU 占用极低\n\n"
    "• 场景：低端扫码枪、高频视频流连续快速预览，良好光照下耗时最短\n\n"
    "• 局限：对重度倾斜和反光略微敏感", 
    Inches(0.9), Inches(2.2), Inches(3.4), Inches(3.9), font_size=Pt(12), align=PP_ALIGN.LEFT)

# Small Card
s_card = slide7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(4.8), Inches(1.5), Inches(3.6), Inches(4.8))
s_card.fill.solid()
s_card.fill.fore_color.rgb = RGBColor(255, 255, 255)
s_card.line.color.rgb = RGBColor(203, 213, 225)
s_card.line.width = Pt(1.5)
add_custom_text(slide7, "PP-OCRv6 Small (均衡版)", Inches(4.9), Inches(1.7), Inches(3.4), Inches(0.4), font_size=Pt(14), bold=True, color=RGBColor(3, 105, 161))
add_custom_text(slide7, 
    "• 体积：检测 10MB + 识别 21MB = 31MB\n\n"
    "• 负载：中等硬件占用，运行流畅\n\n"
    "• 场景：中端 Android PDA 首选，均衡速度与精度\n\n"
    "• 适用：工业常见轻度污损、反光与小角度畸变扫码", 
    Inches(4.9), Inches(2.2), Inches(3.4), Inches(3.9), font_size=Pt(12), align=PP_ALIGN.LEFT)

# Medium Card
m_card = slide7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.8), Inches(1.5), Inches(3.6), Inches(4.8))
m_card.fill.solid()
m_card.fill.fore_color.rgb = RGBColor(255, 255, 255)
m_card.line.color.rgb = RGBColor(2, 132, 199)
m_card.line.width = Pt(2.0)
add_custom_text(slide7, "PP-OCRv6 Medium (高精版)", Inches(8.9), Inches(1.7), Inches(3.4), Inches(0.4), font_size=Pt(14), bold=True, color=RGBColor(2, 132, 199))
add_custom_text(slide7, 
    "• 体积：检测 62MB + 识别 76.8MB = 138MB\n\n"
    "• 负载：计算密集，CPU 负载高\n\n"
    "• 场景：推荐高端终端或配合 Vulkan GPU 硬件加速\n\n"
    "• 适用：重度斜视、反光钢印、细小密集字符等极端工业环境", 
    Inches(8.9), Inches(2.2), Inches(3.4), Inches(3.9), font_size=Pt(12), align=PP_ALIGN.LEFT)

# ----------------- SLIDE 8: 三、模型微调与训练 —— 流程示意图 -----------------
slide8 = prs.slides.add_slide(prs.slide_layouts[6])
add_slide_title(slide8, "三、模型微调与训练 —— 流程示意图")

# Native Flowchart shapes grid
steps = [
    "1. 收集与标注图片", "2. 提取去重字符集", "3. 修改 YAML 参数", "4. 启动微调训练",
    "8. 多线程/Vulkan调优", "7. Android集成部署", "6. 格式转换 Lite/ncnn", "5. 导出 Inference 模型"
]

row1_y = Inches(2.2)
row2_y = Inches(4.5)
box_w = Inches(2.7)
box_h = Inches(0.8)
col_x = [Inches(0.8), Inches(3.8), Inches(6.8), Inches(9.8)]

for idx, step_name in enumerate(steps):
    if idx < 4:
        c_x = col_x[idx]
        c_y = row1_y
    else:
        c_x = col_x[7 - idx]
        c_y = row2_y
        
    box = slide8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, c_x, c_y, box_w, box_h)
    box.fill.solid()
    if idx == 0 or idx == 7:
        box.fill.fore_color.rgb = RGBColor(254, 243, 199)
        box.line.color.rgb = RGBColor(217, 119, 6)
    else:
        box.fill.fore_color.rgb = RGBColor(224, 242, 254)
        box.line.color.rgb = RGBColor(2, 132, 199)
    box.line.width = Pt(1.5)
    
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = step_name
    p.alignment = PP_ALIGN.CENTER
    p.font.name = 'Microsoft YaHei'
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = RGBColor(3, 105, 161)

# Arrows
add_custom_text(slide8, "➔", Inches(3.5), Inches(2.4), Inches(0.3), Inches(0.4), font_size=Pt(16), color=RGBColor(148, 163, 184))
add_custom_text(slide8, "➔", Inches(6.5), Inches(2.4), Inches(0.3), Inches(0.4), font_size=Pt(16), color=RGBColor(148, 163, 184))
add_custom_text(slide8, "➔", Inches(9.5), Inches(2.4), Inches(0.3), Inches(0.4), font_size=Pt(16), color=RGBColor(148, 163, 184))
add_custom_text(slide8, "▼", Inches(10.85), Inches(3.4), Inches(0.6), Inches(0.4), font_size=Pt(16), color=RGBColor(148, 163, 184))
add_custom_text(slide8, "➔", Inches(3.5), Inches(4.7), Inches(0.3), Inches(0.4), font_size=Pt(16), color=RGBColor(148, 163, 184))
add_custom_text(slide8, "➔", Inches(6.5), Inches(4.7), Inches(0.3), Inches(0.4), font_size=Pt(16), color=RGBColor(148, 163, 184))
add_custom_text(slide8, "➔", Inches(9.5), Inches(4.7), Inches(0.3), Inches(0.4), font_size=Pt(16), color=RGBColor(148, 163, 184))

# ----------------- SLIDE 9: 三、模型微调与训练 —— 数据准备与标注清单 -----------------
slide9 = prs.slides.add_slide(prs.slide_layouts[6])
add_slide_title(slide9, "三、模型微调与训练 —— 数据准备与标注清单")
add_slide_bullets(slide9, [
    ("1. 规范化训练数据结构", 0),
    ("数据存放在 train_data/ 下：rec_train/ 放小图，rec_train_list.txt 存放标注", 1),
    ("2. 标注清单格式规范", 0),
    ("格式：图片相对路径 + Tab 制表符 (\\t) + 真实文本内容", 1),
    ("示例：rec_train/word_001.jpg\\t2022050508:50:54", 1),
    ("示例：rec_train/word_002.jpg\\t14B1C946B162", 1),
    ("⚠️ 警告：路径与标注之间必须是单个 Tab 制表符，严禁空格，否则导致训练崩溃", 1)
])

# ----------------- SLIDE 10: 三、模型微调与训练 —— 自动提取字典与黄金法则 -----------------
slide10 = prs.slides.add_slide(prs.slide_layouts[6])
add_slide_title(slide10, "三、模型微调与训练 —— 自动提取字典与黄金法则")
add_slide_bullets(slide10, [
    ("1. 自动提取字符字典", 0),
    ("运行脚本自动扫描所有标注文件中的文本，提去重字符并保存为专属字典：", 1),
    ("命令：python tools/extract_charset.py", 2),
    ("2. 字典文件的“黄金法则”", 0),
    ("每行仅允许写入一个字符", 1),
    ("字典文件末尾必须多保留一个空白回车行！", 1),
    ("⚠️ 注意：如果不留最后一行空回车，字典最后一个字符将无法被加载，移动端对应字符将永远识别失败", 2)
])

# ----------------- NEW SLIDE 11: 三、模型微调与训练 —— 核心脚本执行逻辑 -----------------
slide11 = prs.slides.add_slide(prs.slide_layouts[6])
add_slide_title(slide11, "三、模型微调与训练 —— 核心脚本执行逻辑")
add_slide_bullets(slide11, [
    ("1. extract_charset.py (字符集提取脚本)", 0),
    ("扫描本地样本集内全部 ground-truth 文本文件 (.gt.txt) 并提取所有出现的字符", 1),
    ("剔除重复字符并执行排序，最终在指定路径输出 custom_dict.txt (供训练/推理加载的字典)", 1),
    ("2. prepare_train_data.py (数据分配与打包脚本)", 0),
    ("读取图片和对应标注，按照 8:2 拆分为训练集与验证集，并生成对应的标注列表文本", 1),
    ("在标注清单中为每行数据写入“图片相对路径 + 制表符(\\t) + 标注真实文本”，严防空格错误", 1),
    ("3. train.py & export_model.py (微调与部署转换脚本)", 0),
    ("train.py：加载指定的预训练权重，执行向前传播计算损失，并基于 Adam 优化器在反向传播中更新权重", 1),
    ("export_model.py：剥离模型中仅在训练中使用的梯度与更新节点，冻结拓扑，大幅减小部署包体积", 1)
])

# ----------------- NEW SLIDE 12: 三、模型微调与训练 —— 模糊与金属雕刻可行性分析 -----------------
slide12 = prs.slides.add_slide(prs.slide_layouts[6])
add_slide_title(slide12, "三、模型微调与训练 —— 模糊与金属雕刻可行性分析")

# Left Column Card: Blurry images
blurry_c = slide12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(5.6), Inches(4.8))
blurry_c.fill.solid()
blurry_c.fill.fore_color.rgb = RGBColor(255, 255, 255)
blurry_c.line.color.rgb = RGBColor(203, 213, 225)
blurry_c.line.width = Pt(1.5)
add_custom_text(slide12, "针对模糊图片 (抖动/失焦) 训练分析", Inches(1.0), Inches(1.7), Inches(5.2), Inches(0.4), font_size=Pt(15), bold=True, color=RGBColor(3, 105, 161), align=PP_ALIGN.LEFT)
add_custom_text(slide12, 
    "• 可行性：高可行性。可通过数据增强和端侧过滤解决。\n\n"
    "• 训练数据增强：\n"
    "  在训练 YAML 配置中启用 MotionBlur (运动模糊) 和 GaussianBlur (高斯模糊)。强制模型学习字符的全局拓扑结构，而非脆弱的高频边缘信息。\n\n"
    "• 端侧质量过滤策略：\n"
    "  在 Android 相机预览回调中使用 OpenCV 实时计算 Laplacian 梯度方差。若低于指定阈值（代表严重失焦），直接舍弃该帧并提示“请稳固设备”，从源头规避坏数据送入推理引擎。\n\n"
    "• SVTR 结构优势：\n"
    "  视觉 Transformer 注意力机制对局部模糊天然具备高容忍度，能够有效聚合全局上下文进行正确预测。", 
    Inches(1.0), Inches(2.2), Inches(5.2), Inches(3.9), font_size=Pt(12), align=PP_ALIGN.LEFT)

# Right Column Card: Metal engraving
metal_c = slide12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.9), Inches(1.5), Inches(5.6), Inches(4.8))
metal_c.fill.solid()
metal_c.fill.fore_color.rgb = RGBColor(255, 255, 255)
metal_c.line.color.rgb = RGBColor(2, 132, 199)
metal_c.line.width = Pt(2.0)
add_custom_text(slide12, "针对金属雕刻 (钢印/激光反光) 训练分析", Inches(7.1), Inches(1.7), Inches(5.2), Inches(0.4), font_size=Pt(15), bold=True, color=RGBColor(2, 132, 199), align=PP_ALIGN.LEFT)
add_custom_text(slide12, 
    "• 可行性：中高可行性。属于典型的低对比度噪点场景。\n\n"
    "• 钢印文字区域定位 (DBNet)：\n"
    "  由于钢印与背景材质同色，极易漏检。必须使用实际场景的金属多边形标注对检测模型进行专门微调。\n\n"
    "• C++ 端自适应前处理 (CLAHE)：\n"
    "  在图像送入识别引擎前，使用 OpenCV 对灰度图进行限制对比度自适应直方图均衡化，强制拉大反光下凹凸边缘的明暗度差距。\n\n"
    "• 字典精准裁剪防 hallucination：\n"
    "  对移动端字典做严格裁剪（只保留大写字母和数字），防止金属表面斑点噪点被误识为汉字，降低混淆率。", 
    Inches(7.1), Inches(2.2), Inches(5.2), Inches(3.9), font_size=Pt(12), align=PP_ALIGN.LEFT)

# ----------------- SLIDE 13 (was 11): 三、模型微调与训练 —— YAML 配置与微调 -----------------
slide13 = prs.slides.add_slide(prs.slide_layouts[6])
add_slide_title(slide13, "三、模型微调与训练 —— YAML 配置与微调")

# Left Column code block
yaml_code = (
    "Global:\n"
    "  use_gpu: true                            # 开启 GPU 加速\n"
    "  epoch_num: 100                           # 微调 50~100 轮\n"
    "  save_model_dir: ./output/rec_vincode     # 保存目录\n"
    "  character_dict_path: vincode_full.txt    # 字典路径\n\n"
    "Train:\n"
    "  dataset:\n"
    "    data_dir: ./train_data/                # 数据目录\n"
    "    label_file_list: [rec_train_list.txt]   # 标签列表"
)
add_code_panel(slide13, yaml_code, Inches(0.8), Inches(1.5), Inches(5.6), Inches(4.8))

# Right Column explanation
add_custom_text(slide13, "配置与微调重点说明", Inches(6.9), Inches(1.5), Inches(5.6), Inches(0.4), font_size=Pt(15), bold=True, color=RGBColor(3, 105, 161), align=PP_ALIGN.LEFT)
add_custom_text(slide13,
    "• character_dict_path 务必配置为第二步导出的 full_charset.txt，防止字典映射错误导致文字识别为乱码。\n\n"
    "• 启动微调命令：\n"
    "  请使用以下命令加载官方 best_accuracy.pdparams 权重进行迁移学习微调，切勿从零训练：",
    Inches(6.9), Inches(2.0), Inches(5.6), Inches(2.0), font_size=Pt(13), align=PP_ALIGN.LEFT)
cmd_text = "python tools/train.py \\\n  -c configs/rec/rec_vincode_v4.yml \\\n  -o Global.pretrained_model=./best_accuracy"
add_code_panel(slide13, cmd_text, Inches(6.9), Inches(4.5), Inches(5.6), Inches(1.8))

# ----------------- SLIDE 14 (was 12): 四、端侧转换与集成 —— 推理模型导出与格式转换 -----------------
slide14 = prs.slides.add_slide(prs.slide_layouts[6])
add_slide_title(slide14, "四、端侧转换与集成 —— 推理模型导出与格式转换")
add_slide_bullets(slide14, [
    ("1. 导出推理模型 (Inference Model)", 0),
    ("微调生成的 .pdparams 含有反向计算、优化器等冗余节点，大小偏大，不适合部署", 1),
    ("运行 export_model.py 剪枝掉冗余，仅保留前向推理，体积缩减 70%：", 1),
    ("命令：python tools/export_model.py -c configs/... -o Global.pretrained_model=... -o Global.save_inference_dir=./output/infer_rec", 2),
    ("2. 转换为移动端格式 (以 Paddle Lite 为例)", 0),
    ("运行 paddle_lite_opt 转换为 ARM 架构直接加载 of .nb 格式模型：", 1),
    ("命令：paddle_lite_opt --model_file=inference.pdmodel --param_file=inference.pdiparams --optimize_out=recognize --valid_targets=arm", 2)
])

# ----------------- SLIDE 15 (was 13): 四、端侧转换与集成 —— Android 核心代码示例 -----------------
slide15 = prs.slides.add_slide(prs.slide_layouts[6])
add_slide_title(slide15, "四、端侧转换与集成 —— Android 核心代码示例")

# Left Column explanation
add_custom_text(slide15, "工程集成步骤", Inches(0.8), Inches(1.5), Inches(5.6), Inches(0.4), font_size=Pt(15), bold=True, color=RGBColor(3, 105, 161), align=PP_ALIGN.LEFT)
add_custom_text(slide15,
    "• 把生成的 recognize.nb 及 full_charset.txt 字典拷贝至 Android assets 目录：\n"
    "  app/src/main/assets/models/recognize/\n\n"
    "• 在 initModel 方法中，设置 PowerMode 为 LITE_POWER_HIGH 启用大核保证高频计算。\n\n"
    "• 设置 cpuThreadNum 为 4。切勿设置过高，防止频繁上下文切换反而导致设备发热严重降频。",
    Inches(0.8), Inches(2.1), Inches(5.6), Inches(3.8), font_size=Pt(13), align=PP_ALIGN.LEFT)

# Right Column Code block
kotlin_code = (
    "// 1. 初始化移动端预测器\n"
    "val config = MobileConfig().apply {\n"
    "    setModelFromFile(\"models/rec/recognize.nb\")\n"
    "    setPowerMode(PowerMode.LITE_POWER_HIGH)\n"
    "    setThreads(4) // 绑定大核提升运行速度\n"
    "}\n"
    "val predictor = PaddlePredictor.createPaddlePredictor(config)\n\n"
    "// 2. 传入 Bitmap 图像，执行推理并获取结果\n"
    "suspend fun recognizeImage(\n"
    "    bitmap: Bitmap,\n"
    "    callback: (String?) -> Unit\n"
    ") = withContext(Dispatchers.IO) {\n"
    "    if (bitmap.isRecycled || bitmap.width <= 0) {\n"
    "        callback(null)\n"
    "        return@withContext\n"
    "    }\n"
    "    // 将 Bitmap 转为 OpenCV Mat，并调用 JNI 处理\n"
    "    val mat = Mat()\n"
    "    Utils.bitmapToMat(bitmap, mat)\n"
    "    \n"
    "    // 执行图像文本检测与 OCR 识别并返回结果\n"
    "    val ocrResult = recognizeText(mat)\n"
    "    withContext(Dispatchers.Main) {\n"
    "        callback(ocrResult)\n"
    "    }\n"
    "}"
)
add_code_panel(slide15, kotlin_code, Inches(6.9), Inches(1.5), Inches(5.6), Inches(4.8), font_size=Pt(9.0))

# ----------------- SLIDE 16 (was 14): 四、端侧转换与集成 —— 实际场景识别截图对比 (Editable Placeholders!) -----------------
slide16 = prs.slides.add_slide(prs.slide_layouts[6])
add_slide_title(slide16, "四、端侧转换与集成 —— 实际场景识别截图对比")

# Left Card: Tiny
c_t = slide16.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(3.6), Inches(4.8))
c_t.fill.solid()
c_t.fill.fore_color.rgb = RGBColor(255, 255, 255)
c_t.line.color.rgb = RGBColor(203, 213, 225)
c_t.line.width = Pt(1.5)
add_custom_text(slide16, "Tiny 模型识别效果 (极速型)", Inches(0.9), Inches(1.7), Inches(3.4), Inches(0.4), font_size=Pt(14), bold=True, color=RGBColor(3, 105, 161))
add_media_placeholder(slide16, "[ 请在此处双击插入图片：tiny.png ]\n\n(左边距 1.0\", 顶边距 2.2\", 宽 3.2\", 高 2.2\")", Inches(1.0), Inches(2.2), Inches(3.2), Inches(2.2))
add_custom_text(slide16, "体积: 6.5MB\n耗时极短，CPU占用极低；对极度歪斜和反光钢印敏感度稍高，适合低功耗设备连续扫码。", Inches(0.9), Inches(4.6), Inches(3.4), Inches(1.5), font_size=Pt(12), align=PP_ALIGN.LEFT)

# Middle Card: Small
c_s = slide16.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(4.8), Inches(1.5), Inches(3.6), Inches(4.8))
c_s.fill.solid()
c_s.fill.fore_color.rgb = RGBColor(255, 255, 255)
c_s.line.color.rgb = RGBColor(203, 213, 225)
c_s.line.width = Pt(1.5)
add_custom_text(slide16, "Small 模型识别效果 (均衡型)", Inches(4.9), Inches(1.7), Inches(3.4), Inches(0.4), font_size=Pt(14), bold=True, color=RGBColor(3, 105, 161))
add_media_placeholder(slide16, "[ 请在此处双击插入图片：small.png ]\n\n(左边距 5.0\", 顶边距 2.2\", 宽 3.2\", 高 2.2\")", Inches(5.0), Inches(2.2), Inches(3.2), Inches(2.2))
add_custom_text(slide16, "体积: 31MB\n速度与精度完美折中；能应对工业扫码的常规污损与偏折，为 Android 终端主打推荐模型。", Inches(4.9), Inches(4.6), Inches(3.4), Inches(1.5), font_size=Pt(12), align=PP_ALIGN.LEFT)

# Right Card: Medium
c_m = slide16.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.8), Inches(1.5), Inches(3.6), Inches(4.8))
c_m.fill.solid()
c_m.fill.fore_color.rgb = RGBColor(255, 255, 255)
m_border_color = RGBColor(2, 132, 199)
c_m.line.color.rgb = m_border_color
c_m.line.width = Pt(2.0)
add_custom_text(slide16, "Medium 模型识别效果 (高精型)", Inches(8.9), Inches(1.7), Inches(3.4), Inches(0.4), font_size=Pt(14), bold=True, color=m_border_color)
add_media_placeholder(slide16, "[ 请在此处双击插入图片：medium.png ]\n\n(左边距 9.0\", 顶边距 2.2\", 宽 3.2\", 高 2.2\")", Inches(9.0), Inches(2.2), Inches(3.2), Inches(2.2))
add_custom_text(slide16, "体积: 138MB\n精度最高；能彻底克制工业重度偏斜拉伸与低对比反光钢印。计算密，建议结合 Vulkan 显卡加速。", Inches(8.9), Inches(4.6), Inches(3.4), Inches(1.5), font_size=Pt(12), align=PP_ALIGN.LEFT)

# ----------------- SLIDE 17 (was 15): 四、端侧转换与集成 —— 全屏模式与演示视频 (Editable Placeholders!) -----------------
slide17 = prs.slides.add_slide(prs.slide_layouts[6])
add_slide_title(slide17, "四、端侧转换与集成 —— 全屏模式与演示视频")

# Left Column Card: Fullscreen
c_fs = slide17.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(5.6), Inches(4.8))
c_fs.fill.solid()
c_fs.fill.fore_color.rgb = RGBColor(255, 255, 255)
c_fs.line.color.rgb = RGBColor(203, 213, 225)
c_fs.line.width = Pt(1.5)
add_custom_text(slide17, "全屏模式识别演示 (高解析全局检测)", Inches(1.0), Inches(1.7), Inches(5.2), Inches(0.4), font_size=Pt(15), bold=True, color=RGBColor(3, 105, 161), align=PP_ALIGN.LEFT)
add_media_placeholder(slide17, "[ 请在此处双击插入图片：fullscreen.png ]\n\n(左边距 1.0\", 顶边距 2.2\", 宽 5.2\", 高 2.8\")", Inches(1.0), Inches(2.2), Inches(5.2), Inches(2.8))
add_custom_text(slide17, "全景模式下，检测模型自动锁定屏幕中所有的文字区块，并依次裁剪后送入识别器解码，实现多行文本的毫秒级高精度提取。", Inches(1.0), Inches(5.2), Inches(5.2), Inches(0.9), font_size=Pt(12), align=PP_ALIGN.LEFT)

# Right Column Card: Video
c_vd = slide17.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.9), Inches(1.5), Inches(5.6), Inches(4.8))
c_vd.fill.solid()
c_vd.fill.fore_color.rgb = RGBColor(255, 255, 255)
c_vd.line.color.rgb = RGBColor(2, 132, 199)
c_vd.line.width = Pt(2.0)
add_custom_text(slide17, "扫码悬浮窗 Demo 演示视频", Inches(7.1), Inches(1.7), Inches(5.2), Inches(0.4), font_size=Pt(15), bold=True, color=RGBColor(2, 132, 199))
add_media_placeholder(slide17, "[ 请在此处双击插入视频：\nScreen_recording_20260706_142959.mp4 ]\n\n(左边距 7.8\", 顶边距 2.2\", 宽 3.8\", 高 2.8\")", Inches(7.8), Inches(2.2), Inches(3.8), Inches(2.8))
add_custom_text(slide17, "双击上方占位框区域，或在文件夹中直接播放 record/Screen_recording_20260706_142959.mp4 视频文件进行演示。", Inches(7.1), Inches(5.2), Inches(5.2), Inches(0.9), font_size=Pt(12))

# ----------------- SLIDE 18 (was 16): 四、端侧转换与集成 —— 性能调优锦囊 -----------------
slide18 = prs.slides.add_slide(prs.slide_layouts[6])
add_slide_title(slide18, "四、端侧转换与集成 —— 性能调优锦囊")
add_slide_bullets(slide18, [
    ("1. FP16 半精度压缩", 0),
    ("在模型转换时开启 FP16。模型大小直接减小 50% (3MB)，且精度几乎无损，大幅减小内存带宽压力", 1),
    ("2. 合理配置 CPU 推理线程数", 0),
    ("建议将线程数设置为 2 到 4，绑定大核。避免过高线程造成频繁上下文切换及 CPU 发热烫手", 1),
    ("3. Vulkan GPU 加速机制", 0),
    ("在 Android 部署中开启 Vulkan GPU 加速。将复杂的卷积计算交给显卡，CPU 占用减半，彻底防范扫码发热降频", 1)
])

# ----------------- SLIDE 19 (was 17): 五、总结 —— 迭代优化方向 -----------------
slide19 = prs.slides.add_slide(prs.slide_layouts[6])
add_slide_title(slide19, "五、总结 —— 迭代优化方向")
add_slide_bullets(slide19, [
    ("智能化升级", 0),
    ("引入更智能的甚至是 AI 技术等辅助规则配置与场景自动识别，减少人工干预", 1),
    ("优化复杂环境（反光、模糊、黑暗等）识别算法，进一步提升特殊场景适配能力", 1),
    ("生态化拓展", 0),
    ("完善与多应用的协同功能，开放更多更合理简单的广播或 AIDL 接口，支持广泛的第三方应用调用", 1),
    ("轻量化优化", 0),
    ("持续调优轻量化模型的运行内存占用，适配更多低端 Android 手持 PDA 扫码枪设备，扩大应用覆盖", 1)
])

# Save presentation
prs.save(output_path)
print(f"Clean, highly-designed editable presentation successfully saved to: {output_path}")
