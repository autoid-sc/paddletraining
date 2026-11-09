---
marp: true
theme: default
size: 16:9
paginate: true
footer: "应用开发：徐浩然 | Paddle模型训练和场景应用"
style: |
  section {
    background-image: url('ppt_resources/image2.png');
    background-size: cover;
    background-position: center;
    color: #334155;
    font-size: 22px;
    padding: 70px 80px 80px 80px; /* 调整内边距，使文字居中对齐，不遮挡模板的页眉页脚 */
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  }
  section.lead {
    background-image: url('ppt_resources/image1.png') !important;
    background-size: cover;
    background-position: center;
    padding: 40px;
  }
  h1 {
    color: #0369a1;
    font-size: 44px;
    text-align: center;
    margin-top: 10%;
    border-bottom: none;
  }
  h2 {
    color: #0284c7;
    font-size: 32px;
    margin-top: -15px;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 10px;
  }
  h3 {
    color: #0369a1;
    font-size: 24px;
    margin-top: 15px;
    margin-bottom: 5px;
  }
  footer {
    font-size: 14px;
    color: #64748b;
  }
  .lead {
    text-align: center;
    font-size: 26px;
    color: #475569;
    margin-top: 10px;
  }
  .author-info {
    text-align: center;
    font-size: 18px;
    color: #64748b;
    margin-top: 30px;
    line-height: 1.8;
  }
  .columns {
    display: flex;
    gap: 20px;
    margin-top: 10px;
  }
  .column {
    flex: 1;
    background-color: rgba(255, 255, 255, 0.9); /* 半透明底色，与背景图更好融合 */
    padding: 15px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    border-top: 4px solid #0284c7;
  }
  .badge {
    background-color: #e0f2fe;
    color: #0369a1;
    padding: 2px 8px;
    border-radius: 9999px;
    font-size: 13px;
    font-weight: bold;
    display: inline-block;
    margin-left: 5px;
    vertical-align: middle;
  }
  .img-container {
    display: flex;
    justify-content: space-around;
    gap: 15px;
    margin-top: 10px;
  }
  .img-item {
    text-align: center;
    flex: 1;
    background: rgba(255, 255, 255, 0.9);
    padding: 10px;
    border-radius: 6px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    border-top: 3px solid #cbd5e1;
  }
  .img-item h4 {
    margin: 5px 0;
    font-size: 16px;
    color: #0369a1;
  }
  .img-item img {
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    max-height: 180px;
    width: 100%;
    object-fit: contain;
  }
  .img-item p {
    font-size: 13px;
    color: #64748b;
    margin-top: 5px;
    line-height: 1.3;
  }
  pre {
    background-color: #1e293b;
    color: #f8fafc;
    padding: 12px;
    border-radius: 6px;
    font-size: 15px;
    line-height: 1.4;
    overflow: auto;
  }
  code {
    font-family: 'Consolas', 'Courier New', Courier, monospace;
    font-size: 90%;
  }
---

<!-- _class: lead -->
<!-- _backgroundImage: "url('ppt_resources/image1.png')" -->
<!-- _backgroundSize: cover -->
<!-- _backgroundPosition: center -->
# 应用开发：徐浩然
<p class="lead">Paddle模型训练和场景应用</p>

<div class="author-info">
  报告人：Android 研发团队 / 算法工程组<br>
  适用对象：Android 客户端开发工程师 / 移动端架构师 / 项目技术负责人
</div>

---

## 目录

* **第一部分：引言 & 核心原理解析**
  * 移动端部署挑战与 DBNet/SVTR_LCNet 通俗化原理、方案对比。
* **第二部分：PPOCRv6 特性 & 模型差异对比**
  * 蒸馏、骨干网络升级，以及 Tiny、Small、Medium 模型的详细区别。
* **第三部分：模型微调与训练实战**
  * 数据准备结构、Tab 黄金字典法则、YAML 参数配置与微调。
* **第四部分：端侧转换、部署与实测效果展示**
  * 模型导出及格式转换、Android 端 Kotlin 代码示例。
  * 实际 Tiny、Small、Medium 模型识别效果截图与全屏演示视频。
* **第五部分：总结 & 迭代优化方向**
  * 移动端调优锦囊以及后续“智能化、生态化、轻量化”演进。

---

## 一、引言 —— 移动端 OCR 核心挑战与诉求

### 1. 移动端部署的三大“拦路虎”
* **包体积限制**：常规深度学习模型动辄上百 MB，会使 App 肥胖，严重影响下载转化率。
* **内存与 OOM 风险**：移动设备内存受限，OCR 过程如发生内存抖动，极易被系统强杀（Out of Memory）。
* **持续扫码的发热降频**：PDA 扫码枪在仓库、港口高频扫码时，CPU 持续高负荷运转会导致设备严重发热，进而触发降频保护，导致扫码卡顿、帧率骤降。

### 2. 精度与业务的冲突
* 通用 OCR 识别库在面对特定格式（如车架 VIN 码、特殊工业标签、反光钢印）时，识别率往往不达标。我们需要一种**能够针对特定业务快速定制优化**的方案。

---

## 一、引言 —— 深度学习 OCR 原理通俗化

### 1. 文本检测（DBNet）：找位置的“智能魔棒”
* **传统方式**：用一个矩形方框（如目标检测 YOLO）去套文字。若文字倾斜、弯曲，框里会塞入大量背景噪声，导致识别错误。
* **通俗理解**：DBNet 就像是 Photoshop 里的**智能魔棒/套索工具**。它在像素级别寻找文字的“轮廓边界”。不管是侧向拍摄、弧形排布还是严重倾斜，它都能紧紧贴着文字边缘把它们“抠出来”，再通过仿射变换把它们拉直、切成一行一行的文字小纸条（文本切片）。

### 2. 文本识别（SVTR_LCNet）：读文字的“阅读专家”
* **通俗理解**：它拿到被抠出来的文字行小纸条后，像看书一样，一字句读取，并将其翻译成电子文本字符串。
* **为什么移动端一定要移除 RNN（循环神经网络）？**
  * RNN 处理数据就像“排队买票”，后一个字必须等前一个字处理完才能开始，无法并发。这在手机的多核 CPU 上非常吃亏。
  * **PP-OCRv4/v6 优化**：移除了 RNN 算子，采用超轻量级的深度可分离卷积（LCNet）和视觉 Transformer 注意力机制（SVTR）。它允许在手机 CPU 上进行多线程并行计算，极大地减少了计算延迟。

---

## 一、引言 —— 主流 OCR 方案双栏对比

<div class="columns">
  <div class="column">
    <h3>开箱即用方案</h3>
    <ul>
      <li><b>Google ML Kit (On-device)</b>
        <ul>
          <li><b>优</b>：免去训练，若用 Play 服务甚至 0MB，API 简单。</li>
          <li><b>劣</b>：不支持自定义训练，遇到偏折、反光钢印、特定工业 VIN 码等场景时，识别率低。</li>
        </ul>
      </li>
      <li><b>Tesseract (Tessdata)</b>
        <ul>
          <li><b>优</b>：老牌、开源、极易跨平台集成。</li>
          <li><b>劣</b>：基于传统图像分割+老式 LSTM，不支持现代 GPU 加速，面对偏折、暗光、噪点时识别率极差。</li>
        </ul>
      </li>
    </ul>
  </div>
  
  <div class="column">
    <h3>深度学习定制方案</h3>
    <ul>
      <li><b>EasyOCR (PyTorch)</b>
        <ul>
          <li><b>优</b>：Python 生态好，识别精度高。</li>
          <li><b>劣</b>：模型体积大（100MB+），在 Android 上运行极其吃内存，且没有移动端推理优化，极易 OOM。</li>
        </ul>
      </li>
      <li><b>PaddleOCR (PP-OCR)</b> <span class="badge">推荐</span>
        <ul>
          <li><b>优</b>：拥有轻量化大模型底座（仅约十几 MB）。支持使用 50~100 张图片一键微调，能将特定业务场景精度提升至 98% 以上。且完美适配 ARM 汇编与 Vulkan GPU 加速。</li>
          <li><b>劣</b>：需要搭建 Python 环境并进行模型导出与转换。</li>
        </ul>
      </li>
    </ul>
  </div>
</div>

---

## 二、PPOCRv6 特性 —— 极致轻量化核心选型

### 1. PP-OCRv6 核心技术特征
* **蒸馏算法升级**：通过大模型引导小模型（Distillation）进行训练，在大幅削减参数量的同时，使 Tiny 和 Small 模型保持与大模型相当的识别精度。
* **骨干网络优化**：采用了更适合移动端 ARM 架构的 PP-LCNetV2 骨干网络，对卷积算子进行了汇编级别的优化，计算延迟极低。
* **方向分类器（Cls）增强**：支持自动识别并纠正图片中文字的 0°/90°/180°/270° 旋转角度，配合 DBNet 提取的弯曲文本进行仿射透视校正，完美解决侧视、斜视问题。
* **零字符对齐 CTC 解码**：有效解决紧凑型字符（如 VIN 码、连体字母）的字符粘连问题，大幅降低误识率。

---

## 二、PPOCRv6 特性 —— Tiny, Small, Medium 区别与对比

* **PP-OCRv6 Tiny (极速版)**
  * **文件体积**：检测模型 ~1.9MB，识别模型 ~4.6MB，整体约 **6.5MB**。
  * **优缺点及场景**：内存与 CPU 负载极低。适用于低端扫码枪或需要超高频视频流扫码的场景。在良好光照、标准字体下速度最快，但对重度倾斜和复杂背景略微敏感。
* **PP-OCRv6 Small (均衡版)**
  * **文件体积**：检测模型 ~10MB，识别模型 ~21MB，整体约 **31MB**。
  * **优缺点及场景**：平衡了精度和速度。适用于包含轻微畸变、污损和反光的工业扫码场景，在中端处理器上运行流畅。
* **PP-OCRv6 Medium (高精版)**
  * **文件体积**：检测模型 ~62MB，识别模型 ~76.8MB，整体约 **138MB**。
  * **优缺点及场景**：参数量最大，识别率极高，但 CPU 负载高，推荐使用 Vulkan GPU 进行硬件加速。能够攻克低对比度反光钢印、重度倾斜斜视、细小密集字符等极端环境。

---

## 三、模型微调与训练 —— 数据准备与标注清单

### 1. 规范化训练数据结构
在 `PaddleOCR` 目录下创建 `train_data` 目录，将微调数据（文本行裁剪图）组织成以下结构：
```text
PaddleOCR/train_data/
    ├── rec_train/            # 训练集切片小图（包含 50~100 张微调图片）
    ├── rec_val/              # 验证集切片小图
    ├── rec_train_list.txt    # 训练集标注清单
    └── rec_val_list.txt      # 验证集标注清单
```

### 2. 制作标注清单
`rec_train_list.txt` 每行代表一个样本。图片路径与标注文本之间**必须使用单个 Tab 键**（`\t`）分隔：
```text
rec_train/word_001.jpg	2022050508:50:54
rec_train/word_002.jpg	14B1C946B162
```
* ⚠️ **警告**：绝对不可使用空格，否则训练程序读取时会报错崩溃。

---

## 三、模型微调与训练 —— 自动提取字典与黄金法则

### 1. 自动提取专属字符字典
运行 Python 脚本扫描所有的标注文本，自动提取出所有出现的非重复字符并剔除重复：
```bash
python tools/extract_charset.py  # 自动提取出专属的 full_charset.txt 字典
```

### 2. 字典配置的“黄金法则”
* 每行只能写一个字符。
* **字典文件末尾必须多保留一个回车空行**。如果没有最后这行空白回车，字典里最后那个字符将无法被程序加载，从而导致该字符在移动端永远识别错误！
```text
0
1
A
[这里必须有一个回车空行]
```

---

## 三、模型微调与训练 —— YAML 参数配置与微调

### 1. 修改 YAML 配置文件 (`configs/rec/rec_vincode_v4.yml`)
```yaml
Global:
  use_gpu: true                            # 开启 GPU 显卡训练加速
  epoch_num: 100                           # 微调仅需 50~100 轮即可快速收敛
  save_model_dir: ./output/rec_vincode     # 权重保存目录
  character_dict_path: ppocr/utils/dict/vincode_full.txt # 刚才生成的字典路径
Train:
  dataset:
    name: SimpleDataSet
    data_dir: ./train_data/                # 标注文件里图片路径的前缀目录
    label_file_list: ["./train_data/rec_train_list.txt"]
```

### 2. 启动微调命令
加载官方预训练权重（如 `best_accuracy.pdparams`，已具备辨识边缘与线条能力），开始微调：
```bash
python tools/train.py \
  -c configs/rec/rec_vincode_v4.yml \
  -o Global.pretrained_model=./pretrain/ch_PP-OCRv4_rec_train/best_accuracy
```

---

## 四、端侧转换与集成 —— 推理模型导出与格式转换

### 1. 导出推理模型（Inference Model）
训练权重（`.pdparams`）包含反向传播的梯度、优化器参数等冗余信息。运行导出脚本，把这些辅助训练的节点“修剪”掉，体积可缩减 70%：
```bash
python tools/export_model.py \
  -c configs/rec/rec_vincode_v4.yml \
  -o Global.pretrained_model=./output/rec_vincode/best_accuracy \
  -o Global.save_inference_dir=./output/infer_rec
```

### 2. 转换为移动端格式 (以 Paddle Lite 优化工具为例)
```bash
# 运行优化工具，直接转换成单文件、高压缩的移动端专属模型 (.nb 格式)
paddle_lite_opt \
  --model_file=./output/infer_rec/inference.pdmodel \
  --param_file=./output/infer_rec/inference.pdiparams \
  --optimize_out=./output/nb_models/recognize \
  --valid_targets=arm
```

---

## 四、端侧转换与集成 —— Android 核心代码示例

* 将导出的移动端模型与字典拷贝到 Android 工程的资源目录下：`app/src/main/assets/models/`

```kotlin
class PaddleOCRService(val context: Context) {
    private var predictor: PaddlePredictor? = null

    fun initModel(modelTier: OcrConfig.OcrModelTier) {
        val config = MobileConfig().apply {
            // 根据 OcrConfig.ocrModelTier 动态选择加载对应的模型文件
            val modelFile = when(modelTier) {
                OcrConfig.OcrModelTier.TINY -> "rec_tiny.nb"
                OcrConfig.OcrModelTier.SMALL -> "rec_small.nb"
                OcrConfig.OcrModelTier.MEDIUM -> "rec_medium.nb"
            }
            setModelFromFile("${context.filesDir}/models/recognize/$modelFile")
            setPowerMode(PowerMode.LITE_POWER_HIGH) // 开启大核高性能
            setThreads(4)                           // 限制 CPU 线程数，防止过多导致发热
        }
        predictor = PaddlePredictor.createPaddlePredictor(config)
    }
}
```

---

## 四、端侧转换与集成 —— 实际场景识别截图对比

<div class="img-container">
  <div class="img-item">
    <h4>Tiny 模型识别效果 (极速型)</h4>
    <img src="record/tiny.png" alt="Tiny Model">
    <p>耗时短，体积小 (6.5MB)；对极度倾斜和复杂反光敏感度稍高</p>
  </div>
  <div class="img-item">
    <h4>Small 模型识别效果 (均衡型)</h4>
    <img src="record/small.png" alt="Small Model">
    <p>速度与精度达到极佳平衡 (31MB)；适合中高端扫码终端</p>
  </div>
  <div class="img-item">
    <h4>Medium 模型识别效果 (高精型)</h4>
    <img src="record/medium.png" alt="Medium Model">
    <p>识别率最高 (138MB)；完美攻克侧向重度畸变与暗光噪点</p>
  </div>
</div>

---

## 四、端侧转换与集成 —— 全屏模式与演示视频

<div class="columns">
  <div class="column" style="flex: 1.2;">
    <h3>全屏模式识别演示</h3>
    <img src="record/fullscreen.png" alt="Fullscreen Mode" style="max-height: 230px; width: 100%; object-fit: contain; border: 1px solid #cbd5e1; border-radius: 6px;">
    <p style="font-size: 14px; margin-top: 5px;">全屏景模式下，检测模型自动锁定屏幕中所有的文字区块，并依次裁剪后送入识别器解码，实现多行文本的毫秒级高精度提取。</p>
  </div>
  <div class="column" style="flex: 0.8; text-align: center;">
    <h3>Demo 演示视频录屏</h3>
    <video src="record/Screen_recording_20260706_142959.mp4" controls width="220" style="border: 1px solid #cbd5e1; border-radius: 6px; max-height: 230px;"></video>
    <p style="font-size: 14px; margin-top: 5px;"><a href="record/Screen_recording_20260706_142959.mp4">点击打开原始录屏文件</a></p>
  </div>
</div>

---

## 四、端侧转换与集成 —— 性能调优锦囊

### 1. FP16 半精度压缩
* 默认模型是 32 位浮点数（FP32）。在转换模型时开启 FP16 模式，可把模型大小直接**减小 50%**。
* **收益**：减少了移动端读取模型时的内存带宽压力，运行速度还能提升 10%~20%，且识别准确率几乎无损。

### 2. 合理配置 CPU 推理线程数
* 并非线程数越多越快！Android 手机多是“大小核”架构，建议将线程数设置为 **2 到 4**（绑定到大核）。
* **风险**：线程数设置过高会导致线程间频繁上下文切换，并引发 CPU 急剧发热，造成手机降频卡顿。

### 3. Vulkan GPU 加速机制
* 在 Android 部署中，开启 Vulkan 功能，将计算密集型的矩阵卷积交给手机 GPU。
* **最大收益**：CPU 占用率减半，能有效预防 PDA 扫码枪在持续工作时的**发热、烫手、降频**问题。

---

## 五、总结 —— 迭代优化方向

基于当前产品基础，后续将持续聚焦 “智能化、生态化、轻量化” 三大方向迭代优化：

* **智能化升级**
  * 引入更智能的甚至是 AI 技术等辅助规则配置与场景自动识别，减少人工干预。
  * 优化复杂环境（反光、模糊、黑暗等）识别算法，进一步提升特殊场景适配能力。
* **生态化拓展**
  * 完善与多应用的协同功能，开放更多更合理简单的广播或 AIDL 接口，支持更广泛的第三方应用调用。
* **轻量化优化**
  * 持续调优轻量化模型的运行内存占用，适配更多低端 Android 手持 PDA 设备，扩大应用覆盖范围。
