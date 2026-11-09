package com.seuic.floating.network

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect as AndroidRect
import android.util.Log
import com.baidu.paddle.lite.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.core.MatOfPoint
import org.opencv.imgproc.Imgproc
import java.io.File
import java.io.FileOutputStream
import com.seuic.floating.config.OcrConfig
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Color
import android.graphics.RectF

import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/**
 * PaddleOCR 服务 - 使用 PaddleLite 实现
 * 基于本地 Paddle 模型进行 VIN 码识别
 */
class PaddleOCRService(private val context: Context) : IOCRService {

    private val ocrMutex = Mutex()

    companion object {
        private const val TAG = "PaddleOCRService"

        // PaddleLite nb 字典文件名
        private const val DICT_FILE = "ch_dict.txt"

        // 模型目录
        private const val MODEL_DIR = "models"
        private const val DETECT_DIR = "detect"
        private const val RECOGNIZE_DIR = "recognize"

        @Volatile
        private var instance: PaddleOCRService? = null

        fun getInstance(context: Context): PaddleOCRService {
            return instance ?: synchronized(this) {
                instance ?: PaddleOCRService(context.applicationContext).also { instance = it }
            }
        }
    }

    // PaddleLite 预测器
    private var detPredictor: PaddlePredictor? = null
    private var recPredictor: PaddlePredictor? = null
    private var isInitialized = false
    private var loadedModelTier: OcrConfig.OcrModelTier? = null

    private fun getDetectModelFile(): String {
        return when (OcrConfig.ocrModelTier.value) {
            OcrConfig.OcrModelTier.PLATE -> "plate_det_small.nb"
            else -> "detect_small.nb"
        }
    }

    private fun getRecognizeModelFile(): String {
        return when (OcrConfig.ocrModelTier.value) {
            OcrConfig.OcrModelTier.CHINESE -> "rec_small.nb"
            OcrConfig.OcrModelTier.ENGLISH -> "rec_en_small.nb"
            OcrConfig.OcrModelTier.PLATE -> "plate_rec_small.nb"
            else -> "rec_small.nb"
        }
    }

    private fun getDictFile(): String {
        return when (OcrConfig.ocrModelTier.value) {
            OcrConfig.OcrModelTier.CHINESE -> "ch_dict_full.txt"
            OcrConfig.OcrModelTier.ENGLISH -> "en_dict.txt"
            OcrConfig.OcrModelTier.PLATE -> "plate_dict.txt"
            else -> "ch_dict_full.txt"
        }
    }

    // VIN 码字符白名单（已禁用，使用全字符识别）
    // private val vinCharWhitelist = setOf('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', 'A')

    // 字符集
    private var charSet: List<String> = emptyList()

    /**
     * 初始化 PaddleOCR 引擎
     */
    override suspend fun initialize(): Boolean = withContext(Dispatchers.IO) {
        ocrMutex.withLock {
            val currentTier = OcrConfig.ocrModelTier.value ?: OcrConfig.OcrModelTier.CHINESE
            if (isInitialized && loadedModelTier == currentTier) {
                Log.d(TAG, "PaddleOCR already initialized with tier: $currentTier")
                return@withLock true
            }

            try {
                Log.d(TAG, "初始化 PaddleOCR, 选定模型级别: $currentTier...")

                // If already initialized but with a different tier, release first
                if (isInitialized) {
                    release()
                }

                if (!OpenCVLoader.initDebug()) {
                    Log.e(TAG, "OpenCV 初始化失败")
                    return@withLock false
                }
                Log.d(TAG, "OpenCV 初始化成功")

                // 准备模型文件
                val detectModelFile = getDetectModelFile()
                val recognizeModelFile = getRecognizeModelFile()
                val dictFile = getDictFile()
                val modelPrepared = prepareModelFiles(detectModelFile, recognizeModelFile, dictFile)
                if (!modelPrepared) {
                    Log.w(TAG, "模型文件准备失败")
                    return@withLock false
                }

                // 加载字符集
                if (!loadCharSet()) {
                    Log.w(TAG, "字符集加载失败")
                    return@withLock false
                }

                // 初始化检测模型
                detPredictor = initPredictor(DETECT_DIR, detectModelFile)

                // 初始化识别模型
                recPredictor = initPredictor(RECOGNIZE_DIR, recognizeModelFile)

                if (detPredictor != null && recPredictor != null) {
                    isInitialized = true
                    loadedModelTier = currentTier
                    Log.d(TAG, "PaddleOCR 初始化成功, 级别: $currentTier")
                    Log.d(TAG, "字符集大小: ${charSet.size}")
                    return@withLock true
                } else {
                    Log.e(TAG, "PaddleOCR 预测器初始化失败")
                    return@withLock false
                }

            } catch (e: Exception) {
                Log.e(TAG, "PaddleOCR 初始化失败", e)
                e.printStackTrace()
                return@withLock false
            }
        }
    }

    /**
     * 准备模型文件
     */
    private fun prepareModelFiles(detectModelFile: String, recognizeModelFile: String, dictFile: String): Boolean {
        try {
            // 复制检测模型
            val detModelFile = File(context.filesDir, "$MODEL_DIR/$DETECT_DIR/$detectModelFile")
            val detDir = File(context.filesDir, "$MODEL_DIR/$DETECT_DIR")

            if (!detDir.exists()) {
                detDir.mkdirs()
            }

            if (!copyAsset("$MODEL_DIR/$DETECT_DIR/$detectModelFile", detModelFile) && detectModelFile != "detect_small.nb") {
                Log.w(TAG, "未找到专属检测模型 $detectModelFile，回退使用通用检测模型 detect_small.nb")
                copyAsset("$MODEL_DIR/$DETECT_DIR/detect_small.nb", detModelFile)
            }

            // 复制识别模型
            val recModelFile = File(context.filesDir, "$MODEL_DIR/$RECOGNIZE_DIR/$recognizeModelFile")
            val recDir = File(context.filesDir, "$MODEL_DIR/$RECOGNIZE_DIR")

            if (!recDir.exists()) {
                recDir.mkdirs()
            }

            copyAsset("$MODEL_DIR/$RECOGNIZE_DIR/$recognizeModelFile", recModelFile)

            // 复制字典文件
            val destDictFile = File(context.filesDir, "$MODEL_DIR/$RECOGNIZE_DIR/$dictFile")
            copyAsset("$MODEL_DIR/$RECOGNIZE_DIR/$dictFile", destDictFile)

            Log.d(TAG, "模型文件准备成功 (已强制覆盖更新)")
            Log.d(TAG, "检测模型: ${detModelFile.absolutePath}")
            Log.d(TAG, "识别模型: ${recModelFile.absolutePath}")
            Log.d(TAG, "字典文件: ${destDictFile.absolutePath}")

            return true
        } catch (e: Exception) {
            Log.e(TAG, "模型文件准备失败", e)
            return false
        }
    }

    /**
     * 从 assets 复制文件
     */
    private fun copyAsset(assetPath: String, destFile: File): Boolean {
        return try {
            context.assets.open(assetPath).use { input ->
                FileOutputStream(destFile).use { output ->
                    input.copyTo(output)
                }
            }
            true
        } catch (e: Exception) {
            Log.e(TAG, "复制 Asset 失败: $assetPath", e)
            false
        }
    }

    /**
     * 加载字符集
     */
    private fun loadCharSet(): Boolean {
        return try {
            val dictFile = getDictFile()
            val dictText = context.assets.open("$MODEL_DIR/$RECOGNIZE_DIR/$dictFile")
                .bufferedReader()
                .use { it.readText() }

            val list = mutableListOf<String>()
            val lines = dictText.split(Regex("\\r?\\n"))
            val rawLines = if (dictText.endsWith("\n") || dictText.endsWith("\r")) {
                lines.dropLast(1)
            } else {
                lines
            }
            rawLines.forEach { line ->
                list.add(line.replace("\r", ""))
            }
            // 追加空格字符以匹配 use_space_char=True 的训练行为
            // PLATE 模型 use_space_char=false，不追加空格
            // 注意：字典末尾空行已通过 forEach 产生一个空字符串在 list 中
            // PaddleOCR 导出时该空字符占一个 CTC 类别，因此 list.size + 1(blank) = classCount
            // 不需要额外追加，否则 charSet.size == classCount 会触发误报警告
            val isPlate = OcrConfig.ocrModelTier.value == OcrConfig.OcrModelTier.PLATE
            if (!isPlate) {
                list.add(" ")
            }
            charSet = list
            Log.d(TAG, "加载字符集成功: 大小=${charSet.size}, 字符集=${charSet.joinToString("")}")
            true
        } catch (e: Exception) {
            Log.e(TAG, "加载字符集失败", e)
            // 使用默认字符集
            charSet = listOf("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", "A")
            true
        }
    }

    /**
     * 初始化预测器
     */
    private fun initPredictor(modelDir: String, modelFile: String): PaddlePredictor? {
        return try {
            val config = MobileConfig()
            val modelPath = "${context.filesDir}/$MODEL_DIR/$modelDir/$modelFile"
            val nbFile = File(modelPath)

            Log.d(TAG, "nb模型路径: $modelPath")
            if (!nbFile.exists()) {
                Log.e(TAG, "nb模型文件不存在: $modelPath")
                return null
            }

            config.setModelFromFile(modelPath)
            config.setPowerMode(PowerMode.LITE_POWER_HIGH)
            config.setThreads(4)

            val predictor = PaddlePredictor.createPaddlePredictor(config)
            Log.d(TAG, "预测器初始化成功: $modelDir")
            predictor
        } catch (e: Exception) {
            Log.e(TAG, "预测器初始化失败: $modelDir", e)
            null
        }
    }

    /**
     * 执行 OCR 识别
     */
    override suspend fun recognizeImage(
        bitmap: Bitmap,
        callback: (String?) -> Unit
    ) = withContext(Dispatchers.IO) {
        ocrMutex.withLock {
            if (!isInitialized) {
                Log.w(TAG, "PaddleOCR 未初始化")
                callback.invoke(null)
                return@withLock
            }

            val startTime = System.currentTimeMillis()

            try {
                // 验证 bitmap
                if (bitmap.isRecycled || bitmap.width <= 0 || bitmap.height <= 0) {
                    Log.e(TAG, "Bitmap 无效")
                    callback.invoke(null)
                    return@withLock
                }

                Log.d(TAG, "开始识别: ${bitmap.width}x${bitmap.height}")

                // 转换为 Mat
                val mat = Mat()
                Utils.bitmapToMat(bitmap, mat)

                // 转换为 RGB
                val rgbMat = Mat()
                Imgproc.cvtColor(mat, rgbMat, Imgproc.COLOR_RGBA2RGB)

                // 执行检测
                val boxes = detectText(rgbMat)

                if (boxes.isEmpty()) {
                    Log.d(TAG, "未检测到文本")
                    val duration = System.currentTimeMillis() - startTime
                    OcrConfig.lastRecognitionTime.postValue(duration)
                    OcrConfig.lastOcrResult.postValue("")
                    OcrConfig.lastRecognizedBitmap.postValue(bitmap)
                    callback.invoke(null)
                    return@withLock
                }

                Log.d(TAG, "检测到 ${boxes.size} 个文本框")

                // 执行识别并按自上而下顺序排序
                val sortedBoxes = boxes.sortedBy { it.top }
                val recognizedLinePairs = mutableListOf<Pair<TextBox, String>>()
                for (box in sortedBoxes) {
                    val text = recognizeText(rgbMat, box) ?: ""
                    // 去掉·分隔符（训练标注含·，实际车牌号不含）
                    val trimmed = text.trim().replace("·", "")
                    if (trimmed.isNotEmpty()) {
                        recognizedLinePairs.add(Pair(box, trimmed))
                    }
                }

                val duration = System.currentTimeMillis() - startTime

                // 正则表达式过滤校验每个识别行
                val regexPattern = OcrConfig.regexPattern.value
                val hasRegex = !regexPattern.isNullOrEmpty()
                
                val matchedLinePairs = if (hasRegex && regexPattern != null) {
                    val localPattern = regexPattern
                    recognizedLinePairs.filter { pair ->
                        try {
                            Regex(localPattern).matches(pair.second)
                        } catch (e: Exception) {
                            Log.e(TAG, "正则表达式匹配异常: ${e.message}")
                            false
                        }
                    }
                } else {
                    recognizedLinePairs
                }

                val finalDisplayResult: String
                val finalCallbackResult: String?

                if (matchedLinePairs.isNotEmpty()) {
                    val combinedResult = matchedLinePairs.map { it.second }.joinToString("\n")
                    finalDisplayResult = combinedResult
                    finalCallbackResult = combinedResult
                } else {
                    if (hasRegex) {
                        finalDisplayResult = "未通过正则校验"
                        finalCallbackResult = null
                    } else {
                        finalDisplayResult = ""
                        finalCallbackResult = null
                    }
                }

                // 绘制标注图 (绘制所有识别出来的文本框)
                try {
                    val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
                    val canvas = Canvas(mutableBitmap)
                    
                    val borderPaint = Paint().apply {
                        style = Paint.Style.STROKE
                        strokeWidth = 4f
                    }
                    
                    val bgPaint = Paint().apply {
                        style = Paint.Style.FILL
                    }

                    val textPaint = Paint().apply {
                        color = Color.WHITE
                        textSize = 24f
                        isAntiAlias = true
                        style = Paint.Style.FILL
                    }

                    val colors = intArrayOf(
                        Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW,
                        Color.CYAN, Color.MAGENTA, Color.rgb(255, 165, 0)
                    )

                    for (i in recognizedLinePairs.indices) {
                        val pair = recognizedLinePairs[i]
                        val box = pair.first
                        val text = pair.second
                        
                        val color = colors[i % colors.size]
                        borderPaint.color = color
                        bgPaint.color = color
                        
                        val rect = RectF(box.left, box.top, box.right, box.bottom)
                        canvas.drawRect(rect, borderPaint)
                        
                        val textWidth = textPaint.measureText(text)
                        val textHeight = textPaint.fontMetrics.descent - textPaint.fontMetrics.ascent
                        
                        val textBgRect = RectF(
                            box.left, 
                            box.top - textHeight - 6f, 
                            box.left + textWidth + 12f, 
                            box.top
                        )
                        canvas.drawRect(textBgRect, bgPaint)
                        canvas.drawText(text, box.left + 6f, box.top - textPaint.fontMetrics.descent - 3f, textPaint)
                    }
                    
                    OcrConfig.lastRecognizedBitmap.postValue(mutableBitmap)
                } catch (e: Exception) {
                    Log.e(TAG, "绘制标注框失败", e)
                    OcrConfig.lastRecognizedBitmap.postValue(bitmap)
                }

                OcrConfig.lastRecognitionTime.postValue(duration)
                OcrConfig.lastOcrResult.postValue(finalDisplayResult)

                callback.invoke(finalCallbackResult)

                // 清理
                mat.release()
                rgbMat.release()

            } catch (e: Exception) {
                Log.e(TAG, "识别失败", e)
                e.printStackTrace()
                callback.invoke(null)
            }
        }
    }

    /**
     * 文本检测 - DBNet 后处理
     */
    private fun detectText(mat: Mat): List<TextBox> {
        val boxes = mutableListOf<TextBox>()

        try {
            val predictor = detPredictor ?: return boxes

            val isPlate = OcrConfig.ocrModelTier.value == OcrConfig.OcrModelTier.PLATE

            // Compute resize parameters and produce normalized Mat
            val ratioH: Float
            val ratioW: Float
            val inputH: Int
            val inputW: Int
            val normalized: Mat

            if (isPlate) {
                // Dynamic resize: min side = 640, align to 32 multiples (matches DetResizeForTest)
                val limitSideLen = 640
                val srcH = mat.height()
                val srcW = mat.width()
                val minSide = kotlin.math.min(srcH, srcW)
                val ratio = if (minSide < limitSideLen) {
                    limitSideLen.toFloat() / minSide
                } else {
                    1.0f
                }
                val rh = maxOf(kotlin.math.round((srcH * ratio) / 32.0).toInt() * 32, 32)
                val rw = maxOf(kotlin.math.round((srcW * ratio) / 32.0).toInt() * 32, 32)
                val resizedMat = Mat()
                Imgproc.resize(mat, resizedMat, Size(rw.toDouble(), rh.toDouble()))
                normalized = Mat()
                resizedMat.convertTo(normalized, CvType.CV_32FC3, 1.0 / 255.0)
                resizedMat.release()
                inputH = rh
                inputW = rw
                ratioH = rh.toFloat() / srcH
                ratioW = rw.toFloat() / srcW
            } else {
                // Original: resize + pad to 480x480 (preserves existing CHINESE/ENGLISH behavior)
                val inputSize = 480
                val scale = minOf(inputSize.toFloat() / mat.width(), inputSize.toFloat() / mat.height())
                val newWidth = (mat.width() * scale).toInt()
                val newHeight = (mat.height() * scale).toInt()
                val resizedMat = Mat()
                Imgproc.resize(mat, resizedMat, Size(newWidth.toDouble(), newHeight.toDouble()))
                val normTemp = Mat()
                resizedMat.convertTo(normTemp, CvType.CV_32FC3, 1.0 / 255.0)
                resizedMat.release()
                val padded = Mat.zeros(inputSize, inputSize, CvType.CV_32FC3)
                val roi = padded.submat(0, newHeight, 0, newWidth)
                normTemp.copyTo(roi)
                normTemp.release()
                normalized = padded
                inputH = inputSize
                inputW = inputSize
                ratioH = scale
                ratioW = scale
            }

            // ImageNet normalization for PLATE, identity for others
            val meanVals = if (isPlate) doubleArrayOf(0.485, 0.456, 0.406) else doubleArrayOf(0.0, 0.0, 0.0)
            val stdVals = if (isPlate) doubleArrayOf(0.229, 0.224, 0.225) else doubleArrayOf(1.0, 1.0, 1.0)

            // NCHW with BGR channel order (2-c converts RGB->BGR, matching training img_mode: BGR)
            val inputTensor = predictor.getInput(0)
            inputTensor.resize(longArrayOf(1, 3, inputH.toLong(), inputW.toLong()))
            val matData = FloatArray(3 * inputH * inputW)

            val tempBuffer = FloatArray(inputH * inputW * 3)
            normalized.get(0, 0, tempBuffer)
            for (c in 0 until 3) {
                val offset = c * inputH * inputW
                val colorChannelIdx = 2 - c
                for (h in 0 until inputH) {
                    val rowOffset = h * inputW
                    for (w in 0 until inputW) {
                        val idx = offset + rowOffset + w
                        val srcIdx = (rowOffset + w) * 3
                        val pixel = tempBuffer[srcIdx + colorChannelIdx]
                        matData[idx] = ((pixel - meanVals[c]) / stdVals[c]).toFloat()
                    }
                }
            }

            inputTensor.setData(matData)

            // 执行推理
            predictor.run()

            // 解析检测结果 - DBNet 后处理
            val detectedBoxes = parseDBNetOutput(predictor, mat.width(), mat.height(), ratioW, ratioH)
            boxes.addAll(detectedBoxes)

            // 清理
            normalized.release()

        } catch (e: Exception) {
            Log.e(TAG, "文本检测失败", e)
        }

        return boxes
    }

    /**
     * 解析 DBNet 检测模型输出
     */
    private fun parseDBNetOutput(
        predictor: PaddlePredictor,
        originalWidth: Int,
        originalHeight: Int,
        ratioW: Float,
        ratioH: Float
    ): List<TextBox> {
        val boxes = mutableListOf<TextBox>()

        try {
            // 获取输出张量
            val outputTensor = predictor.getOutput(0)
            val outputData = outputTensor.getFloatData()
            val outputShape = outputTensor.shape()

            Log.d(TAG, "检测输出shape: ${outputShape.joinToString("x")}")

            if (outputShape.size < 3) {
                Log.e(TAG, "检测输出形状不正确")
                return boxes
            }

            // DBNet 输出形状通常是 (1, 1, H, W) 或 (1, H, W)
            val batchSize = outputShape[0].toInt()
            val channels = if (outputShape.size > 1) outputShape[1].toInt() else 1
            val outputHeight = outputShape[outputShape.size - 2].toInt()
            val outputWidth = outputShape.last().toInt()

            Log.d(TAG, "检测输出尺寸: $batchSize x $channels x $outputHeight x $outputWidth")

            // 转换为概率图 - 优化批量写入像素，避免频繁JNI调用
            val probMat = Mat.zeros(outputHeight, outputWidth, CvType.CV_32F)
            val probData = FloatArray(outputHeight * outputWidth)
            for (h in 0 until outputHeight) {
                val rowOffset = h * outputWidth
                for (w in 0 until outputWidth) {
                    val idx = rowOffset + w
                    probData[idx] = outputData[idx]
                }
            }
            probMat.put(0, 0, probData)

            // 阈值分割
            val threshold = 0.3
            val binaryMat = Mat()
            Imgproc.threshold(probMat, binaryMat, threshold, 255.0, Imgproc.THRESH_BINARY)

            // 转换为 8UC1 用于 findContours
            val binaryMat8U = Mat()
            binaryMat.convertTo(binaryMat8U, CvType.CV_8U)

            // 找到轮廓
            val contours = java.util.ArrayList<MatOfPoint>()
            val hierarchy = Mat()
            Imgproc.findContours(
                binaryMat8U, contours, hierarchy,
                Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE
            )

            Log.d(TAG, "找到 ${contours.size} 个轮廓")

            // 对每个轮廓计算边界框
            val minBoxSize = 3.0  // 最小文本框尺寸
            val minArea = minBoxSize * minBoxSize

            for (i in 0 until contours.size) {
                val contour = contours[i]
                val rect = Imgproc.boundingRect(contour)
                val width = rect.width.toDouble()
                val height = rect.height.toDouble()

                // 过滤小框
                if (width < minBoxSize || height < minBoxSize) {
                    continue
                }

                val area = width * height
                if (area < minArea) {
                    continue
                }

                // 缩放回原始图像尺寸 (使用独立的 W/H 缩放比)
                val scaledLeft = rect.x.toDouble() / ratioW
                val scaledTop = rect.y.toDouble() / ratioH
                val scaledRight = (rect.x + rect.width).toDouble() / ratioW
                val scaledBottom = (rect.y + rect.height).toDouble() / ratioH

                // 裁剪到原始图像边界
                val clippedLeft = scaledLeft.coerceIn(0.0, originalWidth.toDouble())
                val clippedTop = scaledTop.coerceIn(0.0, originalHeight.toDouble())
                val clippedRight = scaledRight.coerceIn(0.0, originalWidth.toDouble())
                val clippedBottom = scaledBottom.coerceIn(0.0, originalHeight.toDouble())

                // 计算旋转矩形的四个顶点（用于透视矫正）
                var corners: Array<Point>? = null
                try {
                    val rotatedRect = Imgproc.minAreaRect(MatOfPoint2f(*contour.toArray()))
                    val pts = Array(4) { Point() }
                    // boxPoints 返回顺序: 0=左下, 1=左上, 2=右上, 3=右下
                    rotatedRect.points(pts)
                    // 缩放到原图尺寸
                    for (p in pts) {
                        p.x = p.x / ratioW
                        p.y = p.y / ratioH
                    }
                    // 重新排序为: 左上→右上→右下→左下（顺时针）
                    // 先按 y 排序，前两个是上面，后两个是下面
                    val sortedByY = pts.sortedBy { it.y }
                    val topTwo = sortedByY.take(2).sortedBy { it.x }
                    val bottomTwo = sortedByY.drop(2).sortedByDescending { it.x }
                    // 左上, 右上, 右下, 左下
                    corners = arrayOf(topTwo[0], topTwo[1], bottomTwo[0], bottomTwo[1])
                    // 裁剪到边界内
                    for (p in corners) {
                        p.x = p.x.coerceIn(0.0, originalWidth.toDouble())
                        p.y = p.y.coerceIn(0.0, originalHeight.toDouble())
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "minAreaRect 失败，退化为轴对齐矩形", e)
                    corners = null
                }

                boxes.add(
                    TextBox(
                        clippedLeft.toFloat(),
                        clippedTop.toFloat(),
                        clippedRight.toFloat(),
                        clippedBottom.toFloat(),
                        corners
                    )
                )
            }

            // 清理
            probMat.release()
            binaryMat.release()
            binaryMat8U.release()
            hierarchy.release()
            for (contour in contours) {
                contour.release()
            }

            Log.d(TAG, "过滤后得到 ${boxes.size} 个文本框")

        } catch (e: Exception) {
            Log.e(TAG, "解析 DBNet 输出失败", e)
        }

        return boxes
    }

    /**
     * 文本识别
     */
    private fun recognizeText(mat: Mat, box: TextBox): String? {
        try {
            val predictor = recPredictor ?: return null
            val isPlate = OcrConfig.ocrModelTier.value == OcrConfig.OcrModelTier.PLATE

            // 提取文本框区域并做防错校验
            val roiMat: Mat
            val cropWidth: Int
            val cropHeight: Int

            if (isPlate && box.corners != null && box.corners.size == 4) {
                // PLATE 模式 + 有四边形顶点 → 透视矫正为正矩形
                val corners = box.corners!!
                // 计算透视矫正后的目标尺寸（根据四边形的平均宽高）
                val topWidth = Math.hypot(
                    corners[1].x - corners[0].x,
                    corners[1].y - corners[0].y
                )
                val bottomWidth = Math.hypot(
                    corners[2].x - corners[3].x,
                    corners[2].y - corners[3].y
                )
                val leftHeight = Math.hypot(
                    corners[3].x - corners[0].x,
                    corners[3].y - corners[0].y
                )
                val rightHeight = Math.hypot(
                    corners[2].x - corners[1].x,
                    corners[2].y - corners[1].y
                )
                val targetWidth = ((topWidth + bottomWidth) / 2.0).toInt()
                val targetHeight = ((leftHeight + rightHeight) / 2.0).toInt()

                if (targetWidth <= 0 || targetHeight <= 0) {
                    Log.w(TAG, "透视矫正尺寸无效 (w=$targetWidth, h=$targetHeight)，使用矩形裁剪")
                    val rectW = (box.right - box.left).toInt()
                    val rectH = (box.bottom - box.top).toInt()
                    if (rectW <= 0 || rectH <= 0) return null
                    roiMat = Mat(mat, org.opencv.core.Rect(box.left.toInt(), box.top.toInt(), rectW, rectH))
                    cropWidth = rectW
                    cropHeight = rectH
                } else {
                    // 源点：左上、右上、右下、左下
                    val srcPoints = MatOfPoint2f(
                        corners[0], corners[1], corners[2], corners[3]
                    )
                    // 目标点：对应位置的正矩形
                    val dstPoints = MatOfPoint2f(
                        Point(0.0, 0.0),
                        Point((targetWidth - 1).toDouble(), 0.0),
                        Point((targetWidth - 1).toDouble(), (targetHeight - 1).toDouble()),
                        Point(0.0, (targetHeight - 1).toDouble())
                    )
                    val perspectiveMat = Imgproc.getPerspectiveTransform(srcPoints, dstPoints)
                    val warped = Mat()
                    Imgproc.warpPerspective(
                        mat, warped, perspectiveMat,
                        Size(targetWidth.toDouble(), targetHeight.toDouble()),
                        Imgproc.INTER_LINEAR + Imgproc.CV_WARP_FILL_OUTLIERS,
                        Core.BORDER_CONSTANT,
                        Scalar(0.0, 0.0, 0.0)
                    )
                    perspectiveMat.release()
                    srcPoints.release()
                    dstPoints.release()
                    roiMat = warped
                    cropWidth = targetWidth
                    cropHeight = targetHeight
                }
            } else {
                // 普通矩形裁剪
                val rectWidth = (box.right - box.left).toInt()
                val rectHeight = (box.bottom - box.top).toInt()
                if (rectWidth <= 0 || rectHeight <= 0) {
                    Log.w(TAG, "跳过无效的文本框 (width=$rectWidth, height=$rectHeight)")
                    return null
                }
                val roiRect = org.opencv.core.Rect(
                    box.left.toInt(),
                    box.top.toInt(),
                    rectWidth,
                    rectHeight
                )
                roiMat = Mat(mat, roiRect)
                cropWidth = rectWidth
                cropHeight = rectHeight
            }

            // Resize parameters
            val inputHeight = 48
            val inputWidth: Int
            val resizedW: Int
            val resizedMat = Mat()

            if (isPlate) {
                // Keep aspect ratio: resize to height=48, width<=320, then pad right
                inputWidth = 320
                val ratio = cropWidth.toFloat() / cropHeight
                resizedW = minOf(kotlin.math.ceil(inputHeight * ratio).toInt(), inputWidth)
                val tempResized = Mat()
                Imgproc.resize(roiMat, tempResized,
                    Size(resizedW.toDouble(), inputHeight.toDouble()))
                resizedMat.create(inputHeight, inputWidth, tempResized.type())
                resizedMat.setTo(org.opencv.core.Scalar(0.0, 0.0, 0.0))
                val roi = resizedMat.submat(0, inputHeight, 0, resizedW)
                tempResized.copyTo(roi)
                tempResized.release()
            } else {
                // Original: force resize to 48x320
                inputWidth = 320
                resizedW = inputWidth
                Imgproc.resize(roiMat, resizedMat,
                    Size(inputWidth.toDouble(), inputHeight.toDouble()))
            }

            // 归一化
            val normalized = Mat()
            resizedMat.convertTo(normalized, CvType.CV_32FC3, 1.0 / 255.0)

            // 准备输入数据，NCHW
            val inputData = FloatArray(3 * inputHeight * inputWidth)
            val tempBuffer = FloatArray(inputHeight * inputWidth * 3)
            normalized.get(0, 0, tempBuffer)

            if (isPlate) {
                // PLATE: (x-0.5)/0.5 normalization + BGR channel order (2-c)
                for (c in 0 until 3) {
                    val offset = c * inputHeight * inputWidth
                    val colorChannelIdx = 2 - c
                    for (h in 0 until inputHeight) {
                        val rowOffset = h * inputWidth
                        for (w in 0 until inputWidth) {
                            val srcIdx = (rowOffset + w) * 3
                            val value = tempBuffer[srcIdx + colorChannelIdx]
                            inputData[offset + rowOffset + w] = (value - 0.5f) / 0.5f
                        }
                    }
                }
            } else {
                // Original: (x-0.5)/0.5 with RGB channel order (c directly)
                for (c in 0 until 3) {
                    val offset = c * inputHeight * inputWidth
                    for (h in 0 until inputHeight) {
                        val rowOffset = h * inputWidth
                        for (w in 0 until inputWidth) {
                            val srcIdx = (rowOffset + w) * 3
                            val value = tempBuffer[srcIdx + c]
                            inputData[offset + rowOffset + w] = (value - 0.5f) / 0.5f
                        }
                    }
                }
            }

            // 设置输入
            val inputTensor = predictor.getInput(0)
            inputTensor.resize(longArrayOf(1, 3, inputHeight.toLong(), inputWidth.toLong()))
            inputTensor.setData(inputData)

            // 执行推理
            predictor.run()

            // 获取输出并解码
            val outputTensor = predictor.getOutput(0)
            val outputData = outputTensor.getFloatData()
            val outputShape = outputTensor.shape()
            Log.d(TAG, "识别输出shape: ${outputShape.joinToString("x")}, dataSize=${outputData.size}, dictSize=${charSet.size}")

            // 解码 CTC 输出
            val text = decodeCTC(outputData, outputShape)

            // 清理
            roiMat.release()
            resizedMat.release()
            normalized.release()

            return text

        } catch (e: Exception) {
            Log.e(TAG, "文本识别失败", e)
        }

        return null
    }

    /**
     * 解码 CTC 输出
     */
    private fun decodeCTC(outputData: FloatArray, outputShape: LongArray): String {
        val result = StringBuilder()

        val classCount = when {
            outputShape.isNotEmpty() && outputShape.last() > 0 -> outputShape.last().toInt()
            charSet.isNotEmpty() -> charSet.size + 1
            else -> return ""
        }
        val timeSteps = outputData.size / classCount
        val blankIndex = 0
        var prevIndex = -1

        Log.d(TAG, "CTC解码: timeSteps=$timeSteps, classCount=$classCount, blankIndex=$blankIndex")
        if (classCount != charSet.size + 1 && classCount != charSet.size + 2) {
            Log.w(TAG, "识别模型类别数($classCount)与当前字典大小(${charSet.size})不匹配，请使用训练/导出该recognize.nb时对应的字典")
        }

        for (t in 0 until timeSteps) {
            var maxIndex = 0
            var maxValue = Float.NEGATIVE_INFINITY

            for (c in 0 until classCount) {
                val value = outputData[t * classCount + c]
                if (value > maxValue) {
                    maxValue = value
                    maxIndex = c
                }
            }

            if (maxIndex != blankIndex && maxIndex != prevIndex) {
                val charIndex = maxIndex - 1
                // 改进边界检查，处理字典大小不匹配的情况
                if (charIndex >= 0 && charIndex < charSet.size) {
                    result.append(charSet[charIndex])
                } else if (charIndex < 0 && maxIndex < charSet.size) {
                    // 尝试使用原始索引（可能模型没有blank）
                    result.append(charSet[maxIndex])
                }
            }

            prevIndex = maxIndex
        }

        return result.toString()
    }

    /**
     * 过滤 VIN 码字符（已禁用，使用全字符识别）
     */
    private fun filterVINChars(text: String): String {
        // 全字符识别，不做任何过滤
        return text.trim()
    }

    /**
     * 释放资源
     */
    override fun release() {
        try {
            detPredictor = null
            recPredictor = null
            isInitialized = false
            Log.d(TAG, "PaddleOCR 资源已释放")
        } catch (e: Exception) {
            Log.e(TAG, "释放资源失败", e)
        }
    }

    override fun isInitialized(): Boolean {
        val currentTier = OcrConfig.ocrModelTier.value ?: OcrConfig.OcrModelTier.CHINESE
        return isInitialized && detPredictor != null && recPredictor != null && loadedModelTier == currentTier
    }
}

/**
 * 文本框数据类
 */
data class TextBox(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
    /**
     * 四边形四个顶点（顺时针：左上→右上→右下→左下），
     * 用于透视矫正。为 null 时退化为轴对齐矩形裁剪。
     */
    val corners: Array<Point>? = null
) {
    // 保持 equals/hashCode 兼容（corners 不参与比较，避免影响已有逻辑）
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is TextBox) return false
        return left == other.left && top == other.top && right == other.right && bottom == other.bottom
    }

    override fun hashCode(): Int {
        var result = left.hashCode()
        result = 31 * result + top.hashCode()
        result = 31 * result + right.hashCode()
        result = 31 * result + bottom.hashCode()
        return result
    }
}