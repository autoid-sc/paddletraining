package com.seuic.floating.ui.activity

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.seuic.floating.R
import com.seuic.floating.config.OcrConfig
import com.seuic.floating.network.PaddleOCRService
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class TestActivity : AppCompatActivity() {

    private lateinit var btnStart: Button
    private lateinit var textProgress: TextView
    private lateinit var textSummary: TextView
    private lateinit var layoutResults: LinearLayout

    companion object {
        private const val TAG = "TestActivity"
        private const val TEST_DIR = "test_plates"

        private val PROVINCES = listOf(
            "皖","沪","津","渝","冀","晋","蒙","辽","吉","黑",
            "苏","浙","京","闽","赣","鲁","豫","鄂","湘","粤",
            "桂","琼","川","贵","云","藏","陕","甘","青","宁","新"
        )
        private val ALPHABETS = listOf(
            "A","B","C","D","E","F","G","H","J","K",
            "L","M","N","P","Q","R","S","T","U","V",
            "W","X","Y","Z"
        )
        private val ADS = listOf(
            "A","B","C","D","E","F","G","H","J","K",
            "L","M","N","P","Q","R","S","T","U","V",
            "W","X","Y","Z","0","1","2","3","4","5",
            "6","7","8","9","O"
        )
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_test)

        btnStart = findViewById(R.id.btn_start_test)
        textProgress = findViewById(R.id.text_progress)
        textSummary = findViewById(R.id.text_summary)
        layoutResults = findViewById(R.id.layout_results)

        btnStart.setOnClickListener { startTest() }
    }

    private fun decodeCCPDPlate(filename: String): String {
        return try {
            val name = filename.removeSuffix(".jpg")
            val parts = name.split("-")
            if (parts.size < 5) return ""
            val indices = parts[4].split("_").map { it.toIntOrNull() ?: return "" }
            if (indices.size < 7) return ""
            val prov = PROVINCES.getOrElse(indices[0]) { return "" }
            val alpha = ALPHABETS.getOrElse(indices[1]) { return "" }
            val rest = indices.drop(2).joinToString("") { idx ->
                val c = ADS.getOrElse(idx) { "" }
                if (c == "O") "" else c
            }
            "$prov$alpha$rest"
        } catch (e: Exception) {
            ""
        }
    }

    private fun loadTestImages(): List<Pair<String, Bitmap>> {
        val images = mutableListOf<Pair<String, Bitmap>>()
        try {
            val files = assets.list(TEST_DIR) ?: return emptyList()
            for (file in files.sorted()) {
                if (!file.endsWith(".jpg")) continue
                val stream = assets.open("$TEST_DIR/$file")
                val bitmap = BitmapFactory.decodeStream(stream)
                stream.close()
                if (bitmap != null) {
                    images.add(Pair(file, bitmap))
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "加载测试图片失败", e)
        }
        return images
    }

    private fun startTest() {
        btnStart.isEnabled = false
        layoutResults.removeAllViews()
        textSummary.visibility = android.view.View.GONE
        textProgress.text = "加载图片中..."

        lifecycleScope.launch {
            val images = withContext(Dispatchers.IO) { loadTestImages() }
            if (images.isEmpty()) {
                textProgress.text = "未找到测试图片！请确认 assets/test_plates/ 目录有 .jpg 文件"
                btnStart.isEnabled = true
                return@launch
            }

            textProgress.text = "共 ${images.size} 张图片，开始测试..."
            val ocrService = PaddleOCRService.getInstance(this@TestActivity)

            // 保存用户当前模型选择
            val originalTier = OcrConfig.ocrModelTier.value

            // ===== 第一轮：车牌模型 =====
            textProgress.text = "初始化车牌模型..."
            OcrConfig.ocrModelTier.value = OcrConfig.OcrModelTier.PLATE
            ocrService.release()
            ocrService.initialize()

            val plateResults = mutableListOf<Pair<String?, Long>>()
            Log.i(TAG, "========== 车牌模型识别开始 (${images.size}张) ==========")
            for ((i, pair) in images.withIndex()) {
                val gt = decodeCCPDPlate(pair.first)
                textProgress.text = "车牌模型 [${i + 1}/${images.size}] $gt"
                var result: String? = null
                val t0 = System.currentTimeMillis()
                ocrService.recognizeImage(pair.second) { r -> result = r }
                val elapsed = System.currentTimeMillis() - t0
                plateResults.add(Pair(result, elapsed))
                Log.d(TAG, "  车牌[${i + 1}] ${pair.first.take(20)}... 真实=$gt 识别=${result ?: "(空)"} ${elapsed}ms")
            }
            Log.i(TAG, "========== 车牌模型识别完成 ==========")

            // ===== 第二轮：V2版本模型 =====
            textProgress.text = "初始化V2版本模型..."
            OcrConfig.ocrModelTier.value = OcrConfig.OcrModelTier.CHINESE
            ocrService.release()
            ocrService.initialize()

            val chineseResults = mutableListOf<Pair<String?, Long>>()
            Log.i(TAG, "========== V2版本模型识别开始 (${images.size}张) ==========")
            for ((i, pair) in images.withIndex()) {
                val gt = decodeCCPDPlate(pair.first)
                textProgress.text = "V2版本模型 [${i + 1}/${images.size}] $gt"
                var result: String? = null
                val t0 = System.currentTimeMillis()
                ocrService.recognizeImage(pair.second) { r -> result = r }
                val elapsed = System.currentTimeMillis() - t0
                chineseResults.add(Pair(result, elapsed))
                Log.d(TAG, "  V2  [${i + 1}] ${pair.first.take(20)}... 真实=$gt 识别=${result ?: "(空)"} ${elapsed}ms")
            }
            Log.i(TAG, "========== V2版本模型识别完成 ==========")

            // 恢复用户原始模型
            OcrConfig.ocrModelTier.value = originalTier
            ocrService.release()
            ocrService.initialize()

            // ===== 显示结果 =====
            var plateCorrect = 0
            var chineseCorrect = 0
            var plateCharCorrect = 0
            var chineseCharCorrect = 0
            var totalChars = 0
            var plateTotalTime = 0L
            var chineseTotalTime = 0L

            for ((i, pair) in images.withIndex()) {
                val groundTruth = decodeCCPDPlate(pair.first)
                val (plateText, plateTime) = plateResults[i]
                val (chineseText, chineseTime) = chineseResults[i]
                val plateRaw = plateText ?: ""
                val chineseRaw = chineseText ?: ""
                // 去掉·分隔符和常见误识别的分隔符再比较
                val plateResult = plateRaw.trim().replace("·", "")
                val chineseResult = chineseRaw.trim().replace(Regex("[·.\\-]"), "")
                plateTotalTime += plateTime
                chineseTotalTime += chineseTime

                val plateMatch = plateResult == groundTruth
                val chineseMatch = chineseResult == groundTruth
                if (plateMatch) plateCorrect++
                if (chineseMatch) chineseCorrect++

                // 字符级统计
                totalChars += groundTruth.length
                plateCharCorrect += countMatchingChars(groundTruth, plateResult)
                chineseCharCorrect += countMatchingChars(groundTruth, chineseResult)

                addResultRow(i + 1, groundTruth, plateResult, chineseResult, plateMatch, chineseMatch, plateTime, chineseTime)

                // 详细日志：每张图的识别结果 + 耗时 + 是否正确
                val plateTag = if (plateMatch) "✓" else "✗"
                val chineseTag = if (chineseMatch) "✓" else "✗"
                Log.d(TAG, "[${i + 1}/${images.size}] 真实=$groundTruth | 车牌=${plateResult.trim().ifEmpty { "(空)" }}(${plateTime}ms)$plateTag | V2=${chineseResult.trim().ifEmpty { "(空)" }}(${chineseTime}ms)$chineseTag")
            }

            val total = images.size
            val plateAcc = if (total > 0) plateCorrect * 100.0 / total else 0.0
            val chineseAcc = if (total > 0) chineseCorrect * 100.0 / total else 0.0
            val plateCharAcc = if (totalChars > 0) plateCharCorrect * 100.0 / totalChars else 0.0
            val chineseCharAcc = if (totalChars > 0) chineseCharCorrect * 100.0 / totalChars else 0.0
            val plateAvgTime = if (total > 0) plateTotalTime / total else 0L
            val chineseAvgTime = if (total > 0) chineseTotalTime / total else 0L

            val summary = buildString {
                appendLine("===== 测试完成 (${total} 张) =====")
                appendLine()
                appendLine("【车牌模型】")
                appendLine("  完全匹配: $plateCorrect/$total = ${"%.1f".format(plateAcc)}%")
                appendLine("  字符准确率: $plateCharCorrect/$totalChars = ${"%.1f".format(plateCharAcc)}%")
                appendLine("  平均耗时: ${plateAvgTime}ms/张 (总计 ${plateTotalTime}ms)")
                appendLine()
                appendLine("【V2版本模型】")
                appendLine("  完全匹配: $chineseCorrect/$total = ${"%.1f".format(chineseAcc)}%")
                appendLine("  字符准确率: $chineseCharCorrect/$totalChars = ${"%.1f".format(chineseCharAcc)}%")
                appendLine("  平均耗时: ${chineseAvgTime}ms/张 (总计 ${chineseTotalTime}ms)")
            }

            textSummary.text = summary
            textSummary.visibility = android.view.View.VISIBLE
            textProgress.text = "测试完成！"
            btnStart.isEnabled = true

            Log.i(TAG, summary)
        }
    }

    private fun countMatchingChars(groundTruth: String, result: String): Int {
        var count = 0
        val minLen = minOf(groundTruth.length, result.length)
        for (i in 0 until minLen) {
            if (groundTruth[i] == result[i]) count++
        }
        return count
    }

    private fun addResultRow(
        index: Int,
        groundTruth: String,
        plateResult: String,
        chineseResult: String,
        plateMatch: Boolean,
        chineseMatch: Boolean,
        plateTime: Long,
        chineseTime: Long
    ) {
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(6, 4, 6, 4)
            setBackgroundColor(if (index % 2 == 0) Color.WHITE else Color.parseColor("#F5F5F5"))
        }

        val params = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT)

        fun makeCell(text: String, weight: Float, color: Int = Color.parseColor("#333333")): TextView {
            return TextView(this).apply {
                this.text = text
                textSize = 11f
                setTextColor(color)
                layoutParams = params.apply { this.weight = weight }
                setPadding(2, 2, 2, 2)
            }
        }

        row.addView(makeCell("$index", 0.4f))
        row.addView(makeCell(groundTruth, 1.8f))
        row.addView(makeCell(
            "${plateResult.ifEmpty { "(空)" }}\n${plateTime}ms",
            2f,
            if (plateMatch) Color.parseColor("#1565C0") else Color.parseColor("#D32F2F")
        ))
        row.addView(makeCell(
            "${chineseResult.ifEmpty { "(空)" }}\n${chineseTime}ms",
            2f,
            if (chineseMatch) Color.parseColor("#2E7D32") else Color.parseColor("#FF8F00")
        ))

        layoutResults.addView(row)
    }
}
