package com.seuic.floating.ui.activity

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.view.Gravity
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.Spinner
import android.widget.Switch
import androidx.appcompat.app.AppCompatActivity
import com.seuic.floating.R
import com.seuic.floating.config.OcrConfig
import com.seuic.floating.service.BaseInitializer
import com.seuic.floating.service.FloatingButtonService
import com.seuic.util.common.PermissionUtils
import com.seuic.util.common.ToastUtils
import com.seuic.util.common.constant.PermissionConstants

class SettingsActivity : AppCompatActivity() {

    private lateinit var spinnerScannerType: Spinner
    private lateinit var spinnerKeyAreaMode: Spinner
    private lateinit var spinnerModelTier: Spinner
    private lateinit var spinnerRecognitionRegionMode: Spinner
    private lateinit var switchFloatingButton: Switch
    private lateinit var switchScannerImageUpsideDown: Switch
    private lateinit var layoutRegexSettings: View
    private lateinit var textRegexValue: android.widget.TextView
    private lateinit var btnDeleteRegex: Button
    // 保存原始值用于比较
    private var originalKeyAreaMode: Int = 0
    private var originalFloatingButtonState: Boolean = false
    private var originalRecognitionRegionMode: Int = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_settings)

        initViews()
        loadCurrentSettings()
        setupRealTimeListeners()

        requestPermission()
        observeOcrResults()

        // CCPD 批量测试入口
        findViewById<Button>(R.id.btn_ccpd_test).setOnClickListener {
            startActivity(Intent(this, TestActivity::class.java))
        }
    }

    private fun observeOcrResults() {
        val textOcrTime = findViewById<android.widget.TextView>(R.id.text_ocr_time)
        val textOcrResult = findViewById<android.widget.TextView>(R.id.text_ocr_result)
        val imageOcrAnnotated = findViewById<android.widget.ImageView>(R.id.image_ocr_annotated)

        OcrConfig.lastRecognitionTime.observe(this) { time ->
            textOcrTime?.text = "识别耗时: ${time} ms"
        }

        OcrConfig.lastOcrResult.observe(this) { result ->
            textOcrResult?.text = "识别结果: ${result ?: "--"}"
        }

        OcrConfig.lastRecognizedBitmap.observe(this) { bitmap ->
            if (bitmap != null) {
                imageOcrAnnotated?.setImageBitmap(bitmap)
            } else {
                imageOcrAnnotated?.setImageDrawable(null)
            }
        }

        OcrConfig.regexPattern.observe(this) { pattern ->
            if (pattern.isNullOrEmpty()) {
                textRegexValue.text = "未设置"
                btnDeleteRegex.visibility = View.GONE
            } else {
                textRegexValue.text = pattern
                btnDeleteRegex.visibility = View.VISIBLE
            }
        }
    }
    private fun initViews() {
        spinnerScannerType = findViewById(R.id.spinner_scanner_type)
        spinnerKeyAreaMode = findViewById(R.id.spinner_key_area_mode)
        spinnerModelTier = findViewById(R.id.spinner_model_tier)
        spinnerRecognitionRegionMode = findViewById(R.id.spinner_recognition_region_mode)
        switchFloatingButton = findViewById(R.id.switch_floating_button)
        switchScannerImageUpsideDown = findViewById(R.id.switch_scanner_image_upside_down)
        layoutRegexSettings = findViewById(R.id.layout_regex_settings)
        textRegexValue = findViewById(R.id.text_regex_value)
        btnDeleteRegex = findViewById(R.id.btn_delete_regex)

        // 初始化Model selection
        initModelTierSpinner()

        // 初始化扫描器选择器
        initScannerTypeSpinner()

        // 按键区域模式spinner适配器
        val keyAreaModeAdapter = ArrayAdapter.createFromResource(
            this,
            R.array.key_area_modes,
            android.R.layout.simple_spinner_item
        ).apply {
            setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        }
        spinnerKeyAreaMode.adapter = keyAreaModeAdapter

        // 识别区域模式spinner适配器
        val recognitionRegionModeAdapter = ArrayAdapter.createFromResource(
            this,
            R.array.recognition_region_modes,
            android.R.layout.simple_spinner_item
        ).apply {
            setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        }
        spinnerRecognitionRegionMode.adapter = recognitionRegionModeAdapter
    }

    private fun loadCurrentSettings() {
        originalKeyAreaMode = OcrConfig.keyAreaMode.value ?: 0
        originalFloatingButtonState = OcrConfig.floatingButtonEnabled.value ?: false
        originalRecognitionRegionMode = OcrConfig.recognitionRegionMode.value?.modeId ?: 0

        spinnerKeyAreaMode.setSelection(originalKeyAreaMode)
        switchFloatingButton.isChecked = originalFloatingButtonState
        spinnerRecognitionRegionMode.setSelection(originalRecognitionRegionMode)
    }

    // 新增：设置实时监听器
    private fun setupRealTimeListeners() {
        spinnerKeyAreaMode.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                if (position != originalKeyAreaMode) {
                    originalKeyAreaMode = position
                    OcrConfig.keyAreaMode.value = position
                    ToastUtils.showShort("按键控制已更新")
                }
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }

        // 新增：识别区域选择监听
        spinnerRecognitionRegionMode.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                val selectedMode = OcrConfig.RecognitionRegionMode.fromModeId(position)
                if (selectedMode != OcrConfig.recognitionRegionMode.value) {
                    originalRecognitionRegionMode = position
                    OcrConfig.recognitionRegionMode.value = selectedMode
                    ToastUtils.showShort("已更新识别区域模式: ${selectedMode.displayName}")
                }
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }

        // 新增：悬浮按钮开关监听
        switchFloatingButton.setOnCheckedChangeListener { _, isChecked ->
            if (isChecked != originalFloatingButtonState) {
                applyFloatingButtonSettings(isChecked)
                originalFloatingButtonState = isChecked
            }
        }

        // 扫描头图像颠倒开关监听
        switchScannerImageUpsideDown.isChecked = OcrConfig.scannerImageUpsideDown.value ?: false
        switchScannerImageUpsideDown.setOnCheckedChangeListener { _, isChecked ->
            OcrConfig.scannerImageUpsideDown.value = isChecked
            ToastUtils.showShort(if (isChecked) "扫描头图像颠倒已开启" else "扫描头图像颠倒已关闭")
        }

        // 新增：正则表达式设置项点击
        layoutRegexSettings.setOnClickListener {
            showRegexOptionDialog()
        }

        // 新增：删除正则表达式点击
        btnDeleteRegex.setOnClickListener {
            androidx.appcompat.app.AlertDialog.Builder(this)
                .setTitle("提示")
                .setMessage("确定删除当前的正则表达式校验规则吗？")
                .setPositiveButton("确定") { _, _ ->
                    OcrConfig.regexPattern.value = ""
                    ToastUtils.showShort("正则表达式已删除")
                }
                .setNegativeButton("取消", null)
                .show()
        }
    }



    private fun initModelTierSpinner() {
        val tierNames = OcrConfig.OcrModelTier.values().map { it.displayName }
        val adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_item,
            tierNames
        ).apply {
            setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        }

        spinnerModelTier.adapter = adapter

        // 设置当前选中项
        val currentTier = OcrConfig.ocrModelTier.value ?: OcrConfig.OcrModelTier.CHINESE
        val currentIndex = OcrConfig.OcrModelTier.values().indexOf(currentTier)
        spinnerModelTier.setSelection(currentIndex)

        // 设置选择监听器
        spinnerModelTier.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>, view: View?, position: Int, id: Long) {
                val selectedTier = OcrConfig.OcrModelTier.values()[position]
                if (selectedTier != OcrConfig.ocrModelTier.value) {
                    OcrConfig.ocrModelTier.value = selectedTier
                    ToastUtils.showShort("已切换到: ${selectedTier.displayName}")
                }
            }

            override fun onNothingSelected(parent: AdapterView<*>) {}
        }
    }

    private fun initScannerTypeSpinner() {
        val typeNames = OcrConfig.ScannerType.values().map { it.displayName }
        val adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_item,
            typeNames
        ).apply {
            setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        }

        spinnerScannerType.adapter = adapter

        // 设置当前选中项
        val currentType = OcrConfig.selectedScanner.value ?: OcrConfig.ScannerType.CAMERA
        val currentIndex = OcrConfig.ScannerType.values().indexOf(currentType)
        spinnerScannerType.setSelection(currentIndex)

        // 设置选择监听器
        spinnerScannerType.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>, view: View?, position: Int, id: Long) {
                val selectedType = OcrConfig.ScannerType.values()[position]
                if (selectedType != OcrConfig.selectedScanner.value) {
                    OcrConfig.selectedScanner.value = selectedType
                }
            }

            override fun onNothingSelected(parent: AdapterView<*>) {}
        }
    }

    // 新增：应用悬浮按钮设置（实时生效）
    private fun applyFloatingButtonSettings(isEnabled: Boolean) {
        // 直接设置 OcrConfig 值，会自动保存
        OcrConfig.floatingButtonEnabled.value = isEnabled

        if (isEnabled) {
            // 启动悬浮按钮服务
            startFloatingButtonService()
        } else {
            // 停止悬浮按钮服务
            stopFloatingButtonService()
        }

        ToastUtils.showShort(if (isEnabled) "悬浮按钮已启用" else "悬浮按钮已禁用")
    }

    // 新增：启动悬浮按钮服务
    private fun startFloatingButtonService() {
        val intent = Intent(this, FloatingButtonService::class.java)
        startService(intent)
    }

    // 新增：停止悬浮按钮服务
    private fun stopFloatingButtonService() {
        val intent = Intent(this, FloatingButtonService::class.java)
        stopService(intent)
    }



    private fun requestPermission() {
        PermissionUtils.permission(
            PermissionConstants.CAMERA,
            PermissionConstants.STORAGE
        )
            .callback(object : PermissionUtils.SimpleCallback {
                override fun onGranted() {
                    Log.d(
                        "初始化进来了",
                        "初始化成功"
                    )
                }

                override fun onDenied() {
                    // 建议添加权限拒绝的处理逻辑
                    ToastUtils.showShort("权限被拒绝，无法使用相机功能")
                }
            })
            .request()
    }

    private fun showRegexOptionDialog() {
        val options = arrayOf("简易版 (通过方框按钮生成)", "专业版 (手动输入正则表达式)")
        androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle("选择配置模式")
            .setItems(options) { _, which ->
                if (which == 0) {
                    showEasyRegexDialog()
                } else {
                    showProfessionalRegexDialog()
                }
            }
            .show()
    }

    data class RegexBlock(val displayName: String, val regexValue: String)

    private fun parseRegexToBlocks(regexStr: String): List<RegexBlock> {
        val blocks = mutableListOf<RegexBlock>()
        var i = 0
        val len = regexStr.length
        while (i < len) {
            if (regexStr.startsWith("[\\u4e00-\\u9fa5a-zA-Z0-9]", i)) {
                blocks.add(RegexBlock("汉字/字母/数字", "[\\u4e00-\\u9fa5a-zA-Z0-9]"))
                i += 25
            } else if (regexStr.startsWith("[\\u4e00-\\u9fa5a-zA-Z]", i)) {
                blocks.add(RegexBlock("汉字/字母", "[\\u4e00-\\u9fa5a-zA-Z]"))
                i += 22
            } else if (regexStr.startsWith("[\\u4e00-\\u9fa50-9]", i)) {
                blocks.add(RegexBlock("汉字/数字", "[\\u4e00-\\u9fa50-9]"))
                i += 19
            } else if (regexStr.startsWith("[\\u4e00-\\u9fa5]", i)) {
                blocks.add(RegexBlock("中文字符", "[\\u4e00-\\u9fa5]"))
                i += 16
            } else if (regexStr.startsWith("[\u4e00-\u9fa5]", i)) {
                blocks.add(RegexBlock("中文字符", "[\\u4e00-\\u9fa5]"))
                i += 15
            } else if (regexStr.startsWith("[\\u3002\\uff0c\\u3001\\uff1b\\uff1a\\uff1f\\uff01\\u201c\\u201d\\u300a\\u300b]", i)) {
                blocks.add(RegexBlock("中文标点", "[\\u3002\\uff0c\\u3001\\uff1b\\uff1a\\uff1f\\uff01\\u201c\\u201d\\u300a\\u300b]"))
                i += 58
            } else if (regexStr.startsWith("[0-9]", i)) {
                blocks.add(RegexBlock("0-9", "[0-9]"))
                i += 5
            } else if (regexStr.startsWith("[A-Z]", i)) {
                blocks.add(RegexBlock("A-Z", "[A-Z]"))
                i += 5
            } else if (regexStr.startsWith("[a-z]", i)) {
                blocks.add(RegexBlock("a-z", "[a-z]"))
                i += 5
            } else if (regexStr.startsWith("\\*", i)) {
                blocks.add(RegexBlock("*", "\\*"))
                i += 2
            } else if (regexStr.startsWith("\\+", i)) {
                blocks.add(RegexBlock("+", "\\+"))
                i += 2
            } else if (regexStr.startsWith("\\$", i)) {
                blocks.add(RegexBlock("$", "\\$"))
                i += 2
            } else if (regexStr.startsWith("\\.", i)) {
                blocks.add(RegexBlock(".", "\\."))
                i += 2
            } else if (regexStr.startsWith("\\/", i)) {
                blocks.add(RegexBlock("/", "\\/"))
                i += 2
            } else if (regexStr.startsWith("\\\\", i)) {
                blocks.add(RegexBlock("\\", "\\\\"))
                i += 2
            } else if (regexStr.startsWith("\\s", i)) {
                blocks.add(RegexBlock("空格", "\\s"))
                i += 2
            } else {
                val charStr = regexStr[i].toString()
                val block = when (charStr) {
                    ":" -> RegexBlock("：", ":")
                    "-" -> RegexBlock("-", "-")
                    "," -> RegexBlock(",", ",")
                    "#" -> RegexBlock("#", "#")
                    "=" -> RegexBlock("=", "=")
                    ";" -> RegexBlock(";", ";")
                    " " -> RegexBlock("空格", "\\s")
                    "省" -> RegexBlock("省", "省")
                    "市" -> RegexBlock("市", "市")
                    "区" -> RegexBlock("区", "区")
                    "县" -> RegexBlock("县", "县")
                    "号" -> RegexBlock("号", "号")
                    "元" -> RegexBlock("元", "元")
                    "个" -> RegexBlock("个", "个")
                    "件" -> RegexBlock("件", "件")
                    else -> RegexBlock(charStr, charStr)
                }
                blocks.add(block)
                i += 1
            }
        }
        return blocks
    }

    private fun showEasyRegexDialog() {
        val dialogView = layoutInflater.inflate(R.layout.dialog_regex_easy, null)
        val btnClearSequence = dialogView.findViewById<android.widget.Button>(R.id.btn_clear_sequence)
        val scrollSelectedSequence = dialogView.findViewById<android.widget.HorizontalScrollView>(R.id.scroll_selected_sequence)
        val layoutSelectedSequence = dialogView.findViewById<android.widget.LinearLayout>(R.id.layout_selected_sequence)
        val tvRegexPreview = dialogView.findViewById<android.widget.TextView>(R.id.tv_regex_preview)

        val selectedBlocks = mutableListOf<RegexBlock>()

        fun dpToPx(dp: Int): Int {
            val density = resources.displayMetrics.density
            return (dp * density).toInt()
        }

        fun updatePreview(): String {
            val pattern = selectedBlocks.joinToString("") { it.regexValue }
            tvRegexPreview.text = "实时预览: $pattern"
            return pattern
        }

        fun renderSequence() {
            layoutSelectedSequence.removeAllViews()
            for (index in selectedBlocks.indices) {
                val block = selectedBlocks[index]
                val textView = android.widget.TextView(this).apply {
                    text = block.displayName
                    textSize = 14f
                    setTextColor(android.graphics.Color.parseColor("#333333"))
                    setPadding(dpToPx(8), dpToPx(6), dpToPx(8), dpToPx(6))
                    
                    val drawable = android.graphics.drawable.GradientDrawable().apply {
                        setColor(android.graphics.Color.parseColor("#E0E0E0"))
                        setStroke(dpToPx(1), android.graphics.Color.parseColor("#BDBDBD"))
                        cornerRadius = dpToPx(4).toFloat()
                    }
                    background = drawable
                    
                    val params = android.widget.LinearLayout.LayoutParams(
                        android.widget.LinearLayout.LayoutParams.WRAP_CONTENT,
                        android.widget.LinearLayout.LayoutParams.WRAP_CONTENT
                    ).apply {
                        rightMargin = dpToPx(8)
                    }
                    layoutParams = params
                    
                    setOnClickListener {
                        selectedBlocks.removeAt(index)
                        renderSequence()
                    }
                }
                layoutSelectedSequence.addView(textView)
            }

            scrollSelectedSequence.post {
                scrollSelectedSequence.fullScroll(android.view.View.FOCUS_RIGHT)
            }

            updatePreview()
        }

        // 初始化已有正则
        val currentPattern = OcrConfig.regexPattern.value ?: ""
        if (currentPattern.isNotEmpty()) {
            selectedBlocks.addAll(parseRegexToBlocks(currentPattern))
        }

        renderSequence()

        // 绑定按钮点击事件
        val btnMap = mapOf(
            R.id.btn_type_digits to RegexBlock("0-9", "[0-9]"),
            R.id.btn_type_uppercase to RegexBlock("A-Z", "[A-Z]"),
            R.id.btn_type_lowercase to RegexBlock("a-z", "[a-z]"),
            R.id.btn_type_chinese to RegexBlock("中文字符", "[\\u4e00-\\u9fa5]"),
            R.id.btn_type_chn_digit to RegexBlock("汉字/数字", "[\\u4e00-\\u9fa50-9]"),
            R.id.btn_type_chn_letter to RegexBlock("汉字/字母", "[\\u4e00-\\u9fa5a-zA-Z]"),
            R.id.btn_type_chn_mix to RegexBlock("汉字/字母/数字", "[\\u4e00-\\u9fa5a-zA-Z0-9]"),
            R.id.btn_type_chn_punct to RegexBlock("中文标点", "[\\u3002\\uff0c\\u3001\\uff1b\\uff1a\\uff1f\\uff01\\u201c\\u201d\\u300a\\u300b]"),
            R.id.btn_suf_prov to RegexBlock("省", "省"),
            R.id.btn_suf_city to RegexBlock("市", "市"),
            R.id.btn_suf_dist to RegexBlock("区", "区"),
            R.id.btn_suf_county to RegexBlock("县", "县"),
            R.id.btn_suf_no to RegexBlock("号", "号"),
            R.id.btn_suf_yuan to RegexBlock("元", "元"),
            R.id.btn_suf_unit to RegexBlock("个", "个"),
            R.id.btn_suf_pkg to RegexBlock("件", "件"),
            R.id.btn_sym_colon to RegexBlock("：", ":"),
            R.id.btn_sym_hyphen to RegexBlock("-", "-"),
            R.id.btn_sym_asterisk to RegexBlock("*", "\\*"),
            R.id.btn_sym_comma to RegexBlock(",", ","),
            R.id.btn_sym_hash to RegexBlock("#", "#"),
            R.id.btn_sym_plus to RegexBlock("+", "\\+"),
            R.id.btn_sym_equals to RegexBlock("=", "="),
            R.id.btn_sym_semicolon to RegexBlock(";", ";"),
            R.id.btn_sym_dollar to RegexBlock("$", "\\$"),
            R.id.btn_sym_dot to RegexBlock(".", "\\."),
            R.id.btn_sym_slash to RegexBlock("/", "\\/"),
            R.id.btn_sym_space to RegexBlock("空格", "\\s")
        )

        for ((id, block) in btnMap) {
            dialogView.findViewById<android.widget.Button>(id)?.setOnClickListener {
                selectedBlocks.add(block)
                renderSequence()
            }
        }

        btnClearSequence.setOnClickListener {
            selectedBlocks.clear()
            renderSequence()
        }

        androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle("简易版配置 (选框代表字符)")
            .setView(dialogView)
            .setPositiveButton("确定") { _, _ ->
                val generatedPattern = updatePreview()
                OcrConfig.regexPattern.value = generatedPattern
                if (generatedPattern.isNotEmpty()) {
                    ToastUtils.showShort("已应用正则: $generatedPattern")
                } else {
                    ToastUtils.showShort("已清空正则表达式")
                }
            }
            .setNegativeButton("取消", null)
            .show()
    }

    private fun showProfessionalRegexDialog() {
        val dialogView = layoutInflater.inflate(R.layout.dialog_regex_professional, null)
        val etRegex = dialogView.findViewById<android.widget.EditText>(R.id.et_regex_professional)

        val currentPattern = OcrConfig.regexPattern.value ?: ""
        etRegex.setText(currentPattern)
        etRegex.setSelection(currentPattern.length)

        androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle("专业版配置")
            .setView(dialogView)
            .setPositiveButton("确定") { _, _ ->
                val newPattern = etRegex.text.toString().trim()
                if (newPattern.isNotEmpty()) {
                    try {
                        java.util.regex.Pattern.compile(newPattern)
                        OcrConfig.regexPattern.value = newPattern
                        ToastUtils.showShort("已保存正则表达式")
                    } catch (e: Exception) {
                        ToastUtils.showShort("不合法的正则表达式: ${e.message}")
                    }
                } else {
                    OcrConfig.regexPattern.value = ""
                    ToastUtils.showShort("正则表达式已清除")
                }
            }
            .setNegativeButton("取消", null)
            .show()
    }
}
