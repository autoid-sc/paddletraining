package com.seuic.floating.config

import android.content.Context
import android.content.SharedPreferences

object ConfigManager {
    private const val PREF_NAME = "ocr_config"
    private const val KEY_KEY_AREA_MODE = "key_area_mode"
    private const val KEY_FLOATING_BUTTON_ENABLED = "floating_button_enabled"
    private const val KEY_FLOATING_BUTTON_X = "floating_button_x"
    private const val KEY_FLOATING_BUTTON_Y = "floating_button_y"
    private const val KEY_OCR_ENGINE = "ocr_engine"
    private const val KEY_SCANNER_TYPE = "scanner_type"
    private const val KEY_OCR_MODEL_TIER = "ocr_model_tier"
    private const val KEY_RECOGNITION_REGION_MODE = "recognition_region_mode"
    private const val KEY_REGEX_PATTERN = "regex_pattern"
    private const val KEY_SCANNER_IMAGE_UPSIDE_DOWN = "scanner_image_upside_down"

    private var sharedPreferences: SharedPreferences? = null

    fun init(context: Context) {
        if (sharedPreferences == null) {
            sharedPreferences = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)
        }
    }

    // 保存OCR配置
    fun saveOcrConfig() {
        val prefs = sharedPreferences
        if (prefs != null) {
            with(prefs.edit()) {
                putInt(KEY_KEY_AREA_MODE, OcrConfig.keyAreaMode.value ?: 0)
                putBoolean(KEY_FLOATING_BUTTON_ENABLED, OcrConfig.floatingButtonEnabled.value ?: false)
                putInt(KEY_SCANNER_TYPE, OcrConfig.selectedScanner.value?.typeId ?: 0)
                putInt(KEY_OCR_MODEL_TIER, OcrConfig.ocrModelTier.value?.tierId ?: 0)
                putInt(KEY_RECOGNITION_REGION_MODE, OcrConfig.recognitionRegionMode.value?.modeId ?: 0)
                putString(KEY_REGEX_PATTERN, OcrConfig.regexPattern.value ?: "")
                putBoolean(KEY_SCANNER_IMAGE_UPSIDE_DOWN, OcrConfig.scannerImageUpsideDown.value ?: false)
                apply()
            }
        }
    }

    // 加载OCR配置
    fun loadOcrConfig() {
        val prefs = sharedPreferences
        if (prefs != null) {
            OcrConfig.keyAreaMode.value = prefs.getInt(KEY_KEY_AREA_MODE, 0)
            OcrConfig.floatingButtonEnabled.value = prefs.getBoolean(KEY_FLOATING_BUTTON_ENABLED, false)
            OcrConfig.selectedScanner.value = OcrConfig.ScannerType.fromTypeId(prefs.getInt(KEY_SCANNER_TYPE, 0))
            OcrConfig.ocrModelTier.value = OcrConfig.OcrModelTier.fromTierId(prefs.getInt(KEY_OCR_MODEL_TIER, 0))
            OcrConfig.recognitionRegionMode.value = OcrConfig.RecognitionRegionMode.fromModeId(prefs.getInt(KEY_RECOGNITION_REGION_MODE, 0))
            OcrConfig.regexPattern.value = prefs.getString(KEY_REGEX_PATTERN, "")
            OcrConfig.scannerImageUpsideDown.value = prefs.getBoolean(KEY_SCANNER_IMAGE_UPSIDE_DOWN, false)
        }
    }

    // 保存悬浮按钮位置
    fun saveFloatingButtonPosition(x: Int, y: Int) {
        val prefs = sharedPreferences
        if (prefs != null) {
            with(prefs.edit()) {
                putInt(KEY_FLOATING_BUTTON_X, x)
                putInt(KEY_FLOATING_BUTTON_Y, y)
                apply()
            }
        }
    }

    // 加载悬浮按钮位置
    fun loadFloatingButtonPosition(): Pair<Int, Int> {
        val prefs = sharedPreferences
        if (prefs != null) {
            val x = prefs.getInt(KEY_FLOATING_BUTTON_X, 16)
            val y = prefs.getInt(KEY_FLOATING_BUTTON_Y, 100)
            return Pair(x, y)
        }
        return Pair(16, 100)
    }

    // 保存单个配置项
    fun saveKeyAreaMode(keyAreaMode: Int) {
        sharedPreferences?.edit()?.putInt(KEY_KEY_AREA_MODE, keyAreaMode)?.apply()
    }

    fun saveFloatingButtonEnabled(enabled: Boolean) {
        sharedPreferences?.edit()?.putBoolean(KEY_FLOATING_BUTTON_ENABLED, enabled)?.apply()
    }


    // 扫描器选择相关方法
    fun saveScannerType(typeId: Int) {
        sharedPreferences?.edit()?.putInt(KEY_SCANNER_TYPE, typeId)?.apply()
    }

    fun getScannerType(): Int {
        return sharedPreferences?.getInt(KEY_SCANNER_TYPE, 0) ?: 0
    }

    // Model tier methods
    fun saveOcrModelTier(tierId: Int) {
        sharedPreferences?.edit()?.putInt(KEY_OCR_MODEL_TIER, tierId)?.apply()
    }

    fun getOcrModelTier(): Int {
        return sharedPreferences?.getInt(KEY_OCR_MODEL_TIER, 0) ?: 0 // Default to small (0)
    }

    // Recognition region methods
    fun saveRecognitionRegionMode(modeId: Int) {
        sharedPreferences?.edit()?.putInt(KEY_RECOGNITION_REGION_MODE, modeId)?.apply()
    }

    fun getRecognitionRegionMode(): Int {
        return sharedPreferences?.getInt(KEY_RECOGNITION_REGION_MODE, 0) ?: 0 // Default to full screen (0)
    }

    // 正则表达式存储
    fun saveRegexPattern(pattern: String) {
        sharedPreferences?.edit()?.putString(KEY_REGEX_PATTERN, pattern)?.apply()
    }

    fun getRegexPattern(): String {
        return sharedPreferences?.getString(KEY_REGEX_PATTERN, "") ?: ""
    }

    // 扫描头图像颠倒
    fun saveScannerImageUpsideDown(enabled: Boolean) {
        sharedPreferences?.edit()?.putBoolean(KEY_SCANNER_IMAGE_UPSIDE_DOWN, enabled)?.apply()
    }

    fun getScannerImageUpsideDown(): Boolean {
        return sharedPreferences?.getBoolean(KEY_SCANNER_IMAGE_UPSIDE_DOWN, false) ?: false
    }
}