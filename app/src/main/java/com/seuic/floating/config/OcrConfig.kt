package com.seuic.floating.config

import androidx.lifecycle.MutableLiveData

object OcrConfig {


    // PaddleOCR 模型级别选择 (中文模型, 英文模型)
    enum class OcrModelTier(val tierId: Int, val displayName: String, val modelName: String) {
        CHINESE(0, "V2版本模型", "chinese"),
        ENGLISH(1, "英文模型", "english"),
        PLATE(2, "车牌场景模型", "plate");

        companion object {
            fun fromTierId(tierId: Int): OcrModelTier {
                return values().find { it.tierId == tierId } ?: CHINESE
            }
        }
    }

    var ocrModelTier = object : MutableLiveData<OcrModelTier>() {
        override fun setValue(value: OcrModelTier?) {
            super.setValue(value)
            value?.let {
                ConfigManager.saveOcrModelTier(it.tierId)
            }
        }

        override fun getValue(): OcrModelTier? {
            val value = super.getValue()
            if (value == null) {
                val savedTier = ConfigManager.getOcrModelTier()
                val tier = OcrModelTier.fromTierId(savedTier)
                super.setValue(tier)
            }
            return super.getValue()
        }
    }.apply {
        value = OcrModelTier.CHINESE
    }

    // 0为不设置，1为左ocr右扫码，2为左扫码右ocr
    var keyAreaMode = object : MutableLiveData<Int>() {
        override fun setValue(value: Int?) {
            super.setValue(value)
            value?.let { ConfigManager.saveKeyAreaMode(it) }
        }
    }.apply { value = 0 }

    // 悬浮按钮启用状态
    var floatingButtonEnabled = object : MutableLiveData<Boolean>() {
        override fun setValue(value: Boolean?) {
            super.setValue(value)
            value?.let { ConfigManager.saveFloatingButtonEnabled(it) }
        }
    }.apply { value = false }

    // 识别区域模式 (0: 默认全图识别, 1: 自定义区域识别)
    enum class RecognitionRegionMode(val modeId: Int, val displayName: String) {
        FULL_IMAGE(0, "默认全图识别"),
        CUSTOM_REGION(1, "自定义区域识别");

        companion object {
            fun fromModeId(modeId: Int): RecognitionRegionMode {
                return values().find { it.modeId == modeId } ?: FULL_IMAGE
            }
        }
    }

    var recognitionRegionMode = object : MutableLiveData<RecognitionRegionMode>() {
        override fun setValue(value: RecognitionRegionMode?) {
            super.setValue(value)
            value?.let {
                ConfigManager.saveRecognitionRegionMode(it.modeId)
            }
        }

        override fun getValue(): RecognitionRegionMode? {
            val value = super.getValue()
            if (value == null) {
                val savedMode = ConfigManager.getRecognitionRegionMode()
                val mode = RecognitionRegionMode.fromModeId(savedMode)
                super.setValue(mode)
            }
            return super.getValue()
        }
    }.apply {
        value = RecognitionRegionMode.FULL_IMAGE
    }

    // 扫描器选择（支持多扫描器设备）
    enum class ScannerType(val typeId: Int, val displayName: String, val deviceId: Int) {
        CAMERA(0, "相机", 0),
        SCANNER_HEAD_1(1, "扫描头1", 1),
        SCANNER_HEAD_2(2, "扫描头2", 2);

        companion object {
            fun fromTypeId(typeId: Int): ScannerType {
                return values().find { it.typeId == typeId } ?: CAMERA
            }
        }
    }

    // 扫描器选择
    val selectedScanner = object : MutableLiveData<ScannerType>() {
        override fun setValue(value: ScannerType?) {
            super.setValue(value)
            value?.let {
                ConfigManager.saveScannerType(it.typeId)
            }
        }

        override fun getValue(): ScannerType? {
            val value = super.getValue()
            if (value == null) {
                // 延迟初始化，确保 ConfigManager 已初始化
                val savedType = ConfigManager.getScannerType()
                val scannerType = ScannerType.fromTypeId(savedType)
                super.setValue(scannerType)
            }
            return super.getValue()
        }
    }.apply {
        // 初始化时设置为默认值，后续会通过 getValue() 延迟加载
        value = ScannerType.CAMERA
    }

    // 扫描头图像颠倒（开启后对扫描头图像做180°旋转，适配物理倒装扫描头）
    var scannerImageUpsideDown = object : MutableLiveData<Boolean>() {
        override fun setValue(value: Boolean?) {
            super.setValue(value)
            value?.let { ConfigManager.saveScannerImageUpsideDown(it) }
        }
    }.apply { value = false }

    // 正则表达式校验规则
    val regexPattern = object : MutableLiveData<String?>() {
        override fun setValue(value: String?) {
            super.setValue(value)
            value?.let {
                ConfigManager.saveRegexPattern(it)
            }
        }

        override fun getValue(): String? {
            val value = super.getValue()
            if (value == null) {
                val savedPattern = ConfigManager.getRegexPattern()
                super.setValue(savedPattern)
            }
            return super.getValue()
        }
    }.apply {
        // 延迟加载
        value = null
    }

    // 用于在主界面展示的 LiveData
    val lastRecognizedBitmap = MutableLiveData<android.graphics.Bitmap?>()
    val lastRecognitionTime = MutableLiveData<Long>()
    val lastOcrResult = MutableLiveData<String?>()
}

data class OcrPromptOption(
    val id: Int,
    val name: String,
    val prompt: String
)
