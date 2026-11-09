package com.seuic.ocr.listener

import android.util.Log
import com.seuic.floating.config.OcrConfig
import com.seuic.floating.service.FloatingOcrService
import com.seuic.scankey.IKeyEventCallback
import com.seuic.util.common.ThreadUtils
import com.seuic.util.common.ToastUtils
import java.util.concurrent.atomic.AtomicBoolean

class ScanKeyListener : IKeyEventCallback.Stub() {

    private val TAG = "ScanKeyListener"
    private val isDownEvent = AtomicBoolean(false)
    private var lastClickTime = 0L
    private var lastProcessedTime = 0L
    private var isProcessing = false

    override fun onKeyDown(keyCode: Int) {
        val currentTime = System.currentTimeMillis()
        Log.w(TAG, "onKeyDown: $keyCode")

        if (lastClickTime != 0L && (currentTime - lastClickTime < 200)) {
            isDownEvent.set(true)
            lastClickTime = currentTime
            ThreadUtils.runOnUiThread {
                ToastUtils.showShort("点击太快")
            }
            return
        }

        val debounceTime = 100L
        if (currentTime - lastProcessedTime < debounceTime) {
            isDownEvent.set(true)
            lastClickTime = currentTime
            ThreadUtils.runOnUiThread {
                ToastUtils.showShort("点击太快")
            }
            Log.w(TAG, "忽略快速连续触发，按键码: $keyCode")
            return
        }

        isProcessing = true
        lastClickTime = currentTime

        if (isDownEvent.get()) return

        when (OcrConfig.selectedScanner.value?.typeId ) {
            0 -> {
                handleCameraKeyDown(keyCode)
            }
            1 -> {
                handleScannerKeyDown(keyCode)
            }
        }

        isDownEvent.set(true)
    }

    override fun onKeyUp(keyCode: Int) {
        val currentTime = System.currentTimeMillis()
        if (!isDownEvent.get()) return


        if (isProcessing) {
            lastProcessedTime = currentTime
            isProcessing = false
        } else {
            Log.w(TAG, "忽略未匹配的释放事件，按键码: $keyCode")
        }
        Log.d("ocr模式5", "onKeyUp")

        isDownEvent.set(false)
    }

    private fun handleCameraKeyDown(keyCode: Int) {
        FloatingOcrService.startOcrDialog()
    }

    private fun handleScannerKeyDown(keyCode: Int) {
        FloatingOcrService.startOcrDialog()
    }

}
