package com.seuic.floating.network

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.seuic.floating.config.OcrConfig
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlin.coroutines.resume

/**
 * OCR 服务工厂
 * 创建并管理 PaddleOCR 服务
 */
class OCRServiceFactory(private val context: Context) {

    companion object {
        private const val TAG = "OCRServiceFactory"

        @Volatile
        private var instance: OCRServiceFactory? = null

        fun getInstance(context: Context): OCRServiceFactory {
            return instance ?: synchronized(this) {
                instance ?: OCRServiceFactory(context.applicationContext).also { instance = it }
            }
        }
    }

    private val paddleService = PaddleOCRService(context)

    /**
     * 获取当前 OCR 服务
     */
    private fun getOCRService(): IOCRService {
        return paddleService
    }

    suspend fun initialize(): Boolean {
        val service = getOCRService()
        Log.d(TAG, "初始化 OCR 服务: PaddleOCR")
        return service.initialize()
    }

    /**
     * 执行 OCR 识别
     */
    suspend fun recognizeImage(
        bitmap: Bitmap,
        callback: (String?) -> Unit
    ) {
        val service = getOCRService()

        // 确保服务已初始化
        if (!service.isInitialized()) {
            Log.w(TAG, "OCR 服务未初始化，正在初始化...")
            val success = try {
                service.initialize()
            } catch (e: Exception) {
                Log.e(TAG, "OCR 服务初始化异常: ${e.message}", e)
                false
            }

            if (!success) {
                Log.e(TAG, "PaddleOCR 初始化失败")
                callback.invoke(null)
                return
            }
            Log.d(TAG, "OCR 服务初始化成功")
        }

        service.recognizeImage(bitmap, callback)
    }

    /**
     * 检查 PaddleOCR 是否可用
     */
    fun isPaddleOCRAvailable(): Boolean {
        return try {
            Class.forName("com.baidu.paddle.lite.PaddlePredictor")
            true
        } catch (e: ClassNotFoundException) {
            false
        }
    }

    /**
     * 释放资源
     */
    fun release() {
        paddleService.release()
    }
}

/**
 * OCR 服务接口
 */
interface IOCRService {
    suspend fun initialize(): Boolean
    suspend fun recognizeImage(bitmap: Bitmap, callback: (String?) -> Unit)
    fun release()
    fun isInitialized(): Boolean
}