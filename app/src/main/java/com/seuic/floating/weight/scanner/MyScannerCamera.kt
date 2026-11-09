package com.seuic.floating.weight.scanner

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.PorterDuff
import android.graphics.Rect
import android.util.AttributeSet
import android.util.Log
import android.view.SurfaceView
import com.luoye.bzyuvlib.BZYUVUtil
import com.seuic.floating.config.OcrConfig
import com.seuic.floating.utils.Nv21Utils
import com.seuic.floating.utils.ext.mScanner
import com.seuic.scanner.VideoCallBack
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.ExecutorCoroutineDispatcher
import kotlinx.coroutines.ObsoleteCoroutinesApi
import kotlinx.coroutines.cancelChildren
import kotlinx.coroutines.launch
import kotlinx.coroutines.newSingleThreadContext
import java.util.Arrays
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs
import kotlin.math.roundToInt

/**
 *
 * @ProjectName:    seuic-ocr
 * @Description:     类作用描述
 * @Author:         KP
 * @CreateDate:     2021/8/9 14:48
 * @Version:        1.0
 */
class MyScannerCamera(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0,
) : SurfaceView(context, attrs, defStyleAttr), ISurfaceView, ICamera, VideoCallBack {

    constructor(
        context: Context,
        attrs: AttributeSet? = null,
    ) : this(context, attrs, 0)

    companion object {
        const val cropWidth = 960
        const val cropHeight = 640

        const val SCAN_AIMER_ON = 0x19
        const val LIGHTS = 0x20
        const val EXP_LEVEL = 0x0d
        const val LIGHT_LEVEL = 0x09

        /**
         * 初始化的时候设置aimer和light状态,需要在startVideo之前调用,默认为3
         * 0 表示无aimer无light
         * 1 表示只有aimer
         * 2 表示只有light
         * 3 表示既有aimer也有light
         */
        const val INIT_AIMER_AND_LIGHT = 0x07


        const val SENSOR_EXP_LEVEL = 0x0d
        const val SENSOR_GAIN_MIN = 0x1a
        const val SENSOR_GAIN_MAX = 0x1b
        const val SENSOR_EXP_MIN = 0x1c
        const val SENSOR_EXP_MAX = 0x1d

        const val NO_AIMER_AND_LED = 0
        const val ONLY_AIMER = 1
        const val ONLY_LED = 2
        const val BOTH_AIMER_AND_LED = 3
    }

    private val nV21ToBitmap: FastYUVtoRGB by lazy { FastYUVtoRGB(context.applicationContext) }
    private var imageDataCallback: ImageDataCallback? = null

    @ObsoleteCoroutinesApi
    private val coroutineDispatcher: ExecutorCoroutineDispatcher =
        newSingleThreadContext("MyScanner")
    private var mZoom = 1f
    private val isOpen = AtomicBoolean(false)
    private val isWork = AtomicBoolean(false)


    override fun setImageDataCallback(imageDataCallback: ImageDataCallback?) {
        this.imageDataCallback = imageDataCallback
    }

    override fun isSupportZoom(): Boolean = true

    override fun setZoomValue(value: Float): Boolean {
        mZoom = value
        return true
    }

    fun setLightToggle(enabled: Boolean): Boolean {
        return runCatching {
            mScanner.setParams(LIGHTS, if (enabled) 0x01 else 0x00)
        }.getOrDefault(false)
    }


    override fun connectCamera(isFlash: Boolean): Boolean {

        if (mScanner.open()) {
            //初始值设置
            mScanner.setParams(SENSOR_EXP_LEVEL, 0)
            mScanner.setParams(SENSOR_GAIN_MIN, 1 * 64)
            mScanner.setParams(SENSOR_GAIN_MAX, 8 * 64)

            mScanner.setParams(SENSOR_EXP_MIN, 0x01 * 20)
            mScanner.setParams(SENSOR_EXP_MAX, 0x15 * 20)

            mScanner.setParams(
                INIT_AIMER_AND_LIGHT, if (isFlash) 3 else 1
            )


            mScanner.setVideoCallBack(this)
            mScanner.startVideo(500)
            isOpen.set(true)
            return true
        }
        return false
    }

    override fun disconnectCamera() {
        isOpen.set(false)
        mScanner.runCatching {
            setParams(SCAN_AIMER_ON, 0x00)
            setParams(LIGHTS, 0x00)
            stopVideo()
            clearDraw()

//			close()
        }
        mark.set(false)
        coroutineDispatcher.cancelChildren()
    }

    // 在类中添加一个属性来标记是否已经设置了原始图像尺寸
    // 在类中添加属性来保存上次的图像尺寸
    private var lastWidth = -1
    private var lastHeight = -1
    private val mark = AtomicBoolean(false)

    //	private var index:Int=0
    override fun onVideoCallBack(width: Int, height: Int, image: ByteArray): Boolean {
        // 判断图片大小发生了变化时再去设置
        if (lastWidth != width || lastHeight != height) {
            Log.e("onVideoCallBack", "width: $width,height: $height")
            lastWidth = width
            lastHeight = height
        }
        if (isOpen.get()) {
            val realImage = image.copyOfRange(image.size - (width * height), image.size)
            val nv21buffer = getBuffer(width, height)
            realImage.copyInto(nv21buffer, 0, 0, realImage.size)
            // 扫描头图像颠倒：开启后对 NV21 做 180° 旋转，适配物理倒装扫描头
            if (OcrConfig.scannerImageUpsideDown.value == true) {
                val rotated = ByteArray(nv21buffer.size)
                Nv21Utils.NV21_rotate_to_180(nv21buffer.clone(), rotated, width, height)
                System.arraycopy(rotated, 0, nv21buffer, 0, rotated.size)
            }
            drawScanner(width, height, nv21buffer)
        }
        return true
    }

    private fun drawScanner(width: Int, height: Int, nv21: ByteArray) {
        val canvas = holder.lockCanvas() ?: return

        //绘制四周阴影区域
        var offsetX = 0
        var offsetY = 0

        if (offsetX == 0 && offsetY == 0) {
            offsetX = abs(width - cropWidth) / -2
            offsetY = abs(height - cropHeight) / -2
        } else {
            if (abs(offsetX) > abs(width - cropWidth)) {
                offsetX = abs(width - cropWidth) * -1
            }
            if (abs(offsetY) > abs(height - cropHeight)) {
                offsetY = abs(height - cropHeight) * -1
            }
        }
        val realWidth = (cropWidth / mZoom).roundToInt() / 4 * 4
        val realHeight = (cropHeight / mZoom).roundToInt() / 4 * 4
        val realOffsetX =
            (abs(offsetX) + (cropWidth - realWidth) / 2) / 4 * 4
        val realOffsetY =
            (abs(offsetY) + (cropHeight - realHeight) / 2) / 4 * 4
        val cropNV21 = ByteArray(nv21.size)
        BZYUVUtil.cropNV21(
            nv21,
            cropNV21,
            width,
            height,
            realOffsetX,
            realOffsetY,
            realWidth,
            realHeight
        )
        imageDataCallback?.also {
            CoroutineScope(coroutineDispatcher).launch {
                if (!mark.get()) {
                    mark.set(true)
                    imageDataCallback?.onDataCallback(cropNV21, realWidth, realHeight)
                    mark.set(false)
                }
            }
        }

        canvas.drawBitmap(
            nV21ToBitmap.convertYUVtoRGB2(cropNV21, realWidth, realHeight),
            null,
            Rect(0, 0, canvas.width, canvas.height), null
        )
//		val mat = MatUtils.getMatByByteArray(nv21, width, height)
//		val bitmap = Bitmap.createBitmap(mat.width(), mat.height(), Bitmap.Config.ARGB_8888)
//		Utils.matToBitmap(mat, bitmap)
//		canvas.drawBitmap(
//			bitmap,
//			null,
//			Rect(0, 0, canvas.width, canvas.height), null
//		)
        holder.unlockCanvasAndPost(canvas)
    }

    private fun getBuffer(width: Int, height: Int): ByteArray {
        val frame = ByteArray(width * height * 3 / 2)
        Arrays.fill(frame, 0x80.toByte())
        return frame
    }

    fun clearDraw() {
        var canvas: Canvas? = null
        try {
            canvas = holder.lockCanvas(null)
//			canvas.drawColor(Color.WHITE)
            canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.SRC)
        } catch (e: Exception) {
        } finally {
            if (canvas != null) {
                holder.unlockCanvasAndPost(canvas)
            }
        }
    }
}