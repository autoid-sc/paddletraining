package com.seuic.floating.service

import android.content.Context
import android.util.Log
import androidx.startup.Initializer
import com.seuic.floating.config.OcrConfig
import com.seuic.floating.utils.ext.mScanner
import com.seuic.floating.weight.scanner.MyScannerCamera
import com.seuic.scankey.IKeyEventCallback
import com.seuic.scankey.ScanKeyService
import com.seuic.util.common.BuildConfig
import com.seuic.util.common.CrashUtils
import com.seuic.util.common.LogUtils
import com.seuic.util.common.ThreadUtils
import com.seuic.util.common.ext.logITag
import org.opencv.android.OpenCVLoader
import java.io.File
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

/**
 * @Desc
 * @author KP
 * @date 2021/3/1 16:19
 * @version
 */

class BaseInitializer : Initializer<Unit> {

    companion object {
        private var deviceTime = arrayOf(-1, System.currentTimeMillis())
        fun updateDeviceTime(serverTime: Long) {
            deviceTime[0] = serverTime
            deviceTime[1] = System.currentTimeMillis()
        }

        fun getDeviceTime(): Long =
            maxOf(deviceTime[0] - deviceTime[1], 0) + System.currentTimeMillis()


        //是否可以解析ocr
        var canocr: Boolean = false

        //超时
        val startTime = AtomicLong(0)

        var lastClickTime = 0L
        // 当前使用的扫描模式
//		var currentScanMode: Configs.RecognitionMode? = null

        private val combinedListener by lazy {
            object : IKeyEventCallback.Stub() {
                private var isDownEvent = AtomicBoolean(false)

                // 记录上一次处理按键事件的时间
                private var lastProcessedTime: Long = 0L

                // 当前是否正在处理一次按键事件
                private var isProcessing = false
                override fun onKeyDown(p0: Int) {
                    val currentTime = System.currentTimeMillis()
                    Log.w("xxxxuhr", "识别onKeyDown -- " + currentTime)
                    Log.w("BaseInitializer", "onKeyDown: $p0")
                    // 在按键事件或其他需要检查授权的操作中

                    // 防止快速点击
                    if (lastClickTime != 0L && (currentTime - lastClickTime < 200)) {
                        isDownEvent.set(true)
                        lastClickTime = currentTime
                        Log.w("BaseInitializer", "onKeyDown: 太快了")
                        return
                    }
                    // 判断是否在防抖时间范围内（100ms）处理的是忽略下次up事件，上面的防抖是处理多次down
                    // 开始处理新的按键事件
                    isProcessing = true
                    // 每次按键触发时重置 OCR 处理状态

                    lastClickTime = currentTime
                    Log.w("xxxxuhr", "【KeyDown】currentTime: $currentTime")
                    Log.w("xxxxuhr", "【KeyDown】isDownEvent: ${isDownEvent.get()}")
                    if (isDownEvent.get()) return
                    // 判断当前是哪种扫描模式，并执行对应的逻辑
                    when (OcrConfig.selectedScanner.value?.typeId) {
                        0 -> {
                            handleCameraKeyDown(p0)

                        }

                        1 -> {
                            handleScannerKeyDown(p0)
                        }
                    }

                    isDownEvent.set(true)
                }

                override fun onKeyUp(p0: Int) {
                    val currentTime = System.currentTimeMillis()
                    Log.w("xxxxuhr", "识别onKeyUp -- " + currentTime)
                    if (!isDownEvent.get()) return
                    // 在按键事件或其他需要检查授权的操作中
                    val formatTime = formatTime(currentTime)
                    // 如果是正在处理的按键事件，则记录时间并结束
                    if (isProcessing) {
                        val logMessage = "【KeyUp】按键码: $p0, 时间: $formatTime"
                        Log.w("xxxxuhr", logMessage)

                        // 更新最后处理时间，并重置状态
                        lastProcessedTime = currentTime
                        isProcessing = false
                    } else {
                        Log.w(
                            "xxxxuhr",
                            "【KeyUp】忽略未匹配的释放事件，按键码: $p0, 时间: $formatTime"
                        )
                    }
                    // 统一释放资源
                    ThreadUtils.runOnUiThread {
                        FloatingOcrService.stopOcrDialog()
                        FloatingOcrService.releaseOcrDialog()
                    }
                    isDownEvent.set(false)


                }

                // 处理摄像头模式下的按键
                private fun handleCameraKeyDown(p0: Int) {
                    // 原有逻辑
				    if (OcrConfig.keyAreaMode.value == 1) {
				    	canocr = p0 == 248
				    } else if (OcrConfig.keyAreaMode.value == 2) {
				    	canocr = p0 == 249
				    } else if (OcrConfig.keyAreaMode.value == 0) {
				    	canocr = true
				    } else {
				    	canocr = false
				    }
				    Log.i("xxx", "canocr: $canocr")
                    handleDefaultKeyDown(p0)
                }

                // 处理扫描头模式下的按键
                private fun handleScannerKeyDown(p0: Int) {
                    // 原有逻辑
                    if (OcrConfig.keyAreaMode.value == 1) {
                        canocr = p0 == 248
                    } else if (OcrConfig.keyAreaMode.value == 2) {
                        canocr = p0 == 249
                    } else if (OcrConfig.keyAreaMode.value == 0) {
                        canocr = true
                    } else {
                        canocr = false
                    }
                    Log.i("xxx", "canocr: $canocr")

                    handleDefaultKeyDown(p0)
                }

                // 默认通用按键逻辑
                private fun handleDefaultKeyDown(p0: Int) {
                    Log.e("base", "canocr: $canocr")
                    if (canocr) {
                        // 每次按键触发时重置OCR服务的上报状态
                        FloatingOcrService.resetHasReportedResult()
                        FloatingOcrService.startOcrDialog()
                    }

                }

            }
        }

        private fun formatTime(currentTime: Long): String {
            val totalMillis = currentTime % 1000
            val seconds = (currentTime / 1000).toInt()
            val minutes = seconds / 60
            val hours = minutes / 60

            val secondsPart = seconds % 60
            val minutesPart = minutes % 60
            val hoursPart = hours % 24

            return String.format(
                Locale.getDefault(),
                "%02d:%02d:%02d.%03d",
                hoursPart,
                minutesPart,
                secondsPart,
                totalMillis
            )
        }

        //普通模式注册监听扫描键
        fun registerScanKey() {
            Log.e("APP内监听", "0");

            // 根据OcrConfig.keyAreaMode的值注册不同的按键监听
            val keyCodes = when (OcrConfig.keyAreaMode.value) {
                0 -> "248,249,250"  // 阻塞监听三个按键
                1 -> "248"          // 阻塞监听248按键
                2 -> "249"          // 阻塞监听249按键
                else -> "248,249,250"
            }
            Log.e("注册键", "OCR工作模式:普通模式：$keyCodes")
            ScanKeyService.getInstance().registerCallback(combinedListener, keyCodes)
        }


        //反注册监听扫描键
        fun unRegisterScanKey() {
            logITag("注册键", "注销ocr里面的")
            ScanKeyService.getInstance().unregisterCallback(combinedListener)
        }


        private val mark = AtomicBoolean(false)


        fun disconnectCamera() {
            //	isOpen.set(false)
            mScanner.runCatching {
                setParams(MyScannerCamera.SCAN_AIMER_ON, 0x00)
                setParams(MyScannerCamera.LIGHTS, 0x00)
                stopVideo()
            }
            mark.set(false)
            //	coroutineDispatcher.cancelChildren()
        }

    }


    override fun create(context: Context) {
        //	CommonConfig.getInstance().isDEBUG = BuildConfig.DEBUG
        logITag("BaseInitializer", "开始")
        CrashUtils.init(File(context.getExternalFilesDir("")!!.absolutePath, "crash"))
        //fragment自适应
//        AutoSizeConfig.getInstance().isCustomFragment = true
        if (!OpenCVLoader.initDebug()) {
            Log.e("OpenCV", "Initialization Failed")
        } else {
            Log.d("OpenCV", "Initialization Success")
        }
        LogUtils.getConfig().apply {
            isLogSwitch = BuildConfig.DEBUG
//			isLog2FileSwitch = true
        }



//        if (Configs.bootStart) {
//            AppInitializer.getInstance(context)
//                .initializeComponent(LevitatedBallInitializer::class.java)
//        }


//        val filter = IntentFilter()
//        filter.addAction(Intent.ACTION_SCREEN_OFF)
//        filter.addAction(Intent.ACTION_SCREEN_ON)
//        context.registerReceiver(ScreenOnOffReceiver(), filter)

        logITag("BaseInitializer", "结束")
    }

    override fun dependencies(): List<Class<out Initializer<*>>> = arrayListOf(
//        BroadcastInitializer::class.java,
    )
}