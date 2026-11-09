package com.seuic.floating.service

import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.PixelFormat
import android.os.Build
import android.provider.Settings
import android.util.Log
import android.view.GestureDetector
import android.view.Gravity
import android.view.LayoutInflater
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager
import androidx.camera.view.PreviewView
import androidx.constraintlayout.widget.ConstraintLayout
import androidx.lifecycle.LifecycleService
import androidx.lifecycle.MutableLiveData
import com.seuic.core.utils.ext.appContext
import com.seuic.core.utils.ext.windowManager
import com.seuic.floating.R
import com.seuic.floating.config.OcrConfig
import com.seuic.floating.config.OcrImageBean
import com.seuic.floating.config.OcrImageListener
import com.seuic.floating.utils.CameraUtils
import com.seuic.floating.utils.ForegroundNotificationUtils
import com.seuic.floating.utils.KeyInfoExtractor
import com.seuic.floating.utils.ext.mScanner
import com.seuic.floating.network.OCRServiceFactory
import com.seuic.floating.weight.scanner.FastYUVtoRGB
import com.seuic.floating.weight.scanner.ImageDataCallback
import com.seuic.floating.weight.scanner.MyScannerCamera
import com.seuic.scanner.VideoCallBack
import com.seuic.scanservice.DecodeInfoParcel
import com.seuic.util.common.BarUtils
import com.seuic.util.common.ThreadUtils
import com.seuic.util.common.ToastUtils
import com.seuic.util.common.ext.logITag
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.concurrent.atomic.AtomicBoolean


class FloatingOcrService : LifecycleService(),
    androidx.lifecycle.Observer<FloatingOcrService.Companion.FloatWindowState> {

    companion object {
        const val OCR_WAIT_TIMEOUT = 0x3A5 // OCR和条码上报的等待超时，单位ms
        const val OCR_REPORT_MODE = 0x3A4
        const val OCR_BARCODE_SEPARATOR = 0x3A6
        private var tempImageData: ByteArray? = null
        private var tempImageWidth: Int = 0
        private var tempImageHeight: Int = 0
        const val ACTION_SCAN_GET_IMAGE_RESP = "com.scan.getImageResp"
        private val nV21ToBitmap: FastYUVtoRGB by lazy { FastYUVtoRGB(appContext) }
        //增加更严格的防重复上报机制
        private val hasReportedResult = AtomicBoolean(false)
        private val ocrServiceFactory = OCRServiceFactory(appContext)
        private val keyInfoExtractor = KeyInfoExtractor()
        @Volatile
        private var isCameraOpening = AtomicBoolean(false)
        @Volatile
        private var isCameraClosing = AtomicBoolean(false)
        //定义浮动窗口布局
        var mFloatLayout: ConstraintLayout? = null
        val wmParams: WindowManager.LayoutParams by lazy {
            val wmParams = WindowManager.LayoutParams()

            // 使用正确的窗口类型
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                wmParams.type = WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
            } else {
                wmParams.type = WindowManager.LayoutParams.TYPE_PHONE
            }

            wmParams.format = PixelFormat.RGBA_8888
            // 添加适当的flags以避免影响其他窗口的输入事件
            wmParams.flags = WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                    WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL or
                    WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN

            wmParams.gravity = Gravity.CENTER_HORIZONTAL or Gravity.TOP
            wmParams.x = 0
            wmParams.y = 0

            val displayMetrics = appContext.resources.displayMetrics
            val screenWidth = displayMetrics.widthPixels

            // 设置悬浮窗口宽度为屏幕宽度
            wmParams.width = screenWidth

            // 动态计算高度以适应不同模式的最大需求
            // 摄像头模式: 768*1024 (宽*高比例约0.75)
            // 扫描头模式: 960*640 (宽*高比例约1.5)
            // 选择最大比例所需的高度，确保两种模式都能完整显示
            val maxHeightRatio = maxOf(1024.0f/768.0f, 640.0f/960.0f) // 取较大的高度比例
            wmParams.height = (screenWidth * maxHeightRatio).toInt()

            wmParams
        }
        private lateinit var touchGestureDetector: GestureDetector
        private val mSimpleOnGestureListener = object : GestureDetector.SimpleOnGestureListener() {
            override fun onScroll(
                e1: MotionEvent?,
                e2: MotionEvent,
                distanceX: Float,
                distanceY: Float,
            ): Boolean {
                /// 确保mFloatLayout不为空且已经测量过
                mFloatLayout?.let { layout ->
                    if (layout.measuredHeight > 0) {
                        wmParams.x = 0
                        wmParams.y = e2.rawY.toInt() - layout.measuredHeight / 2 - BarUtils.getStatusBarHeight()

                        try {
                            windowManager.updateViewLayout(layout, wmParams)
                        } catch (e: Exception) {
                            Log.w("floating", "更新悬浮窗位置失败", e)
                        }
                    }
                }
                return true
            }
        }

        // 初始化 GestureDetector
        private fun initializeGestureDetector(context: Context) {
            touchGestureDetector = GestureDetector(context, mSimpleOnGestureListener)
        }


        val state = MutableLiveData(FloatWindowState.NONE)
        val isShowing = AtomicBoolean(false)

        enum class FloatWindowState {
            NONE,
            SHOW_WINDOW,
            DISMISS_WINDOW,
            TAKE_PHOTO
        }

        /**
         * @description 开始浮动弹窗OCR识别
         * @param
         * @return
         */
        fun startOcrDialog() {
            if (!isShowing.get() && state.value != FloatWindowState.SHOW_WINDOW) {
                state.postValue(FloatWindowState.SHOW_WINDOW)  // 设置为显示OCR窗口状态
            }

        }

        fun stopOcrDialog() {
            // 处理缓存的最后一帧图像
            processCachedImage()
            if (state.value != FloatWindowState.DISMISS_WINDOW)
                state.postValue(FloatWindowState.DISMISS_WINDOW)
        }

        fun releaseOcrDialog() {
            dismissFloatView()

        }

        // 添加重置方法
        fun resetHasReportedResult() {
            hasReportedResult.set(false)
            Log.d("OcrService", "OCR状态已重置，允许下一次识别")
        }

        // 添加获取状态的方法（如果需要在外部访问）
        fun isResultReported(): Boolean = hasReportedResult.get()

        fun getImage() {
            if (state.value == FloatWindowState.TAKE_PHOTO)
                state.postValue(FloatWindowState.NONE)

            state.postValue(FloatWindowState.TAKE_PHOTO)
        }

        private fun processCachedImage() {
            if (tempImageData != null) {
                Log.d("OcrService", "处理缓存的图像"+tempImageWidth+"  "+tempImageHeight)
            }else {
                Log.d("OcrService", "没有缓存的图像"+tempImageWidth+"  "+tempImageHeight)
            }
            tempImageData?.let { imageData ->
                // Capture view dimensions on the Main thread!
                var viewFinderWidth = 0
                var viewFinderHeight = 0
                var scanViewfinderLeft = 0
                var scanViewfinderTop = 0
                var scanViewfinderWidth = 0
                var scanViewfinderHeight = 0

                val viewFinder = mFloatLayout?.findViewById<View>(R.id.view_finder)
                val scanViewfinder = mFloatLayout?.findViewById<View>(R.id.scan_viewfinder)
                if (viewFinder != null && scanViewfinder != null && scanViewfinder.visibility == View.VISIBLE) {
                    viewFinderWidth = viewFinder.width
                    viewFinderHeight = viewFinder.height
                    scanViewfinderLeft = scanViewfinder.left - viewFinder.left
                    scanViewfinderTop = scanViewfinder.top - viewFinder.top
                    scanViewfinderWidth = scanViewfinder.width
                    scanViewfinderHeight = scanViewfinder.height
                    Log.d("OcrService", "视图坐标采集: viewFinder(${viewFinderWidth}x${viewFinderHeight}), scanViewfinder(left=$scanViewfinderLeft, top=$scanViewfinderTop, width=$scanViewfinderWidth, height=$scanViewfinderHeight)")
                }

                try {
                    // 创建OcrImageBean
                    Log.e("Floating", "创建OcrImageBean"+tempImageWidth+"  "+tempImageHeight)
                    val imageBean = OcrImageBean(imageData, tempImageWidth, tempImageHeight)

                    // 保存图像文件
                    GlobalScope.launch(Dispatchers.IO) {
                        Log.e("Floating", "开始处理图像数据 - size: ${imageData.size}, width: ${imageBean.width}, height: ${imageBean.height}")
                        var bitmap: Bitmap? = null
                        try {
                            if (OcrConfig.selectedScanner.value?.typeId == 0){
                                Log.d("Floating", "使用 convertYUVtoRGB 转换图像")
                                bitmap = nV21ToBitmap.convertYUVtoRGB(imageData,
                                    imageBean.width,
                                    imageBean.height)
                            }else {
                                Log.d("Floating", "使用 convertYUVtoRGB2 转换图像")
                                bitmap = nV21ToBitmap.convertYUVtoRGB2(imageData,
                                    imageBean.width,
                                    imageBean.height)
                            }
                        } catch (e: Exception) {
                            Log.e("Floating", "YUV转RGB转换失败", e)
                        }


                        val localBitmap = bitmap
                        Log.e("Floating", "转换结果 - bitmap: ${localBitmap != null}, width: ${localBitmap?.width ?: "null"}, height: ${localBitmap?.height ?: "null"}")
                        if (localBitmap != null && localBitmap.width > 0 && localBitmap.height > 0) {
                            var finalBitmap: Bitmap = localBitmap
                            if ((OcrConfig.selectedScanner.value?.typeId == 0 || OcrConfig.selectedScanner.value?.typeId == 1) && viewFinderWidth > 0 && viewFinderHeight > 0) {
                                try {
                                    val scaleX = localBitmap.width.toFloat() / viewFinderWidth
                                    val scaleY = localBitmap.height.toFloat() / viewFinderHeight

                                    val cropLeft = (scanViewfinderLeft * scaleX).toInt().coerceIn(0, localBitmap.width)
                                    val cropTop = (scanViewfinderTop * scaleY).toInt().coerceIn(0, localBitmap.height)
                                    val cropWidth = (scanViewfinderWidth * scaleX).toInt().coerceAtMost(localBitmap.width - cropLeft)
                                    val cropHeight = (scanViewfinderHeight * scaleY).toInt().coerceAtMost(localBitmap.height - cropTop)

                                    if (cropWidth > 0 && cropHeight > 0) {
                                        finalBitmap = Bitmap.createBitmap(localBitmap, cropLeft, cropTop, cropWidth, cropHeight)
                                        Log.d("OcrService", "模式裁剪成功: ${localBitmap.width}x${localBitmap.height} -> ${finalBitmap.width}x${finalBitmap.height}")
                                    }
                                } catch (e: Exception) {
                                    Log.e("Floating", "裁剪图像失败", e)
                                }
                            }

                            Log.e("Floating", "111111111111111111111111111111${imageData.size}")
//                            val filePath = saveLastImage(finalBitmap)

                            Log.e("Floating", "保存图像成功，路径为：----------${finalBitmap.height}")
                            // 调用OCR识别
                            processImageForOCR(finalBitmap)
                            Log.d("OcrService", "开始识别图像")
                        } else {
                            Log.e("Floating", "Bitmap 无效或尺寸为0，跳过OCR识别")
                        }
                    }
                } catch (e: Exception) {
                    Log.e("Floating", "处理缓存图像失败", e)
                } finally {
                    // 清空缓存
                    tempImageData = null
                    tempImageWidth = 0
                    tempImageHeight = 0
                }
            }
        }
        private fun processImageForOCR(bitmap: Bitmap) {
            GlobalScope.launch(Dispatchers.Main) {
                try {
                    val engineName = "PaddleOCR"
                    Log.e("Floating", "开始识别图像，使用 $engineName")

                    // 使用 OCR 服务工厂
                    ocrServiceFactory.recognizeImage(bitmap) { ocrText ->
                        if (ocrText != null) {
                            Log.e("Floating", "$engineName 识别成功，文本长度: ${ocrText.length}")
                            handleOCRResult(ocrText)
                        } else {
                            Log.e("Floating", "$engineName 识别失败")
                        }
                    }
                } catch (e: Exception) {
                    Log.e("Floating", "图像处理失败", e)
                }
            }
        }

        private fun handleOCRResult(result: String) {
            try {
                // 直接返回 OCR 识别出的文本
                val extractedText = result

                if (extractedText.isNotEmpty()) {
                    Log.d("Floating", "OCR识别文本: $extractedText")

                    // 直接使用OCR识别的原始文本
                    val decodeInfoParcel = DecodeInfoParcel()
                    decodeInfoParcel.barcode = extractedText
                    decodeInfoParcel.length = extractedText.length
                    decodeInfoParcel.codetype = "OCR"

                    Log.d("Floating", "OCR识别结果: $extractedText")

                    ThreadUtils.runOnUiThreadDelayed({
                        // 先改成扫码模式
                        mScanner?.run {
                            setParams(OCR_REPORT_MODE, 0)
                            setParams(OCR_BARCODE_SEPARATOR, 0)
                            setParams(
                                OCR_WAIT_TIMEOUT,
                                500
                            ) // OCR和条码上报的等待超时，单位ms
                            logITag("onDecodeComplete重新更改scanner模式--")
                        }

                        Log.d(
                            "Floating",
                            "scanner回调参数---进来了-Mainacti1222222= ${decodeInfoParcel.barcode}"
                        )

                        // 提交到scanner.notify方法
                        mScanner.notifyDecodeInfo(decodeInfoParcel)

                        // 延时500ms切回原来模式成扫码模式
                        ThreadUtils.runOnUiThreadDelayed({
                            mScanner?.run {
                                setParams(OCR_REPORT_MODE, 5)
                                setParams(OCR_BARCODE_SEPARATOR, 0)
                                setParams(
                                    OCR_WAIT_TIMEOUT,
                                    500
                                ) // OCR和条码上报的等待超时，单位ms
                                logITag("onDecodeComplete原来的scanner模式--延时切回")
                            }
                        }, 500)
                    }, 300)
                }
            } catch (e: Exception) {
                Log.e("Floating", "解析OCR结果失败", e)
            }
        }


        fun dismissFloatView() {
            ThreadUtils.runOnUiThread {
                // 确保mFloatLayout不为空且已经测量过
                mFloatLayout?.let { layout ->
                    dismissFloatViewInternal()
                }
            }
        }

        private fun dismissFloatViewInternal() {
            try {
                // 根据 scanType 清理相应的资源
                when (OcrConfig.selectedScanner.value?.typeId) {
                    0 -> {
                        // CAMERA 模式，关闭相机
                        isCameraClosing.set(true)
                        try {
                            CameraUtils.closeCamera()
                        } finally {
                            isCameraClosing.set(false)
                        }
                    }
                    1 -> {
                        // SCANNER 模式，断开扫描头连接
                        mFloatLayout?.let { layout ->
                            try {
                                val scannerCamera = layout.findViewById<MyScannerCamera>(R.id.view_finder)
                                scannerCamera?.disconnectCamera()
                            } catch (e: Exception) {
                                Log.w("floatingWindow:", "断开扫描头连接失败", e)
                            }
                        }
                    }
                }

                // 移除视图
                mFloatLayout?.let { layout ->
                    try {
                        windowManager.removeViewImmediate(layout)
                    } catch (e: IllegalArgumentException) {
                        // 视图未附加到窗口管理器，忽略错误
                        Log.w("floatingWindow:", "视图未附加到窗口管理器，无需移除")
                    } catch (e: Exception) {
                        Log.w("floatingWindow:", "移除视图失败", e)
                    }
                }
            } catch (e: Exception) {
                Log.w("floatingWindow:", "移除悬浮窗失败", e)
            } finally {
                mFloatLayout = null
                isShowing.set(false)
                isCameraOpening.set(false)
                isCameraClosing.set(false)
                // 清理图像缓存
                tempImageData = null
                tempImageWidth = 0
                tempImageHeight = 0
            }
        }
    }

    private fun showFloatView() {
        // 检查权限
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M &&
            !Settings.canDrawOverlays(this)) {
            Log.e(TAG, "没有悬浮窗权限")
            return
        }
        if (isCameraOpening.get() || isCameraClosing.get()) {
            Log.w(TAG, "相机正在操作中，忽略重复请求")
            ToastUtils.showShort("请稍后再试")
            return
        }
        if (!isShowing.get() && mFloatLayout == null) {
            try {
                val inflater = LayoutInflater.from(applicationContext)

                // 根据 scanType 选择不同的布局和处理逻辑
                when (OcrConfig.selectedScanner.value?.typeId) {
                    0 -> {
                        // CAMERA 模式
                        mFloatLayout = inflater.inflate(R.layout.camera_window_layout, null) as ConstraintLayout
                        val viewFinder = mFloatLayout?.findViewById<PreviewView>(R.id.view_finder)

                        // 初始化 GestureDetector 确保在设置触摸监听之前已准备就绪
                        initializeGestureDetector(applicationContext)

                        // 设置触摸监听器
                        mFloatLayout?.setOnTouchListener { _, event ->
                            touchGestureDetector.onTouchEvent(event)
                            true
                        }

                        // 添加到窗口
                        windowManager.addView(mFloatLayout, wmParams)
                        isShowing.set(true)

                        // 打开相机
                        if (viewFinder != null) {
                            isCameraOpening.set(true)
                            CameraUtils.openCamera(null, viewFinder) { imageBean ->
                                isCameraOpening.set(false)
                                listener.invoke(imageBean)
                            }
                        }
                    }
                    1 -> {
                        // SCANNER 模式
                        mFloatLayout = inflater.inflate(R.layout.scanner_window_layout, null) as ConstraintLayout
                        val scannerCamera = mFloatLayout?.findViewById<MyScannerCamera>(R.id.view_finder)

                        // 初始化 GestureDetector 确保在设置触摸监听之前已准备就绪
                        initializeGestureDetector(applicationContext)

                        // 设置触摸监听器
                        mFloatLayout?.setOnTouchListener { _, event ->
                            touchGestureDetector.onTouchEvent(event)
                            true
                        }

                        // 添加到窗口
                        windowManager.addView(mFloatLayout, wmParams)
                        isShowing.set(true)

                        // 连接扫描头相机
                        scannerCamera?.setImageDataCallback(object : ImageDataCallback {
                            override fun onDataCallback(image: ByteArray, width: Int, height: Int) {

                                listener.invoke(OcrImageBean(image, width, height))
                            }
                        })
                        scannerCamera?.connectCamera()
                    }
                    else -> {
                        // 默认使用 CAMERA 模式（同case 0）
                        mFloatLayout = inflater.inflate(R.layout.camera_window_layout, null) as ConstraintLayout
                        val viewFinder = mFloatLayout?.findViewById<PreviewView>(R.id.view_finder)

                        // 初始化 GestureDetector 确保在设置触摸监听之前已准备就绪
                        initializeGestureDetector(applicationContext)

                        // 设置触摸监听器
                        mFloatLayout?.setOnTouchListener { _, event ->
                            touchGestureDetector.onTouchEvent(event)
                            true
                        }

                        // 添加到窗口
                        windowManager.addView(mFloatLayout, wmParams)
                        isShowing.set(true)

                        // 打开相机
                        if (viewFinder != null) {
                            isCameraOpening.set(true)
                            CameraUtils.openCamera(null, viewFinder) { imageBean ->
                                isCameraOpening.set(false)
                                listener.invoke(imageBean)
                            }
                        }
                    }
                }

                // 根据识别区域模式设置对齐扫描框的可见性
                val scanViewfinder = mFloatLayout?.findViewById<View>(R.id.scan_viewfinder)
                val isCustomRegion = OcrConfig.recognitionRegionMode.value == OcrConfig.RecognitionRegionMode.CUSTOM_REGION
                scanViewfinder?.visibility = if (isCustomRegion) View.VISIBLE else View.GONE

                Log.d(TAG, "悬浮窗添加成功，模式: ${if (OcrConfig.selectedScanner.value?.typeId == 0) "CAMERA" else "SCANNER"}")
            } catch (e: Exception) {
                Log.e(TAG, "添加悬浮窗失败", e)
                mFloatLayout = null
                isShowing.set(false)
                isCameraOpening.set(false)
                isCameraClosing.set(false)
            }
        }
    }


    private val listener: OcrImageListener = object : OcrImageListener {
        override fun invoke(p1: OcrImageBean) {
            Log.e("xuhrrrr", "invoke")
            // 处理OCR识别
            tempImageData = p1.image.copyOf()
            tempImageWidth = p1.width
            tempImageHeight = p1.height

        }

    }

    override fun onCreate() {
        super.onCreate()
        ForegroundNotificationUtils.startForegroundNotification(this)
        //监听注册
        //指定dsp库路径，是使用snpe的前提,最好在APP初始化时调用一次，这样后续就不必再调用
        state.observeForever(this::onChanged)

    }


    private val TAG = this.javaClass.name
    private lateinit var imageData: ByteArray
    private var getImageNow: Boolean? = false
    private var count = 0
    private var lastWidth = -1
    private var lastHeight = -1


    private var mVideoCallBack = VideoCallBack { w: Int, h: Int, bytes: ByteArray ->
        // 图像回调
        Log.d(TAG, "打开ocr扫描摄像头+++")
        if (lastWidth != w || lastHeight != h) {
            Log.e("onVideoCallBack", "width: $w,height: $h")
            lastWidth = w
            lastHeight = h
        }

        val srcPos: Int = bytes.size - w * h
        val destPos = 0
        val length = w * h
        imageData = ByteArray(w * h)
        System.arraycopy(bytes, srcPos, imageData, destPos, length)

        Log.d("xuhrrr", "getImageNow = $getImageNow, count = $count，bytes = ${bytes.size}")

        if (getImageNow == true && bytes.isNotEmpty()) {
            if (count < 10) {
                count++
                // 缓存当前帧数据
                tempImageData = bytes.copyOf()
                tempImageWidth = w
                tempImageHeight = h
                return@VideoCallBack true
            }

            // 达到第10帧，缓存图像数据但不立即处理
            tempImageData = bytes.copyOf()
            tempImageWidth = w
            tempImageHeight = h

            // 重置计数器但不停止视频
            count = 0

        }

        true
    }


    // 在 OcrService 类中添加公共方法
    fun resetOcrState() {
        hasReportedResult.set(false)
        Log.d(TAG, "OCR状态已重置，允许下一次识别")
    }


    override fun onDestroy() {
        super.onDestroy()
        // 确保悬浮窗被移除
        dismissFloatView()

        // 清理相机资源
        CameraUtils.closeCamera()

        // 清理通知
        ForegroundNotificationUtils.deleteForegroundNotification(this)

        // 移除观察者
        state.removeObserver(this)
    }


    private fun takePhoto() {
        getLastImage()
    }

    private fun dismissWindow() {
        dismissFloatView()
    }

    private fun showWindow() {
        ThreadUtils.runOnUiThread {
            Log.d(TAG, "准备显示悬浮窗，当前线程: ${Thread.currentThread().name}")
            showFloatView()
        }
    }


    //**********************************************************************************
    fun getLastImage() {
        getImageNow = true
        count = 0 // 可选：强制重置计数器
        mScanner?.startScan()
        mScanner?.startVideo(500)
        mScanner?.setVideoCallBack(mVideoCallBack)
    }




    override fun onChanged(value: FloatWindowState) {
        val floatWindowState = value as? FloatWindowState
        if (floatWindowState == null) {
            return
        }
        Log.i(TAG, "service启动：$floatWindowState")
        when (floatWindowState) {
            FloatWindowState.SHOW_WINDOW -> {
                resetOcrState()  // 显示窗口前重置状态
                showWindow()
            }

            FloatWindowState.TAKE_PHOTO -> {
                resetOcrState()  // 拍照前重置状态
                takePhoto()
            }

            else -> dismissWindow()
        }
    }


}
