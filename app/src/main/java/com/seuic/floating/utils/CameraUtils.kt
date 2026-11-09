package com.seuic.floating.utils

import android.annotation.SuppressLint
import android.util.Log
import android.util.Size
import android.view.Surface
import android.view.View
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.LifecycleRegistry
import com.google.common.util.concurrent.ListenableFuture
import com.seuic.core.utils.ext.TAG
import com.seuic.core.utils.ext.appContext
import com.seuic.floating.config.OcrImageBean
import com.seuic.floating.config.OcrListener
import com.seuic.floating.utils.ext.yuv420888ToNv21
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 *
 * @ProjectName:    PSBC_OCR
 * @Description:     类作用描述
 * @Author:         KP
 * @CreateDate:     2021/9/17 18:33
 * @Version:        1.0
 */
class WindowLifecycleOwner : LifecycleOwner {
	private val mLifecycleRegistry = LifecycleRegistry(this)

	init {
		mLifecycleRegistry.currentState = Lifecycle.State.CREATED
	}

	fun start() {
		mLifecycleRegistry.currentState = Lifecycle.State.STARTED
	}

	fun destroy() {
		mLifecycleRegistry.currentState = Lifecycle.State.DESTROYED
	}

	override val lifecycle: Lifecycle
		get() = mLifecycleRegistry
}

object CameraUtils {
	private var activeLifecycleOwner: WindowLifecycleOwner? = null
	private var cameraProviderFuture: ListenableFuture<ProcessCameraProvider>? = null
	private var cameraProvider: ProcessCameraProvider? = null

	//    private var imageAnalyzer: ImageAnalysis? = null
	private var camera: Camera? = null
	private val cameraExecutor: ExecutorService by lazy { Executors.newSingleThreadExecutor() }
	private var imageAnalyzer: ImageAnalysis? = null

	private var mListener: ((OcrImageBean) -> Unit)? = null

	private val previewAnalysis: PreviewAnalyzer by lazy {
		PreviewAnalyzer { image, width, height ->
			mListener?.invoke(OcrImageBean(image, width, height))
		}
	}

	private var mIsZoomEnabled = false
	private var mZoomSize = 0F

	private var currentSessionId = 0
	private var isClosed = false

	/** Declare and bind preview, capture and analysis use cases */
	@SuppressLint("RestrictedApi")
	private fun bindCameraUseCases(lifecycleOwner: LifecycleOwner, view_finder: PreviewView?, sessionId: Int) {
// Get screen metrics used to setup camera for full screen resolution
		// Bind the CameraProvider to the LifeCycleOwner
		val cameraSelector =
			CameraSelector.Builder().requireLensFacing(CameraSelector.LENS_FACING_BACK)
				.build()
		cameraProviderFuture = ProcessCameraProvider.getInstance(appContext).apply {
			addListener({
				if (sessionId != currentSessionId || isClosed) {
					Log.w(TAG, "Session expired or closed ($sessionId != $currentSessionId or $isClosed), aborting bindCameraUseCases")
					return@addListener
				}
				// CameraProvider

				val provider: ProcessCameraProvider = get()
				cameraProvider = provider
				var preview: Preview? = null
				if (view_finder != null) {
					preview = Preview.Builder()
						// We request aspect ratio but no resolution
						// Set initial target rotation
						.setTargetResolution(Size(720, 960))
						.setTargetRotation(Surface.ROTATION_0)
						.build()
					// Attach the viewfinder's surface provider to preview use case
					preview.setSurfaceProvider(view_finder.surfaceProvider)
				}


				// ImageAnalysis
				imageAnalyzer = ImageAnalysis.Builder()
					.setImageQueueDepth(1)
					.setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
					// We request aspect ratio but no resolution
//					.setTargetAspectRatio(screenAspectRatio)
					.setTargetResolution(Size(720, 960))
					// Set initial target rotation, we will have to call this again if rotation changes
					// during the lifecycle of this use case
					.setTargetRotation(Surface.ROTATION_0)
					.build() // The analyzer can then be assigned to the instance
				imageAnalyzer?.setAnalyzer(cameraExecutor, previewAnalysis)

				// Must unbind the use-cases before rebinding them
				provider.unbindAll()

				runCatching {
					// A variable number of use-cases can be passed here -
					// camera provides access to CameraControl & CameraInfo
					camera = if (preview != null) {
						provider.bindToLifecycle(
							lifecycleOwner,
							cameraSelector,
							preview,
							imageAnalyzer
						)
					} else {
						provider.bindToLifecycle(
							lifecycleOwner,
							cameraSelector,
							imageAnalyzer
						)
					}.apply {
						//闪光灯开关
						cameraControl.enableTorch(false)  // 改为动态判断
//						cameraControl.enableTorch(shouldEnableTorch())  // 改为动态判断
//						if (mIsZoomEnabled && mZoomSize >= 0 && mZoomSize <= 100) {
//							cameraControl.setLinearZoom(mZoomSize / 100)
//						}
					}
				}.onFailure {
					Log.e(TAG, "Use case binding failed", it)
				}
			}, ContextCompat.getMainExecutor(appContext))
		}

	}
	// 添加判断方法
	private fun shouldEnableTorch(): Boolean {
		// 可以根据环境光线、用户设置等条件判断
		return false  // 默认关闭，或根据需要开启
	}

	fun openCamera(
		lifecycleOwner: LifecycleOwner? = null,
		preview: PreviewView? = null,
		listener: (OcrImageBean) -> Unit,
	) {
		val sessionId = ++currentSessionId
		isClosed = false
		previewAnalysis.reset()
		mListener = listener
		if (lifecycleOwner == null) {
			val customOwner = WindowLifecycleOwner()
			activeLifecycleOwner = customOwner
			customOwner.start()
			preview?.addOnAttachStateChangeListener(object : View.OnAttachStateChangeListener {
				override fun onViewAttachedToWindow(v: View) {
					customOwner.start()
				}

				override fun onViewDetachedFromWindow(v: View) {
					customOwner.destroy()
				}

			})
			bindCameraUseCases(customOwner, preview, sessionId)
		} else {
			bindCameraUseCases(lifecycleOwner, preview, sessionId)
		}
	}

	fun closeCamera() {
		currentSessionId++
		isClosed = true
		try {
			cameraProvider?.unbindAll()
		} catch (e: Exception) {
			Log.w(TAG, "Error unbinding camera provider", e)
		}
		activeLifecycleOwner?.destroy()
		activeLifecycleOwner = null
	}

	class PreviewAnalyzer(val listener: OcrListener) : ImageAnalysis.Analyzer {
		private var isCanInit: AtomicBoolean = AtomicBoolean(true)

		fun reset() {
			isCanInit.set(true)
		}

		//image默认高480  宽640
		@SuppressLint("UnsafeOptInUsageError", "UnsafeExperimentalUsageError")
		override fun analyze(image: ImageProxy) {
			image.image?.takeIf { it.width > 0 && it.height > 0 }
				?.also {
					val nv21 = it.yuv420888ToNv21()
					val rotateNV21 = ByteArray(nv21.size)
//				logI("旋转${image.imageInfo.rotationDegrees}")

					when (image.imageInfo.rotationDegrees) {
						0 -> {
							listener(nv21, it.width, it.height)
						}
						90 -> {
							Nv21Utils.NV21_rotate_to_90(nv21, rotateNV21, it.width, it.height)
							listener(rotateNV21, it.height, it.width)
						}
						180 -> {
							Nv21Utils.NV21_rotate_to_180(nv21, rotateNV21, it.width, it.height)
							listener(rotateNV21, it.width, it.height)
						}
						270 -> {
							Nv21Utils.NV21_rotate_to_270(nv21, rotateNV21, it.width, it.height)
							listener(rotateNV21, it.height, it.width)
						}
					}
				}
			image.close()
		}
	}


}