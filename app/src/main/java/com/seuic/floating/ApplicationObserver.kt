package com.seuic.floating

import android.content.Intent
import android.os.Build
import android.util.Log
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleObserver
import androidx.lifecycle.OnLifecycleEvent
import com.seuic.core.utils.ext.appContext
import com.seuic.floating.service.BaseInitializer
import com.seuic.floating.service.FloatingOcrService
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch


class ApplicationObserver : LifecycleObserver {
	private val TAG = this.javaClass.name
	private var refCount = 0
	/**
	 * 在应用程序的整个生命周期中只会被调用一次
	 */
	@OnLifecycleEvent(Lifecycle.Event.ON_CREATE)
	fun onCreate() {
		if (Build.VERSION.SDK_INT >= 26) {
			appContext.startForegroundService(Intent(appContext, FloatingOcrService::class.java))
		} else {
			appContext.startService(Intent(appContext, FloatingOcrService::class.java))
		}

		// 注册广播
		CoroutineScope(Dispatchers.Main).launch {
			delay(5000)
			// 初始化按键监听
			BaseInitializer.unRegisterScanKey()
			BaseInitializer.registerScanKey()
		}

	}
	@OnLifecycleEvent(Lifecycle.Event.ON_START)
	fun onStart() {
		Log.e("xuhrrrrr", "Lifecycle.Event.ON_START")


	}

	@OnLifecycleEvent(Lifecycle.Event.ON_RESUME)
	fun onResume() {
//		OcrService.releaseOcrDialog()
		Log.d(TAG, "Lifecycle.Event.ON_RESUME")
	}

	@OnLifecycleEvent(Lifecycle.Event.ON_PAUSE)
	fun onPause() {
//			BaseInitializer.disconnectCamera()
//			OcrService.stopOcrDialog()

	}

	@OnLifecycleEvent(Lifecycle.Event.ON_STOP)
	fun onStop() {


	}

	/**
	 * 永远不会被调用，系统不会分发调用 ON_DESTROY 事件
	 */
	@OnLifecycleEvent(Lifecycle.Event.ON_DESTROY)
	fun onDestroy() {
		Log.d(TAG, "Lifecycle.Event.ON_DESTROY")
		BaseInitializer.unRegisterScanKey()
	}


}