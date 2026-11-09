package com.seuic.floating.service

import android.annotation.SuppressLint
import android.content.Context
import android.content.Intent
import android.graphics.PixelFormat
import android.os.Build
import android.os.IBinder
import android.provider.Settings
import android.util.Log
import android.view.Gravity
import android.view.LayoutInflater
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager
import androidx.lifecycle.LifecycleService
import com.seuic.floating.R
import com.seuic.floating.config.ConfigManager

class FloatingButtonService : LifecycleService() {
    private var floatingButtonView: View? = null
    private lateinit var windowManager: WindowManager
    private lateinit var layoutParams: WindowManager.LayoutParams
    private var isDialogShowing = false  // 标记对话框是否显示

    override fun onCreate() {
        super.onCreate()
        createFloatingButton()
    }

    private fun createFloatingButton() {
        // 检查悬浮窗权限
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M &&
            !Settings.canDrawOverlays(this)) {
            Log.e("FloatingButtonService", "没有悬浮窗权限")
            return
        }
        windowManager = getSystemService(Context.WINDOW_SERVICE) as WindowManager

        // 创建悬浮按钮布局
        floatingButtonView = LayoutInflater.from(this).inflate(R.layout.floating_button_layout, null)

        // 加载保存的位置信息
        val savedPosition = ConfigManager.loadFloatingButtonPosition()

        // 设置窗口参数
        layoutParams = WindowManager.LayoutParams().apply {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                type = WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
            } else {
                type = WindowManager.LayoutParams.TYPE_PHONE
            }

            width = dpToPx(80)
            height = dpToPx(80)
            format = PixelFormat.TRANSLUCENT
            flags = WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                    WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS
            gravity = Gravity.TOP or Gravity.START
            x = savedPosition.first
            y = savedPosition.second
        }

        // 添加到窗口
        windowManager.addView(floatingButtonView, layoutParams)

        // 设置触摸事件处理
        setupTouchHandling()
    }

    @SuppressLint("ClickableViewAccessibility")
    private fun setupTouchHandling() {
        var initialX: Int = 0
        var initialY: Int = 0
        var initialTouchX: Float = 0f
        var initialTouchY: Float = 0f

        floatingButtonView?.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    // 记录初始位置和触摸点
                    initialX = layoutParams.x
                    initialY = layoutParams.y
                    initialTouchX = event.rawX
                    initialTouchY = event.rawY

                    // 按下时启动OCR对话框
                    FloatingOcrService.startOcrDialog()
                    true
                }
                MotionEvent.ACTION_UP -> {
                    // 抬起时停止OCR对话框
                    FloatingOcrService.stopOcrDialog()

                    // 保存当前位置
                    ConfigManager.saveFloatingButtonPosition(layoutParams.x, layoutParams.y)
                    true
                }
                MotionEvent.ACTION_MOVE -> {
                    // 计算位移并更新位置
                    val deltaX = event.rawX - initialTouchX
                    val deltaY = event.rawY - initialTouchY

                    layoutParams.x = (initialX + deltaX).toInt()
                    layoutParams.y = (initialY + deltaY).toInt()

                    windowManager.updateViewLayout(floatingButtonView, layoutParams)
                    true
                }
                else -> false
            }
        }
    }

    private fun dpToPx(dp: Int): Int {
        return (dp * resources.displayMetrics.density).toInt()
    }

    override fun onDestroy() {
        super.onDestroy()
        try {
            // 在销毁时保存最终位置
            floatingButtonView?.let {
                ConfigManager.saveFloatingButtonPosition(layoutParams.x, layoutParams.y)
                windowManager.removeViewImmediate(it)
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
}
