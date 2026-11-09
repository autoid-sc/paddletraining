package com.seuic.floating.config

import androidx.annotation.FloatRange

interface OnLightToggleListener {
	fun onLightToggle(enabled: Boolean)
}

interface OnZoomChangeListener {
	fun onZoomChange(
		@FloatRange(
			from = 0.0,
			to = 100.0
		) value: Float
	)
}

interface CameraReleaseListener {
	fun release(): Boolean
}

typealias OcrImageListener = (OcrImageBean) -> Unit

typealias OcrListener = (image_nv21: ByteArray, width: Int, height: Int) -> Unit


enum class TaskState {
	TASK_START, TASK_FINISH, TASK_PAUSE,
}
typealias TemplateDownloadListener = (taskState: TaskState, width: Int, height: Int) -> Unit

