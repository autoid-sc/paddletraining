package com.seuic.floating.weight.scanner

import androidx.annotation.FloatRange

interface ISurfaceView {
    open  fun setImageDataCallback(imageDataCallback: ImageDataCallback?)
    open  fun isSupportZoom(): Boolean
    open fun setZoomValue(@FloatRange(from = 1.0, to = 2.0) value: Float): Boolean
}
