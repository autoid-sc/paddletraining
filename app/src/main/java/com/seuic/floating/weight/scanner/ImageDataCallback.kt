package com.seuic.floating.weight.scanner

interface ImageDataCallback {
    open fun onDataCallback(image: ByteArray, width: Int, height: Int)
}