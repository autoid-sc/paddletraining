package com.seuic.floating.utils.ext

import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Rect
import android.media.Image


/**
 * @Desc
 * @author KP
 * @date 2021/7/8 8:59
 * @version
 */

/**
 * yuv420_888转Nv21
 * camerax与camera2返回数据类型为yuv420_888 解码支持类型为NV21与灰度类型
 * @param image yuv420_888数据
 * @return  Nv21数据
 * @create 2021/3/5 15:35
 */
fun Image.yuv420888ToNv21(): ByteArray {
    val pixelCount = cropRect.width() * cropRect.height()
    val pixelSizeBits = ImageFormat.getBitsPerPixel(ImageFormat.YUV_420_888)
    val outputBuffer = ByteArray(pixelCount * pixelSizeBits / 8)
    imageToByteBuffer(this, outputBuffer, pixelCount)
    return outputBuffer
}

private fun imageToByteBuffer(image: Image, outputBuffer: ByteArray, pixelCount: Int) {
    assert(image.format == ImageFormat.YUV_420_888)

    val imageCrop = image.cropRect
    val imagePlanes = image.planes

    imagePlanes.forEachIndexed { planeIndex, plane ->
        // How many values are read in input for each output value written
        // Only the Y plane has a value for every pixel, U and V have half the resolution i.e.
        //
        // Y Plane            U Plane    V Plane
        // ===============    =======    =======
        // Y Y Y Y Y Y Y Y    U U U U    V V V V
        // Y Y Y Y Y Y Y Y    U U U U    V V V V
        // Y Y Y Y Y Y Y Y    U U U U    V V V V
        // Y Y Y Y Y Y Y Y    U U U U    V V V V
        // Y Y Y Y Y Y Y Y
        // Y Y Y Y Y Y Y Y
        // Y Y Y Y Y Y Y Y
        val outputStride: Int

        // The index in the output buffer the next value will be written at
        // For Y it's zero, for U and V we start at the end of Y and interleave them i.e.
        //
        // First chunk        Second chunk
        // ===============    ===============
        // Y Y Y Y Y Y Y Y    V U V U V U V U
        // Y Y Y Y Y Y Y Y    V U V U V U V U
        // Y Y Y Y Y Y Y Y    V U V U V U V U
        // Y Y Y Y Y Y Y Y    V U V U V U V U
        // Y Y Y Y Y Y Y Y
        // Y Y Y Y Y Y Y Y
        // Y Y Y Y Y Y Y Y
        var outputOffset: Int

        when (planeIndex) {
            0 -> {
                outputStride = 1
                outputOffset = 0
            }
            1 -> {
                outputStride = 2
                // For NV21 format, U is in odd-numbered indices
                outputOffset = pixelCount + 1
            }
            2 -> {
                outputStride = 2
                // For NV21 format, V is in even-numbered indices
                outputOffset = pixelCount
            }
            else -> {
                // Image contains more than 3 planes, something strange is going on
                return@forEachIndexed
            }
        }

        val planeBuffer = plane.buffer
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride

        // We have to divide the width and height by two if it's not the Y plane
        val planeCrop = if (planeIndex == 0) {
            imageCrop
        } else {
            Rect(
                imageCrop.left / 2,
                imageCrop.top / 2,
                imageCrop.right / 2,
                imageCrop.bottom / 2
            )
        }

        val planeWidth = planeCrop.width()
        val planeHeight = planeCrop.height()

        // Intermediate buffer used to store the bytes of each row
        val rowBuffer = ByteArray(plane.rowStride)

        // Size of each row in bytes
        val rowLength = if (pixelStride == 1 && outputStride == 1) {
            planeWidth
        } else {
            // Take into account that the stride may include data from pixels other than this
            // particular plane and row, and that could be between pixels and not after every
            // pixel:
            //
            // |---- Pixel stride ----|                    Row ends here --> |
            // | Pixel 1 | Other Data | Pixel 2 | Other Data | ... | Pixel N |
            //
            // We need to get (N-1) * (pixel stride bytes) per row + 1 byte for the last pixel
            (planeWidth - 1) * pixelStride + 1
        }

        for (row in 0 until planeHeight) {
            // Move buffer position to the beginning of this row
            planeBuffer.position(
                (row + planeCrop.top) * rowStride + planeCrop.left * pixelStride
            )

            if (pixelStride == 1 && outputStride == 1) {
                // When there is a single stride value for pixel and output, we can just copy
                // the entire row in a single step
                planeBuffer.get(outputBuffer, outputOffset, rowLength)
                outputOffset += rowLength
            } else {
                // When either pixel or output have a stride > 1 we must copy pixel by pixel
                planeBuffer.get(rowBuffer, 0, rowLength)
                for (col in 0 until planeWidth) {
                    outputBuffer[outputOffset] = rowBuffer[col * pixelStride]
                    outputOffset += outputStride
                }
            }
        }
    }
}

/**
 * Bitmap转化为ARGB数据，再转化为NV21数据
 * @param src    传入的Bitmap，格式为Bitmap.Config.ARGB_8888
 * @param width  NV21图像的宽度
 * @param height NV21图像的高度
 * @return nv21数据
 */
fun bitmapToNv21(src: Bitmap?, width: Int, height: Int): ByteArray? {
    return if (src != null && src.width >= width && src.height >= height) {
        val argb = IntArray(width * height)
        src.getPixels(argb, 0, width, 0, 0, width, height)
        argbToNv21(argb, width, height)
    } else {
        null
    }
}

/**
 * ARGB数据转化为NV21数据
 *
 * @param argb   argb数据
 * @param width  宽度
 * @param height 高度
 * @return nv21数据
 */
private fun argbToNv21(argb: IntArray, width: Int, height: Int): ByteArray? {
    val frameSize = width * height
    var yIndex = 0
    var uvIndex = frameSize
    var index = 0
    val nv21 = ByteArray(width * height * 3 / 2)
    for (j in 0 until height) {
        for (i in 0 until width) {
            val R = argb[index] and 0xFF0000 shr 16
            val G = argb[index] and 0x00FF00 shr 8
            val B = argb[index] and 0x0000FF
            val Y = (66 * R + 129 * G + 25 * B + 128 shr 8) + 16
            val U = (-38 * R - 74 * G + 112 * B + 128 shr 8) + 128
            val V = (112 * R - 94 * G - 18 * B + 128 shr 8) + 128
            nv21[yIndex++] = (if (Y < 0) 0 else if (Y > 255) 255 else Y).toByte()
            if (j % 2 == 0 && index % 2 == 0 && uvIndex < nv21.size - 2) {
                nv21[uvIndex++] = (if (V < 0) 0 else if (V > 255) 255 else V).toByte()
                nv21[uvIndex++] = (if (U < 0) 0 else if (U > 255) 255 else U).toByte()
            }
            ++index
        }
    }
    return nv21
}