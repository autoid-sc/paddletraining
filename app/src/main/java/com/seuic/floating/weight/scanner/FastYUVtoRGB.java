package com.seuic.floating.weight.scanner;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.ScriptIntrinsicYuvToRGB;
import android.renderscript.Type;

import java.io.ByteArrayOutputStream;
import java.io.IOException;

/**
 * 使用RenderScript将视频YUV流转换为BMP
 * 注：这个类适用于CameraPreview不变的情况
 */
public class FastYUVtoRGB {
    private RenderScript rs;
    private ScriptIntrinsicYuvToRGB yuvToRgbIntrinsic;
    private Type.Builder yuvType, rgbaType;
    private Allocation in, out;
    private int lastYuvLength = -1;
    private int lastWidth = -1;
    private int lastHeight = -1;

    public FastYUVtoRGB(Context context) {
        rs = RenderScript.create(context);
        yuvToRgbIntrinsic = ScriptIntrinsicYuvToRGB.create(rs, Element.U8_4(rs));
    }


    public Bitmap convertYUVtoRGB(byte[] yuvData, int width, int height) {
        // 验证输入参数
        if (yuvData == null || yuvData.length == 0 || width <= 0 || height <= 0) {
            android.util.Log.e("FastYUVtoRGB", "Invalid input: yuvData=" + (yuvData != null ? yuvData.length : "null") + ", width=" + width + ", height=" + height);
            return null;
        }

        try {
            if (yuvType == null || yuvData.length != lastYuvLength || width != lastWidth || height != lastHeight) {
                lastYuvLength = yuvData.length;
                lastWidth = width;
                lastHeight = height;

                if (in != null) {
                    in.destroy();
                }
                if (out != null) {
                    out.destroy();
                }

                yuvType = new Type.Builder(rs, Element.U8(rs)).setX(yuvData.length);
                in = Allocation.createTyped(rs, yuvType.create(), Allocation.USAGE_SCRIPT);

                rgbaType = new Type.Builder(rs, Element.RGBA_8888(rs)).setX(width).setY(height);
                out = Allocation.createTyped(rs, rgbaType.create(), Allocation.USAGE_SCRIPT);
            }
            in.copyFrom(yuvData);
            yuvToRgbIntrinsic.setInput(in);
            yuvToRgbIntrinsic.forEach(out);
            Bitmap bmpout = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
            out.copyTo(bmpout);
            return bmpout;
        } catch (Exception e) {
            android.util.Log.e("FastYUVtoRGB", "convertYUVtoRGB failed", e);
            return null;
        }
    }

    public  Bitmap convertYUVtoRGB2(byte[] nv21, int width, int height) {
        // 验证输入参数
        if (nv21 == null || nv21.length == 0 || width <= 0 || height <= 0) {
            android.util.Log.e("FastYUVtoRGB", "Invalid input for convertYUVtoRGB2: nv21=" + (nv21 != null ? nv21.length : "null") + ", width=" + width + ", height=" + height);
            return null;
        }

        Bitmap bitmap = null;
        try {
            YuvImage image = new YuvImage(nv21, ImageFormat.NV21, width, height, null);
            ByteArrayOutputStream stream = new ByteArrayOutputStream();
            image.compressToJpeg(new Rect(0, 0, width, height), 100, stream);
            bitmap = BitmapFactory.decodeByteArray(stream.toByteArray(), 0, stream.size());
            stream.close();
        } catch (IOException e) {
            android.util.Log.e("FastYUVtoRGB", "convertYUVtoRGB2 failed", e);
        }
        return bitmap;
    }
}