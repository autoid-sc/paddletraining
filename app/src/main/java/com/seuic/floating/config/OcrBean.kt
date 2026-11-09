package com.seuic.floating.config


/**
 *
 * @ProjectName:    seuic-ocr
 * @Description:     类作用描述
 * @Author:         KP
 * @CreateDate:     2021/7/27 17:29
 * @Version:        1.0
 */
/**
 * @param image NV21
 * @return
 */
data class OcrImageBean(val image: ByteArray, val width: Int, val height: Int)

/**
 * @description OCR信息展示bean
 * @param ocrStr  ocr信息
 * @param scope   ocr识别分数
 * @param useTime ocr识别耗时
 * @param imagePath ocr文件保存位置
 * @return
 */

data class OcrBean(
	var ocrStr: String,
	var isReplaced: Boolean,
	val scope: Double,
	val useTime: Long,
	val imagePath: String,
)
