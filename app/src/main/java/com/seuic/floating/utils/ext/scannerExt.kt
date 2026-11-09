package com.seuic.floating.utils.ext

import android.content.Context
import com.seuic.core.utils.ext.appContext
import com.seuic.scanner.Scanner
import com.seuic.scanner.ScannerFactory
import java.lang.Exception

/**
 *
 * @ProjectName:    seuic-ocr
 * @Description:     类作用描述
 * @Author:         KP
 * @CreateDate:     2021/11/10 10:55
 * @Version:        1.0
 */

val mScanner: Scanner by lazy {
	getScanner(appContext)!!
//	ScannerFactory.getScanner(appContext)
}

private fun getScanner(context: Context): Scanner? {
	val cls: Class<*> = ScannerFactory::class.java
	try {
		val method = cls.getDeclaredMethod("getScannerNoStopService",
			Context::class.java)
		method.isAccessible = true
		return method.invoke(null, context) as Scanner
	} catch (e: Exception) {
		e.printStackTrace()
	}
	return null
}