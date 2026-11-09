package com.seuic.floating.utils.ext

import android.graphics.drawable.Drawable
import android.view.MotionEvent
import android.widget.TextView


/**
 * @Desc
 * @author KP
 * @date 2021/7/9 9:12
 * @version
 */
enum class DrawableType(val pos: Int) {
    LEFT(0), TOP(1), RIGHT(2), BOTTOM(3)
}
typealias OnDrawableClickListener = (DrawableType, Drawable) -> Unit

fun TextView.setOnDrawableClickListener(listener: OnDrawableClickListener) {
    setOnTouchListener { v, event ->
        if (event.action == MotionEvent.ACTION_UP) {
            val drawableLeft: Drawable? = compoundDrawables[DrawableType.LEFT.pos]
            if (drawableLeft != null && event.rawX <= left + drawableLeft.bounds.width()) {
                listener(DrawableType.LEFT, drawableLeft)
                return@setOnTouchListener true
            }
            val drawableRight: Drawable? =
                compoundDrawables[DrawableType.RIGHT.pos]
            if (drawableRight != null && event.rawX >= right - drawableRight.bounds.width()) {
                listener(DrawableType.RIGHT, drawableRight)
                return@setOnTouchListener true
            }
        }

        false
    }
}