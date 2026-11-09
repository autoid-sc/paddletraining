package com.seuic.floating.utils.ext

fun String.getIp() =
    Regex("((2[0-4]\\d|25[0-5]|[01]?\\d\\d?)\\.){3}(2[0-4]\\d|25[0-5]|[01]?\\d\\d?)").find(this)?.value