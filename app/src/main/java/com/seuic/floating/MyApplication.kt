package com.seuic.floating

import androidx.lifecycle.ProcessLifecycleOwner
import com.seuic.floating.config.ConfigManager


class MyApplication : BaseApplication() {


    override fun onCreate() {
        super.onCreate()
        ProcessLifecycleOwner.get().lifecycle.addObserver(ApplicationObserver())
        ConfigManager.init(this)
        ConfigManager.loadOcrConfig()
    }


}