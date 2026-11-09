package com.seuic.floating;

import android.content.Context;

import androidx.multidex.MultiDex;
import androidx.multidex.MultiDexApplication;


public class BaseApplication extends MultiDexApplication {
    private static volatile BaseApplication instance;

    public static synchronized BaseApplication getInstance() {
        if (instance == null) {
            throw new IllegalStateException("BaseApplication instance not initialized.");
        }
        return instance;
    }

    @Override
    public void onCreate() {
        super.onCreate();
        instance = this;

    }

    @Override
    protected void attachBaseContext(Context base) {
        super.attachBaseContext(base);
        MultiDex.install(this);
    }


    // 在系统退出时（onTerminate）解除注册
    @Override
    public void onTerminate() {
        super.onTerminate();

    }
}
