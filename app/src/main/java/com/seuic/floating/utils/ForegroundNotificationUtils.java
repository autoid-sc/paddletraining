package com.seuic.floating.utils;


import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.Service;
import android.content.Context;
import android.content.pm.ServiceInfo;
import android.graphics.BitmapFactory;
import android.os.Build;

import com.seuic.floating.R;


public class ForegroundNotificationUtils {
    // 通知渠道的id
    private static final String CHANNEL_ID = "Keep alive totem";
    private static final int CHANNEL_POSITION = -1;
    public static void startForegroundNotification(Service service){
        String SERVICE_STARTED=service.getString(R.string.service_started);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O){
            //启动前台服务而不显示通知的漏洞已在 API Level 25 修复
            NotificationManager manager = (NotificationManager)service.getSystemService(Context.NOTIFICATION_SERVICE);
            NotificationChannel Channel = new NotificationChannel(CHANNEL_ID,SERVICE_STARTED,NotificationManager.IMPORTANCE_DEFAULT);
//            Channel.enableLights(true);//设置提示灯
//            Channel.setLightColor(Color.GREEN);//设置提示灯颜色
            Channel.setShowBadge(false);//显示logo
            Channel.setDescription("");//设置描述
            Channel.setLockscreenVisibility(Notification.VISIBILITY_PUBLIC); //设置锁屏可见 VISIBILITY_PUBLIC=可见
            Channel.enableVibration(false);
            Channel.setSound(null,null);
            manager.createNotificationChannel(Channel);

            Notification notification = new Notification.Builder(service,CHANNEL_ID)
                    .setContentTitle(SERVICE_STARTED)//标题
                    .setWhen(System.currentTimeMillis())
                    .setSmallIcon(R.mipmap.ic_launcher)//小图标一定需要设置,否则会报错(如果不设置它启动服务前台化不会报错,但是你会发现这个通知不会启动),如果是普通通知,不设置必然报错
                    .setLargeIcon(BitmapFactory.decodeResource(service.getResources(),R.mipmap.ic_launcher_round))
                    .build();
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                service.startForeground(CHANNEL_POSITION, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_CAMERA);
            } else {
                service.startForeground(CHANNEL_POSITION, notification);
            }
        }else {
            //利用漏洞在 API Level 18 及以上的 Android 系统中，启动前台服务而不显示通知
//            service.startForeground(Foreground_ID, new Notification());
            Notification notification = new Notification.Builder(service)
                    .setContentTitle(SERVICE_STARTED)//设置标题
                    .setWhen(System.currentTimeMillis())//设置创建时间
                    .setSmallIcon(R.mipmap.ic_launcher)//设置状态栏图标
                    .setLargeIcon(BitmapFactory.decodeResource(service.getResources(),R.mipmap.ic_launcher_round))//设置通知栏图标
                    .build();
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                service.startForeground(CHANNEL_POSITION, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_CAMERA);
            } else {
                service.startForeground(CHANNEL_POSITION, notification);
            }
        }
    }

    public static void deleteForegroundNotification(Service service){
        NotificationManager mNotificationManager = (NotificationManager) service.getSystemService(Context.NOTIFICATION_SERVICE);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel mChannel = mNotificationManager.getNotificationChannel(CHANNEL_ID);
            if (null != mChannel) {
                mNotificationManager.deleteNotificationChannel(CHANNEL_ID);
            }
        }else {
            mNotificationManager.cancel(CHANNEL_POSITION);
        }
    }

}
