
#-----------基本配置--------------
# 代码混淆压缩比，在0~7之间，默认为5，一般不需要改
-optimizationpasses 5

# 混淆时不使用大小写混合，混淆后的类名为小写
-dontusemixedcaseclassnames

# 指定不去忽略非公共的库的类
-dontskipnonpubliclibraryclasses

# 指定不去忽略非公共的库的类的成员
-dontskipnonpubliclibraryclassmembers

# 不做预校验，可加快混淆速度
# preverify是proguard的4个步骤之一
# Android不需要preverify，去掉这一步可以加快混淆速度
-dontpreverify

# 不优化输入的类文件
-dontoptimize

# 混淆时生成日志文件，即映射文件
-verbose

# 指定映射文件的名称
-printmapping proguardMapping.txt

#混淆时所采用的算法
-optimizations !code/simplification/arithmetic,!field/*,!class/merging/*

# 保护代码中的Annotation不被混淆
-keepattributes *Annotation*

# 忽略警告
-ignorewarnings

# 保护泛型不被混淆
-keepattributes Signature

# 抛出异常时保留代码行号
-keepattributes Exceptions,InnerClasses,Signature,Deprecated,SourceFile,LineNumberTable,Annotation,EnclosingMethod,MethodParameters

########kotlin###########
-keep class kotlin.** { *; }
-keep class kotlinx.** { *; }
-keep class kotlin.Metadata { *; }
-keep interface kotlinx.** { *; }
-dontwarn kotlinx.**
-keep class org.jetbrains.** { *; }
-keep interface org.jetbrains.** { *; }
-dontwarn org.jetbrains.**
-dontwarn com.ted.floo.**
-keep class kotlin.** { *; }
-keep interface kotlin.** { *; }
-keepclassmembers class kotlin.Metadata {
    public <methods>;
}
-keepclasseswithmembers @kotlin.Metadata class * { *; }
-keepclassmembers class **.WhenMappings {
    <fields>;
}
-assumenosideeffects class kotlin.jvm.internal.Intrinsics {
    static void checkParameterIsNotNull(java.lang.Object, java.lang.String);
}

#*RetrofitClient$是你自定义的调用了自定义拦截器的类
-keepclasseswithmembers @kotlin.Metadata class com.ted.floo.connect.http.*RetrofitClient$*{
      *;
}

-keep class retrofit2.** { *; }
-keep interface retrofit2.** { *; }
-dontwarn retrofit2.**

-keep class com.google.** { *; }
-keep interface com.google.** { *; }
-dontwarn com.google.**

-keep class com.google.protobuf.* {*;}
-dontwarn com.google.protobuf.**




#-----------需要保留的东西--------------
# 保留所有的本地native方法不被混淆
-keepclasseswithmembernames class * {
    native <methods>;
}

# 保留了继承自Activity、Application、Fragment这些类的子类
-keep public class * extends android.app.Fragment
-keep public class * extends android.app.Activity
-keep public class * extends android.app.Application
-keep public class * extends android.app.Service
-keep public class * extends android.content.BroadcastReceiver
-keep public class * extends android.content.ContentProvider
-keep public class * extends android.app.backup.BackupAgentHelper
-keep public class * extends android.preference.Preference
-keep public class * extends android.view.View
-keep public class com.android.vending.licensing.ILicensingService


# support-v4
-dontwarn android.support.v4.**
-keep class android.support.v4.** { *; }
-keep interface android.support.v4.** { *; }
-keep public class * extends android.support.v4.**
# support-v7
-dontwarn android.support.v7.**                                             #去掉警告
-keep class android.support.v7.** { *; }                                    #过滤android.support.v7
-keep interface android.support.v7.app.** { *; }
-keep public class * extends android.support.v7.**
# androidx
-keep class com.google.android.material.** {*;}
-keep class androidx.** {*;}
-keep public class * extends androidx.**
-keep interface androidx.** {*;}
-dontwarn com.google.android.material.**
-dontnote com.google.android.material.**
-dontwarn androidx.**

#----------------保护指定的类和类的成员，但条件是所有指定的类和类成员是要存在------------------------------------
-keepclasseswithmembernames class * {
    public <init>(android.content.Context, android.util.AttributeSet);
}

-keepclasseswithmembernames class * {
    public <init>(android.content.Context, android.util.AttributeSet, int);
}

# 保持自定义控件类不被混淆，指定格式的构造方法不去混淆
-keepclasseswithmembers class * {
    public <init>(android.content.Context);
    public <init>(android.content.Context, android.util.AttributeSet);
    public <init>(android.content.Context, android.util.AttributeSet, int);
}

# 保持自定义控件类不被混淆
-keep public class * extends android.view.View {
    public <init>(android.content.Context);
    public <init>(android.content.Context, android.util.AttributeSet);
    public <init>(android.content.Context, android.util.AttributeSet, int);
    public void set*(...);
    *** get*();
}

# 保留在Activity中的方法参数是View的方法
# 从而我们在layout里边编写onClick就不会被影响
-keepclassmembers class * extends android.app.Activity {
    public void *(android.view.View);
}

-keep public class * extends android.app.Fragment {
    <init>(...);
}

-keep public class * extends androidx.fragment.app.Fragment {
    <init>(...);
}

#保留startup
-keep class androidx.startup.AppInitializer
-keep class * extends androidx.startup.Initializer
-keep class * extends androidx.startup.Initializer {
    # Keep the public no-argument constructor while allowing other methods to be optimized.
   <fields>;
   <methods>;
}
-keep class org.greenrobot.greendao.**
-keepclassmembers class org.greenrobot.greendao.** { *; }
# If you do not use SQLCipher:
-dontwarn org.greenrobot.greendao.database.**
-keepclassmembers class * extends de.greenrobot.dao.AbstractDao {
    public static java.lang.String TABLENAME;
}
-keep class **$Properties { *; }


# 保留枚举 enum 类不被混淆
-keepclassmembers enum * {
    public static **[] values();
    public static ** valueOf(java.lang.String);
}

# 保留 Parcelable 不被混淆
-keep class * implements android.os.Parcelable {
    public static final android.os.Parcelable$Creator *;
}

# 保留 Serializable 不被混淆
-keepnames class * implements java.io.Serializable
-keepclassmembers class * implements java.io.Serializable {
    static final long serialVersionUID;
    private static final java.io.ObjectStreamField[] serialPersistentFields;
    !static !transient <fields>;
    !private <fields>;
    !private <methods>;
    private void writeObject(java.io.ObjectOutputStream);
    private void readObject(java.io.ObjectInputStream);
    java.lang.Object writeReplace();
    java.lang.Object readResolve();
}

# 不混淆资源类
-keepclassmembers class **.R$* {
 public static <fields>;
}

# 对于带有回调函数onXXEvent()的，不能被混淆
-keepclassmembers class * {
    void *(**On*Event);
}

#保留反射
-keep class kotlin.reflect.jvm.internal.** { *; }
-keep class kotlin.Metadata { *; }

#保留 Coroutines
# ServiceLoader support
-keepnames class kotlinx.coroutines.internal.MainDispatcherFactory {}
-keepnames class kotlinx.coroutines.CoroutineExceptionHandler {}

# Most of volatile fields are updated with AFU and should not be mangled
-keepclassmembers class kotlinx.coroutines.** {
    volatile <fields>;
}

# Same story for the standard library's SafeContinuation that also uses AtomicReferenceFieldUpdater
-keepclassmembers class kotlin.coroutines.SafeContinuation {
    volatile <fields>;
}

# These classes are only required by kotlinx.coroutines.debug.AgentPremain, which is only loaded when
# kotlinx-coroutines-core is used as a Java agent, so these are not needed in contexts where ProGuard is used.
-dontwarn java.lang.instrument.ClassFileTransformer
-dontwarn sun.misc.SignalHandler
-dontwarn java.lang.instrument.Instrumentation
-dontwarn sun.misc.Signal


# WebView
# fqcn.of.javascript.interface.for.Webview 换成javascriptinterface的类名
#-keepclassmembers class fqcn.of.javascript.interface.for.Webview {
#   public *;
#}
-keepclassmembers class * extends android.webkit.WebViewClient {
    public void *(android.webkit.WebView, java.lang.String, android.graphics.Bitmap);
    public boolean *(android.webkit.WebView, java.lang.String);
}
-keepclassmembers class * extends android.webkit.WebViewClient {
    public void *(android.webkit.WebView, java.lang.String);
}

###########三方############
###hilt/dagger###
 -dontwarn dagger.**
 -dontwarn com.squareup.javapoet.**
 -dontwarn com.google.common.**

 ###### GSON ##########
 -keepattributes Signature
 # For using GSON @Expose annotation
 -keepattributes *Annotation*
 # Gson specific classes
 -dontwarn sun.misc.**
 -keep class com.google.gson.stream.** { *; }
 # Application classes that will be serialized/deserialized over Gson
 # Prevent proguard from stripping interface information from TypeAdapterFactory,
 # JsonSerializer, JsonDeserializer instances (so they can be used in @JsonAdapter)
 -keep class * implements com.google.gson.TypeAdapterFactory
 -keep class * implements com.google.gson.JsonSerializer
 -keep class * implements com.google.gson.JsonDeserializer
 # Prevent R8 from leaving Data object members always null
 -keepclassmembers,allowobfuscation class * {
   @com.google.gson.annotations.SerializedName <fields>;
 }
 ##---------------End: proguard configuration for Gson  ----------

#####-------KEEP注解-----#########
-keep @androidx.annotation.Keep class **{
 <fields>;
 <methods>;
}

#保留 okhttp
# JSR 305 annotations are for embedding nullability information.
-dontwarn javax.annotation.**

# A resource is loaded with a relative path so the package of this class must be preserved.
-keepnames class okhttp3.internal.publicsuffix.PublicSuffixDatabase

# Animal Sniffer compileOnly dependency to ensure APIs are compatible with older versions of Java.
-dontwarn org.codehaus.mojo.animal_sniffer.*

# OkHttp platform used only on JVM and when Conscrypt dependency is available.
-dontwarn okhttp3.internal.platform.ConscryptPlatform
-dontwarn org.conscrypt.ConscryptHostnameVerifier

#保留 okio
# Animal Sniffer compileOnly dependency to ensure APIs are compatible with older versions of Java.
-dontwarn org.codehaus.mojo.animal_sniffer.*

-keep class com.seuic.multocr.**{*;}
#保留mvp接口

#####-------keep需要访问的class-----#########
-keep class com.seuic.ocr.mvp.ui.dialog.**{*; }
-keep class com.seuic.ocr.weight.**{*; }
-keep class com.seuic.ocr.bean.**{*; }
-keep class com.fsa.decoder.**{*; }
-keep class com.seuic.decoder.**{*; }
-keep class org.tensorflow.**{*; }
-keep class com.seuic.scankey.**{*; }
-keep class com.seuic.ddscanner.**{*; }
-keep class com.seuic.scanner.**{*; }
-keep class com.seuic.core.utils.**{*; }
-keep class com.seuic.core.utils.**{*; }
-keep class com.seuic.ocr.down.**{*; }
-keep class com.seuic.ocr.initializer.**{*; }
-keep class de.greenrobot.dao.**{*; }
-keep class com.seuic.ocr.down.db**{*; }


# 保留 GreenRobot EventBus 的核心类
-keep class de.greenrobot.event.** { *; }

# 保留带有 @Subscribe 注解的方法

