# Acoustic HDD Sentinel ProGuard Rules

# ONNX Runtime Mobile
-keep class ai.onnxruntime.** { *; }
-dontwarn ai.onnxruntime.**

# Room Database
-keep class * extends androidx.room.RoomDatabase
-dontwarn androidx.room.paging.**

# Keep Data Entities and Models
-keep class com.acoustichddsentinel.data.local.** { *; }
-keep class com.acoustichddsentinel.ml.** { *; }
-keep class com.acoustichddsentinel.dsp.** { *; }
