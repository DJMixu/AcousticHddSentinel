package com.acoustichddsentinel

import android.app.Application
import com.acoustichddsentinel.data.local.AppDatabase
import com.acoustichddsentinel.data.repository.DiagnosticRepository

/**
 * Główna klasa aplikacji Android.
 */
class AcousticSentinelApp : Application() {
    val database: AppDatabase by lazy { AppDatabase.getDatabase(this) }
    val repository: DiagnosticRepository by lazy { DiagnosticRepository(database.diagnosticDao()) }

    override fun onCreate() {
        super.onCreate()
    }
}
