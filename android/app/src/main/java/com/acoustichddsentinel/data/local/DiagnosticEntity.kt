package com.acoustichddsentinel.data.local

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * Tabela bazy SQLite przechowująca historię diagnoz dysków twardych.
 * Zgodna ze specyfikacją specs.txt.
 */
@Entity(tableName = "diagnostics")
data class DiagnosticEntity(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,

    @ColumnInfo(name = "hdd_name")
    val hddName: String,

    @ColumnInfo(name = "timestamp")
    val timestamp: Long,

    @ColumnInfo(name = "predicted_class")
    val predictedClass: String,

    @ColumnInfo(name = "confidence")
    val confidence: Float,

    @ColumnInfo(name = "has_anomaly")
    val hasAnomaly: Boolean,

    @ColumnInfo(name = "spectrogram_path")
    val spectrogramPath: String? = null,

    @ColumnInfo(name = "audio_path")
    val audioPath: String? = null
)
