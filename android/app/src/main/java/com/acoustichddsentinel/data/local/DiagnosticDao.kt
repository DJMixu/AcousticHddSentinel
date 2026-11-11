package com.acoustichddsentinel.data.local

import androidx.room.*
import kotlinx.coroutines.flow.Flow

/**
 * Interfejs DAO dla tabeli diagnostics.
 */
@Dao
interface DiagnosticDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(diagnostic: DiagnosticEntity): Long

    @Query("SELECT * FROM diagnostics ORDER BY timestamp DESC")
    fun getAllDiagnostics(): Flow<List<DiagnosticEntity>>

    @Query("SELECT * FROM diagnostics WHERE id = :id LIMIT 1")
    suspend fun getDiagnosticById(id: Long): DiagnosticEntity?

    @Query("DELETE FROM diagnostics WHERE id = :id")
    suspend fun deleteById(id: Long)

    @Query("DELETE FROM diagnostics")
    suspend fun deleteAll()
}
