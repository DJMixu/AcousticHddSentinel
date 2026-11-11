package com.acoustichddsentinel.data.repository

import com.acoustichddsentinel.data.local.DiagnosticDao
import com.acoustichddsentinel.data.local.DiagnosticEntity
import kotlinx.coroutines.flow.Flow

/**
 * Repozytorium historii diagnoz HDD.
 */
class DiagnosticRepository(private val dao: DiagnosticDao) {
    val allDiagnostics: Flow<List<DiagnosticEntity>> = dao.getAllDiagnostics()

    suspend fun saveDiagnostic(entity: DiagnosticEntity): Long = dao.insert(entity)

    suspend fun getDiagnosticById(id: Long): DiagnosticEntity? = dao.getDiagnosticById(id)

    suspend fun deleteDiagnostic(id: Long) = dao.deleteById(id)

    suspend fun clearHistory() = dao.deleteAll()
}
