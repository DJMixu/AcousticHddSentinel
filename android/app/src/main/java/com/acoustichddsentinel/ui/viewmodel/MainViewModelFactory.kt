package com.acoustichddsentinel.ui.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.acoustichddsentinel.data.repository.DiagnosticRepository
import com.acoustichddsentinel.ml.AcousticClassifier

/**
 * Fabryka ViewModelu wstrzykująca instancje repozytorium i klasyfikatora.
 */
class MainViewModelFactory(
    private val repository: DiagnosticRepository,
    private val classifier: AcousticClassifier
) : ViewModelProvider.Factory {

    @Suppress("UNCHECKED_CAST")
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        if (modelClass.isAssignableFrom(MainViewModel::class.java)) {
            return MainViewModel(repository, classifier) as T
        }
        throw IllegalArgumentException("Nieznana klasa ViewModel: ${modelClass.name}")
    }
}
