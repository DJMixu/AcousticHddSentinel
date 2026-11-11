package com.acoustichddsentinel.ui;

import android.app.Application;
import android.graphics.Bitmap;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.lifecycle.AndroidViewModel;
import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.acoustichddsentinel.audio.AudioChunk;
import com.acoustichddsentinel.audio.AudioRecorder;
import com.acoustichddsentinel.data.DiagnosticModel;
import com.acoustichddsentinel.data.DiagnosticRepository;
import com.acoustichddsentinel.dsp.AudioPreprocessor;
import com.acoustichddsentinel.dsp.DspConstants;
import com.acoustichddsentinel.dsp.SpectrogramImageSaver;
import com.acoustichddsentinel.ml.AcousticClassifier;
import com.acoustichddsentinel.ml.ClassificationResult;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Główny ViewModel aplikacji w architekturze MVVM.
 * Koordynuje rejestrator AudioRecord, preprocesor DSP, silnik inferencyjny ONNX Runtime oraz bazę danych SQLite.
 */
public class MainViewModel extends AndroidViewModel {

    private static final String TAG = "MainViewModel";

    private final MutableLiveData<DiagnosticUiState> uiState = new MutableLiveData<>(DiagnosticUiState.idle());
    private final MutableLiveData<List<DiagnosticModel>> historyList = new MutableLiveData<>(new ArrayList<>());

    private final AudioRecorder audioRecorder;
    private final AudioPreprocessor audioPreprocessor;
    private final DiagnosticRepository repository;
    private final ExecutorService backgroundExecutor;

    private AcousticClassifier classifier;

    public MainViewModel(@NonNull Application application) {
        super(application);
        this.audioRecorder = new AudioRecorder();
        this.audioPreprocessor = new AudioPreprocessor();
        this.repository = new DiagnosticRepository(application);
        this.backgroundExecutor = Executors.newFixedThreadPool(2);

        // Asynchroniczne załadowanie modelu ONNX
        backgroundExecutor.execute(() -> {
            try {
                classifier = new AcousticClassifier(application);
                Log.i(TAG, "Klasyfikator ONNX zainicjalizowany pomyślnie.");
            } catch (Exception e) {
                Log.e(TAG, "Błąd inicjalizacji klasyfikatora ONNX", e);
                uiState.postValue(DiagnosticUiState.error("Błąd ładowania modelu ONNX: " + e.getMessage()));
            }
        });

        loadHistory();
    }

    public LiveData<DiagnosticUiState> getUiState() {
        return uiState;
    }

    public LiveData<List<DiagnosticModel>> getHistoryList() {
        return historyList;
    }

    /**
     * Uruchamia procedurę diagnozy: rejestracja audio mikrofonem sprzętowym, DSP i klasyfikacja.
     *
     * @param hddName         nazwa lub model badanego dysku HDD
     * @param durationSeconds czas nagrywania (od 5 do 30 sekund)
     */
    public void startDiagnosis(String hddName, int durationSeconds) {
        final String effectiveHddName = (hddName == null || hddName.trim().isEmpty())
            ? "HDD Generic"
            : hddName.trim();

        if (classifier == null) {
            uiState.setValue(DiagnosticUiState.error("Silnik AI (ONNX) nie jest jeszcze gotowy."));
            return;
        }

        uiState.setValue(DiagnosticUiState.recording(0, durationSeconds * 1000, 0.0f));

        audioRecorder.startRecording(durationSeconds, new AudioRecorder.RecordCallback() {
            @Override
            public void onProgress(int elapsedMs, int totalMs, float currentRms) {
                uiState.postValue(DiagnosticUiState.recording(elapsedMs, totalMs, currentRms));
            }

            @Override
            public void onRecordingComplete(List<AudioChunk> chunks, float overallRms) {
                if (chunks == null || chunks.isEmpty()) {
                    uiState.postValue(DiagnosticUiState.error("Nie zarejestrowano wystarczającej ilości danych audio."));
                    return;
                }

                if (overallRms < DspConstants.RMS_SILENCE_THRESHOLD) {
                    uiState.postValue(DiagnosticUiState.error(
                        String.format(java.util.Locale.US,
                            "Sygnał audio zbyt cichy (RMS: %.4f < %.4f). Przyłóż mikrofon bliżej obudowy dysku HDD.",
                            overallRms, DspConstants.RMS_SILENCE_THRESHOLD)
                    ));
                    return;
                }

                uiState.postValue(DiagnosticUiState.processing("Przetwarzanie DSP (STFT, Mel, Delta) oraz inferencja CNN..."));
                processAndClassifyChunks(effectiveHddName, chunks);
            }

            @Override
            public void onError(String message) {
                uiState.postValue(DiagnosticUiState.error(message));
            }
        });
    }

    /**
     * Analityczny potok wielookienkowy w tle:
     * Jeżeli w KTÓRYMKOLWIEK oknie 3.0s wystąpi anomalia (clicking, beeping, scratching),
     * całe badanie klasyfikowane jest jako anomalia (has_anomaly = 1).
     */
    private void processAndClassifyChunks(String hddName, List<AudioChunk> chunks) {
        backgroundExecutor.execute(() -> {
            try {
                boolean hasAnyAnomaly = false;
                ClassificationResult bestAnomalyResult = null;
                AudioChunk representativeChunk = chunks.get(0);
                long totalInferenceTime = 0;

                Map<String, Float> accumulatedProbs = new HashMap<>();
                for (String cName : AcousticClassifier.CLASS_NAMES) {
                    accumulatedProbs.put(cName, 0.0f);
                }

                for (AudioChunk chunk : chunks) {
                    float[] flatTensor = audioPreprocessor.processAudioChunkToFlatTensor(chunk.getSamples());
                    ClassificationResult res = classifier.classifyChunk(flatTensor);
                    totalInferenceTime += res.getInferenceTimeMs();

                    for (Map.Entry<String, Float> entry : res.getProbabilities().entrySet()) {
                        accumulatedProbs.put(entry.getKey(), accumulatedProbs.get(entry.getKey()) + entry.getValue());
                    }

                    if (res.isAnomaly()) {
                        hasAnyAnomaly = true;
                        if (bestAnomalyResult == null || res.getConfidence() > bestAnomalyResult.getConfidence()) {
                            bestAnomalyResult = res;
                            representativeChunk = chunk;
                        }
                    }
                }

                int numChunks = chunks.size();
                Map<String, Float> averageProbs = new HashMap<>();
                for (Map.Entry<String, Float> entry : accumulatedProbs.entrySet()) {
                    averageProbs.put(entry.getKey(), entry.getValue() / numChunks);
                }

                String finalDominantClass;
                float finalConfidence;

                if (hasAnyAnomaly && bestAnomalyResult != null) {
                    finalDominantClass = bestAnomalyResult.getDominantClass();
                    finalConfidence = bestAnomalyResult.getConfidence();
                } else {
                    finalDominantClass = "normal";
                    finalConfidence = averageProbs.getOrDefault("normal", 0.99f);
                }

                // Generowanie spektrogramu Mel dla reprezentatywnego okna
                float[][] logMel = audioPreprocessor.extractLogMelForVisualization(representativeChunk.getSamples());
                Bitmap specBitmap = SpectrogramImageSaver.createSpectrogramBitmap(logMel);
                String specPath = SpectrogramImageSaver.saveSpectrogramPng(getApplication(), specBitmap, "spec_" + finalDominantClass);

                ClassificationResult overallResult = new ClassificationResult(
                    finalDominantClass,
                    finalConfidence,
                    averageProbs,
                    hasAnyAnomaly,
                    totalInferenceTime / numChunks
                );

                DiagnosticModel diagnosticModel = new DiagnosticModel(
                    hddName,
                    System.currentTimeMillis(),
                    hasAnyAnomaly,
                    specPath,
                    finalDominantClass,
                    finalConfidence
                );

                // Zapis do relacyjnej bazy SQLite
                repository.insert(diagnosticModel, id -> {
                    diagnosticModel.setId(id);
                    loadHistory();
                });

                uiState.postValue(DiagnosticUiState.success(diagnosticModel, overallResult, specBitmap));

            } catch (Exception e) {
                Log.e(TAG, "Błąd podczas przetwarzania wycinków audio", e);
                uiState.postValue(DiagnosticUiState.error("Błąd przetwarzania: " + e.getMessage()));
            }
        });
    }

    public void cancelRecording() {
        if (audioRecorder.isRecording()) {
            audioRecorder.stopRecording();
            uiState.setValue(DiagnosticUiState.idle());
        }
    }

    public void loadHistory() {
        repository.getAll(historyList::postValue);
    }

    public void deleteHistoryItem(DiagnosticModel item) {
        repository.deleteById(item.getId(), success -> loadHistory());
    }

    public void clearAllHistory() {
        repository.clearAll(success -> loadHistory());
    }

    @Override
    protected void onCleared() {
        super.onCleared();
        if (audioRecorder != null) {
            audioRecorder.stopRecording();
        }
        if (classifier != null) {
            classifier.close();
        }
        backgroundExecutor.shutdown();
    }
}
