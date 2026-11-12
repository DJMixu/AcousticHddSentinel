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
import com.acoustichddsentinel.dsp.RpmHarmonicsAnalyzer;
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
    private final RpmHarmonicsAnalyzer rpmHarmonicsAnalyzer;
    private final DiagnosticRepository repository;
    private final ExecutorService backgroundExecutor;

    private AcousticClassifier classifier;

    public MainViewModel(@NonNull Application application) {
        super(application);
        this.audioRecorder = new AudioRecorder();
        this.audioPreprocessor = new AudioPreprocessor();
        this.rpmHarmonicsAnalyzer = new RpmHarmonicsAnalyzer();
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
     * Uruchamia procedurę diagnozy w adaptacyjnym stylu Shazam:
     * - Strumieniowa rejestracja audio mikrofonem sprzętowym
     * - Wnioskowanie w locie co 1.5 sekundy (od t = 3.0s)
     * - Wczesne zatrzymanie (Early Stopping), jeśli wynik jest jednoznaczny (pewność >= 80%)
     * - Do 5 okien weryfikacyjnych (max ~9.0s) z głosowaniem konsensusowym
     *
     * @param hddName nazwa lub model badanego dysku HDD
     */
    public void startDiagnosis(String hddName) {
        startDiagnosis(hddName, 9);
    }

    /**
     * Uruchamia procedurę diagnozy z określonym maksymalnym czasem nasłuchu.
     *
     * @param hddName    nazwa lub model badanego dysku HDD
     * @param maxSeconds maksymalny czas nasłuchu (np. 9 sekund dla 5 okien)
     */
    public void startDiagnosis(String hddName, int maxSeconds) {
        final String effectiveHddName = (hddName == null || hddName.trim().isEmpty())
            ? "HDD Generic"
            : hddName.trim();

        if (classifier == null) {
            uiState.setValue(DiagnosticUiState.error("Silnik AI (ONNX) nie jest jeszcze gotowy."));
            return;
        }

        uiState.setValue(DiagnosticUiState.recording(0, maxSeconds * 1000, 0.0f, "Słucham dźwięków dysku... Zbliż telefon do HDD"));

        List<ClassificationResult> windowResults = new ArrayList<>();
        List<AudioChunk> windowChunks = new ArrayList<>();
        final String[] detectedRpmContainer = new String[1];
        final boolean[] hasAnomalyContainer = new boolean[1];
        final ClassificationResult[] bestAnomalyContainer = new ClassificationResult[1];
        final AudioChunk[] repChunkContainer = new AudioChunk[1];

        audioRecorder.startAdaptiveListening(maxSeconds, new AudioRecorder.AdaptiveStreamCallback() {
            @Override
            public void onProgress(int elapsedMs, int maxMs, float currentRms) {
                DiagnosticUiState current = uiState.getValue();
                String msg = current != null ? current.getStatusMessage() : null;
                uiState.postValue(DiagnosticUiState.recording(elapsedMs, maxMs, currentRms, msg));
            }

            @Override
            public boolean onWindowReady(int windowIndex, AudioChunk chunk) {
                try {
                    float[] samples = chunk.getSamples();
                    float maxAbs = 0f;
                    for (float s : samples) {
                        float abs = Math.abs(s);
                        if (abs > maxAbs) maxAbs = abs;
                    }
                    float crestFactor = maxAbs / (chunk.getRms() + 1e-9f);

                    RpmHarmonicsAnalyzer.RpmResult rpmRes = rpmHarmonicsAnalyzer.analyzeChunk(samples);
                    if (rpmRes.hasRotation && rpmRes.detectedRpm != null) {
                        detectedRpmContainer[0] = rpmRes.nominalRpm + " RPM";
                    }

                    float[] flatTensor = audioPreprocessor.processAudioChunkToFlatTensor(samples);
                    ClassificationResult res = classifier.classifyChunk(flatTensor);

                    windowResults.add(res);
                    windowChunks.add(chunk);

                    if (repChunkContainer[0] == null) {
                        repChunkContainer[0] = chunk;
                    }

                    boolean isAnomaly = res.isAnomaly();
                    String dominantClass = res.getDominantClass();
                    float confidence = res.getConfidence();

                    // Bezpiecznik fizyczny: clicking bez wysokiego Crest Factora (< 3.0) i bez wysokiej pewności
                    if (isAnomaly && "clicking".equals(dominantClass) && crestFactor < 3.0f && confidence < 0.75f) {
                        isAnomaly = false;
                        dominantClass = "normal";
                    }

                    if (isAnomaly) {
                        hasAnomalyContainer[0] = true;
                        if (bestAnomalyContainer[0] == null || confidence > bestAnomalyContainer[0].getConfidence()) {
                            bestAnomalyContainer[0] = new ClassificationResult(
                                dominantClass, confidence, res.getProbabilities(), true,
                                res.getInferenceTimeMs(), detectedRpmContainer[0], null
                            );
                            repChunkContainer[0] = chunk;
                        }
                    }

                    // ===============================================================
                    // LOGIKA EARLY STOPPING (Jak w prawdziwym Shazam):
                    // ===============================================================

                    // 1. Ewidentna awaria mechaniczna (Clicking, Beeping, Scratching) z wysoką pewnością (>= 75%)
                    if (isAnomaly && confidence >= 0.75f) {
                        Log.i(TAG, "Early Stopping: Wykryto ewidentną awarię (" + dominantClass + ", " + (confidence * 100) + "%) w oknie " + windowIndex);
                        return true;
                    }

                    // 2. Ewidentny dysk sprawny lub dźwięk obcy (Normal) z wysoką pewnością (>= 80%)
                    if (!isAnomaly && confidence >= 0.80f) {
                        Log.i(TAG, "Early Stopping: Jednoznaczny stan sprawny/normalny (" + (confidence * 100) + "%) w oknie " + windowIndex);
                        return true;
                    }

                    // 3. Spójność dwóch kolejnych okien (Consensus Voting)
                    if (windowIndex >= 1) {
                        ClassificationResult prevRes = windowResults.get(windowIndex - 1);
                        if (dominantClass.equals(prevRes.getDominantClass()) && confidence >= 0.60f && prevRes.getConfidence() >= 0.60f) {
                            Log.i(TAG, "Early Stopping: Konsensus 2 kolejnych okien (" + dominantClass + ") w oknie " + windowIndex);
                            return true;
                        }
                    }

                    // Jeśli wynik wciąż niejednoznaczny (np. zakłócenie na początku) i są kolejne okna:
                    if (windowIndex < 4) {
                        int nextWindow = windowIndex + 2;
                        int elapsedMs = (windowIndex + 1) * 1500 + 1500;
                        uiState.postValue(DiagnosticUiState.recording(
                            elapsedMs, maxSeconds * 1000, chunk.getRms(),
                            "Słucham dalej dla potwierdzenia... (okno " + nextWindow + "/5)"
                        ));
                    }

                    return false; // Słuchamy dalej
                } catch (Exception e) {
                    Log.e(TAG, "Błąd analizy okna " + windowIndex, e);
                    return false;
                }
            }

            @Override
            public void onStreamComplete(List<AudioChunk> evaluatedChunks, float overallRms, boolean stoppedEarly) {
                if (evaluatedChunks == null || evaluatedChunks.isEmpty()) {
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

                uiState.postValue(DiagnosticUiState.processing("Finalizowanie diagnozy i generowanie raportu..."));
                finalizeDiagnostic(effectiveHddName, windowResults, windowChunks, repChunkContainer[0],
                    hasAnomalyContainer[0], bestAnomalyContainer[0], detectedRpmContainer[0]);
            }

            @Override
            public void onError(String message) {
                uiState.postValue(DiagnosticUiState.error(message));
            }
        });
    }

    /**
     * Finalizuje badanie: agreguje prawdopodobieństwa, zapisuje spektrogram oraz wynik do bazy SQLite.
     */
    private void finalizeDiagnostic(String hddName,
                                    List<ClassificationResult> windowResults,
                                    List<AudioChunk> windowChunks,
                                    AudioChunk representativeChunk,
                                    boolean hasMechanicalAnomaly,
                                    ClassificationResult bestAnomalyResult,
                                    String detectedRpmOverall) {
        backgroundExecutor.execute(() -> {
            try {
                if (windowResults.isEmpty() || representativeChunk == null) {
                    uiState.postValue(DiagnosticUiState.error("Brak danych analitycznych do wystawienia diagnozy."));
                    return;
                }

                int numWindows = windowResults.size();
                Map<String, Float> avgProbs = new HashMap<>();
                long totalInfTime = 0;

                for (ClassificationResult wr : windowResults) {
                    totalInfTime += wr.getInferenceTimeMs();
                    for (Map.Entry<String, Float> e : wr.getProbabilities().entrySet()) {
                        avgProbs.put(e.getKey(), avgProbs.getOrDefault(e.getKey(), 0.0f) + e.getValue() / numWindows);
                    }
                }

                String finalDominantClass;
                float finalConfidence;
                String customRecommendation;

                if (hasMechanicalAnomaly && bestAnomalyResult != null) {
                    finalDominantClass = bestAnomalyResult.getDominantClass();
                    finalConfidence = bestAnomalyResult.getConfidence();
                    customRecommendation = ClassificationResult.generateRecommendation(finalDominantClass, true);
                } else {
                    finalDominantClass = "normal";
                    finalConfidence = avgProbs.getOrDefault("normal", 0.90f);
                    if (detectedRpmOverall != null) {
                        customRecommendation = "Dysk sprawny (" + detectedRpmOverall + "). Prawidłowa prędkość obrotowa, brak uszkodzeń głowic.";
                    } else {
                        customRecommendation = "Dźwięk prawidłowy / otoczenie. Sieć neuronowa nie wykryła żadnych sygnatur uszkodzeń HDD.";
                    }
                }

                // Generowanie spektrogramu Mel dla reprezentatywnego okna
                float[][] logMel = audioPreprocessor.extractLogMelForVisualization(representativeChunk.getSamples());
                Bitmap specBitmap = SpectrogramImageSaver.createSpectrogramBitmap(logMel);
                String specPath = SpectrogramImageSaver.saveSpectrogramPng(getApplication(), specBitmap, "spec_" + finalDominantClass);

                ClassificationResult overallResult = new ClassificationResult(
                    finalDominantClass,
                    finalConfidence,
                    avgProbs,
                    hasMechanicalAnomaly,
                    totalInfTime / numWindows,
                    detectedRpmOverall,
                    customRecommendation
                );

                DiagnosticModel diagnosticModel = new DiagnosticModel(
                    hddName,
                    System.currentTimeMillis(),
                    hasMechanicalAnomaly,
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
                Log.e(TAG, "Błąd podczas finalizowania diagnozy", e);
                uiState.postValue(DiagnosticUiState.error("Błąd diagnozy: " + e.getMessage()));
            }
        });
    }

    public void stopRecording() {
        if (audioRecorder.isRecording()) {
            audioRecorder.stopRecording();
        }
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
