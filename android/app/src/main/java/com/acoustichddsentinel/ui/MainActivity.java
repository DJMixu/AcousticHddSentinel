package com.acoustichddsentinel.ui;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.view.View;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.LinearLayoutManager;

import com.acoustichddsentinel.R;
import com.acoustichddsentinel.data.DiagnosticModel;
import com.acoustichddsentinel.databinding.ActivityMainBinding;
import com.acoustichddsentinel.ml.ClassificationResult;
import com.acoustichddsentinel.ml.ModelType;
import com.google.android.material.dialog.MaterialAlertDialogBuilder;

import java.io.File;
import java.util.Locale;
import java.util.Map;

/**
 * Główna aktywność aplikacji diagnostycznej (Widok w MVVM).
 * Obsługuje interakcję z użytkownikiem, uprawnienia systemowe RECORD_AUDIO,
 * suwak czasu nagrywania (5-30s), wizualizację spektrogramu oraz historię badań w SQLite.
 */
public class MainActivity extends AppCompatActivity {

    private ActivityMainBinding binding;
    private MainViewModel viewModel;
    private DiagnosticAdapter adapter;

    private final ActivityResultLauncher<String> requestPermissionLauncher =
        registerForActivityResult(new ActivityResultContracts.RequestPermission(), isGranted -> {
            if (isGranted) {
                startRecordingFromUi();
            } else {
                Toast.makeText(this, "Uprawnienie do mikrofonu jest niezbędne do analizy dźwięku HDD.", Toast.LENGTH_LONG).show();
            }
        });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        viewModel = new ViewModelProvider(this).get(MainViewModel.class);

        setupViews();
        observeViewModel();
    }

    private void setupViews() {
        // Czysta interakcja w stylu Shazam - pojedynczy przycisk Hero
        binding.btnHeroShazam.setOnClickListener(v -> handleHeroClick());

        // Przycisk czyszczenia historii
        binding.btnClearHistory.setOnClickListener(v -> showClearHistoryConfirmationDialog());

        // Konfiguracja listy historii (RecyclerView)
        adapter = new DiagnosticAdapter(new DiagnosticAdapter.OnItemActionListener() {
            @Override
            public void onDeleteClick(DiagnosticModel item) {
                viewModel.deleteHistoryItem(item);
                Toast.makeText(MainActivity.this, "Usunięto wpis diagnozy.", Toast.LENGTH_SHORT).show();
            }

            @Override
            public void onItemClick(DiagnosticModel item) {
                showHistoryItemDialog(item);
            }
        });

        binding.rvHistory.setLayoutManager(new LinearLayoutManager(this));
        binding.rvHistory.setAdapter(adapter);

        setupModelSelectionToggle();
        setupRawAudioToggle();
    }

    private void setupModelSelectionToggle() {
        binding.toggleGroupModel.addOnButtonCheckedListener((group, checkedId, isChecked) -> {
            if (!isChecked) {
                return;
            }
            ModelType targetType = (checkedId == R.id.btnModelAlternative) ? ModelType.ALTERNATIVE : ModelType.LEGACY;
            if (viewModel.getActiveModelType() != targetType) {
                viewModel.setModelType(targetType);
                Toast.makeText(this, getString(R.string.toast_model_switched, targetType.getTitle()), Toast.LENGTH_SHORT).show();
            }
        });
    }

    private void setupRawAudioToggle() {
        binding.switchRawAudio.setOnCheckedChangeListener((btn, isChecked) -> {
            if (viewModel.isRawAudioMode() != isChecked) {
                viewModel.setRawAudioMode(isChecked);
                int msgRes = isChecked ? R.string.toast_raw_audio_enabled : R.string.toast_raw_audio_disabled;
                Toast.makeText(this, msgRes, Toast.LENGTH_SHORT).show();
            }
        });
    }

    private void handleHeroClick() {
        DiagnosticUiState currentState = viewModel.getUiState().getValue();
        if (currentState != null && currentState.getStatus() == DiagnosticUiState.Status.RECORDING) {
            // Ponowne kliknięcie podczas słuchania zatrzymuje nagrywanie i natychmiast diagnozuje
            viewModel.stopRecording();
        } else {
            // Start nowego badania - schowaj poprzednią kartę wyniku
            binding.cardResult.setVisibility(View.GONE);
            checkPermissionAndStart();
        }
    }

    private void checkPermissionAndStart() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
            startRecordingFromUi();
        } else {
            requestPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO);
        }
    }

    private void startRecordingFromUi() {
        String hddName = binding.etHddName.getText() != null
            ? binding.etHddName.getText().toString().trim()
            : "";
        // Adaptacyjny tryb Shazam (Early Stopping po 3.0s lub do 9.0s z głosowaniem konsensusowym)
        viewModel.startDiagnosis(hddName);
    }

    private void observeViewModel() {
        viewModel.getUiState().observe(this, this::renderUiState);

        viewModel.getSelectedModelType().observe(this, modelType -> {
            int targetId = (modelType == ModelType.ALTERNATIVE) ? R.id.btnModelAlternative : R.id.btnModelLegacy;
            if (binding.toggleGroupModel.getCheckedButtonId() != targetId) {
                binding.toggleGroupModel.check(targetId);
            }
        });

        viewModel.getRawAudioMode().observe(this, isRaw -> {
            if (binding.switchRawAudio.isChecked() != isRaw) {
                binding.switchRawAudio.setChecked(isRaw);
            }
        });

        viewModel.getHistoryList().observe(this, list -> {
            adapter.submitList(list);
            if (list == null || list.isEmpty()) {
                binding.tvHistoryEmpty.setVisibility(View.VISIBLE);
                binding.rvHistory.setVisibility(View.GONE);
            } else {
                binding.tvHistoryEmpty.setVisibility(View.GONE);
                binding.rvHistory.setVisibility(View.VISIBLE);
            }
        });
    }

    private android.animation.ObjectAnimator pulseAnimatorX;
    private android.animation.ObjectAnimator pulseAnimatorY;

    private void startPulseAnimation() {
        if (pulseAnimatorX == null || pulseAnimatorY == null) {
            pulseAnimatorX = android.animation.ObjectAnimator.ofFloat(binding.viewPulseRing, "scaleX", 1.0f, 1.35f);
            pulseAnimatorY = android.animation.ObjectAnimator.ofFloat(binding.viewPulseRing, "scaleY", 1.0f, 1.35f);

            pulseAnimatorX.setDuration(800);
            pulseAnimatorY.setDuration(800);

            pulseAnimatorX.setRepeatCount(android.animation.ValueAnimator.INFINITE);
            pulseAnimatorY.setRepeatCount(android.animation.ValueAnimator.INFINITE);

            pulseAnimatorX.setRepeatMode(android.animation.ValueAnimator.REVERSE);
            pulseAnimatorY.setRepeatMode(android.animation.ValueAnimator.REVERSE);
        }
        binding.viewPulseRing.setVisibility(View.VISIBLE);
        if (!pulseAnimatorX.isRunning()) {
            pulseAnimatorX.start();
            pulseAnimatorY.start();
        }
    }

    private void stopPulseAnimation() {
        if (pulseAnimatorX != null && pulseAnimatorX.isRunning()) {
            pulseAnimatorX.cancel();
            pulseAnimatorY.cancel();
        }
        binding.viewPulseRing.setScaleX(1.0f);
        binding.viewPulseRing.setScaleY(1.0f);
        binding.viewPulseRing.setVisibility(View.INVISIBLE);
    }

    private void renderUiState(DiagnosticUiState state) {
        if (state == null) return;

        boolean isBusy = (state.getStatus() == DiagnosticUiState.Status.RECORDING || state.getStatus() == DiagnosticUiState.Status.PROCESSING);
        binding.toggleGroupModel.setEnabled(!isBusy);
        binding.btnModelLegacy.setEnabled(!isBusy);
        binding.btnModelAlternative.setEnabled(!isBusy);
        binding.switchRawAudio.setEnabled(!isBusy);

        if (state.getStatus() == DiagnosticUiState.Status.RECORDING) {
            getWindow().addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        } else {
            getWindow().clearFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        }

        switch (state.getStatus()) {
            case IDLE:
                stopPulseAnimation();
                binding.btnHeroShazam.setEnabled(true);
                binding.tvHeroHint.setText("Stuknij, aby zbadać stan HDD");
                binding.layoutProgressSection.setVisibility(View.GONE);
                binding.progressBarAudio.setIndeterminate(false);
                binding.progressBarAudio.setProgress(0);
                binding.progressBarMicLevel.setProgress(0);
                binding.tvRecordingStatus.setText("Gotowy do badania (Shazam Mode)");
                break;

            case RECORDING:
                binding.cardResult.setVisibility(View.GONE);
                startPulseAnimation();
                binding.btnHeroShazam.setEnabled(true);
                binding.tvHeroHint.setText("Trwa badanie akustyczne HDD...");
                binding.layoutProgressSection.setVisibility(View.VISIBLE);
                binding.progressBarAudio.setIndeterminate(false);

                int elapsedMs = state.getRecordingElapsedMs();
                int totalMs = state.getRecordingTotalMs();
                int progress = totalMs > 0 ? Math.min(100, (elapsedMs * 100) / totalMs) : 0;
                if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.N) {
                    binding.progressBarAudio.setProgress(progress, true);
                } else {
                    binding.progressBarAudio.setProgress(progress);
                }

                // Czas badania
                binding.tvProgressTime.setText(String.format(Locale.US, "Czas: %.1f s / %.1f s", elapsedMs / 1000.0f, totalMs / 1000.0f));

                // Faza okna analitycznego
                if (elapsedMs < 3000) {
                    int pct3s = (elapsedMs * 100) / 3000;
                    binding.tvProgressWindow.setText(String.format(Locale.US, "Okno 1/5 • Wymagane min. 3.0s (%d%%)", pct3s));
                } else if (elapsedMs < 4500) {
                    binding.tvProgressWindow.setText("Okno 2/5 • Weryfikacja spójności");
                } else if (elapsedMs < 6000) {
                    binding.tvProgressWindow.setText("Okno 3/5 • Potwierdzenie profilu");
                } else if (elapsedMs < 7500) {
                    binding.tvProgressWindow.setText("Okno 4/5 • Konsensus wielookienkowy");
                } else {
                    binding.tvProgressWindow.setText("Okno 5/5 • Maksymalny czas");
                }

                // Poziom sygnału mikrofonu na żywo (VU Meter)
                float rms = state.getRecordingRms();
                int micPercent = (int) Math.min(100f, Math.max(0f, (float) (Math.sqrt(Math.max(0f, rms)) * 260.0f)));
                if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.N) {
                    binding.progressBarMicLevel.setProgress(micPercent, true);
                } else {
                    binding.progressBarMicLevel.setProgress(micPercent);
                }

                if (rms < 0.0004f) {
                    binding.tvMicLevelLabel.setText("Zbyt cicho • zbliż telefon do HDD");
                    binding.tvMicLevelLabel.setTextColor(ContextCompat.getColor(this, R.color.status_warning_text));
                } else if (rms < 0.25f) {
                    binding.tvMicLevelLabel.setText(String.format(Locale.US, "Optymalny (RMS: %.4f)", rms));
                    binding.tvMicLevelLabel.setTextColor(ContextCompat.getColor(this, R.color.status_normal_text));
                } else {
                    binding.tvMicLevelLabel.setText(String.format(Locale.US, "Bardzo głośny (RMS: %.3f)", rms));
                    binding.tvMicLevelLabel.setTextColor(ContextCompat.getColor(this, R.color.status_danger_text));
                }

                if (state.getStatusMessage() != null) {
                    binding.tvRecordingStatus.setText(state.getStatusMessage());
                }
                break;

            case PROCESSING:
                binding.cardResult.setVisibility(View.GONE);
                stopPulseAnimation();
                binding.btnHeroShazam.setEnabled(false);
                binding.tvHeroHint.setText("Analizowanie sygnatury akustycznej...");
                binding.layoutProgressSection.setVisibility(View.VISIBLE);
                binding.tvProgressTime.setText("Przetwarzanie AI...");
                binding.tvProgressWindow.setText("Klasyfikacja 2D CNN");
                binding.progressBarAudio.setIndeterminate(true);
                binding.progressBarMicLevel.setProgress(100);
                binding.tvMicLevelLabel.setText("DSP & ONNX");
                binding.tvMicLevelLabel.setTextColor(ContextCompat.getColor(this, R.color.accent));
                binding.tvRecordingStatus.setText(state.getStatusMessage());
                break;

            case SUCCESS:
                stopPulseAnimation();
                binding.btnHeroShazam.setEnabled(true);
                binding.tvHeroHint.setText("Stuknij, aby zbadać ponownie");
                binding.layoutProgressSection.setVisibility(View.GONE);
                binding.progressBarAudio.setIndeterminate(false);
                binding.tvRecordingStatus.setText("Diagnoza zakończona pomyślnie.");

                renderSuccessCard(state);
                break;

            case ERROR:
                stopPulseAnimation();
                binding.btnHeroShazam.setEnabled(true);
                binding.tvHeroHint.setText("Stuknij, aby spróbować ponownie");
                binding.layoutProgressSection.setVisibility(View.GONE);
                binding.progressBarAudio.setIndeterminate(false);
                binding.tvRecordingStatus.setText("Błąd: " + state.getErrorMessage());

                Toast.makeText(this, state.getErrorMessage(), Toast.LENGTH_LONG).show();
                break;
        }
    }

    private void renderSuccessCard(DiagnosticUiState state) {
        DiagnosticModel model = state.getCompletedDiagnostic();
        ClassificationResult result = state.getClassificationResult();

        if (model == null || result == null) return;

        binding.cardResult.setVisibility(View.VISIBLE);
        binding.tvResultHddName.setText(model.getHddName());

        if (model.hasAnomaly()) {
            binding.ivResultIcon.setImageResource(R.drawable.ic_warning);
            binding.tvResultBadge.setText("AWARIA");
            binding.tvResultBadge.setBackground(ContextCompat.getDrawable(this, R.drawable.bg_badge_anomaly));
            binding.tvResultBadge.setTextColor(ContextCompat.getColor(this, R.color.status_fault_text));
            binding.tvResultClass.setText(formatFaultClassName(model.getPredictedClass()));
            binding.tvResultConfidence.setText(String.format(Locale.US,
                "Pewność: %.1f%% (czas inferencji: %d ms)",
                model.getConfidence() * 100.0f, result.getInferenceTimeMs()));
        } else if ("ambient".equalsIgnoreCase(model.getPredictedClass())) {
            binding.ivResultIcon.setImageResource(R.drawable.ic_info);
            binding.tvResultBadge.setText("BRAK OBECNOŚCI DYSKU");
            binding.tvResultBadge.setBackground(ContextCompat.getDrawable(this, R.drawable.bg_badge_ambient));
            binding.tvResultBadge.setTextColor(ContextCompat.getColor(this, R.color.status_ambient_text));
            binding.tvResultClass.setText("SZUM TŁA / INNE URZĄDZENIE");
            binding.tvResultConfidence.setText("Brak harmonicznych obrotów silnika 5400/7200 RPM");
        } else {
            binding.ivResultIcon.setImageResource(R.drawable.ic_check_circle);
            binding.tvResultBadge.setText("DYSK SPRAWNY");
            binding.tvResultBadge.setBackground(ContextCompat.getDrawable(this, R.drawable.bg_badge_normal));
            binding.tvResultBadge.setTextColor(ContextCompat.getColor(this, R.color.status_normal_text));
            String rpmInfo = result.getDetectedRpm() != null ? " (" + result.getDetectedRpm() + ")" : "";
            binding.tvResultClass.setText("PROFIL PRAWIDŁOWY" + rpmInfo);
            binding.tvResultConfidence.setText(String.format(Locale.US,
                "Pewność: %.1f%% (czas inferencji: %d ms)",
                model.getConfidence() * 100.0f, result.getInferenceTimeMs()));
        }

        binding.tvResultRecommendation.setText(result.getRecommendation());

        // Rozkład prawdopodobieństw Softmax
        Map<String, Float> probs = result.getProbabilities();
        float probNormal = probs.getOrDefault("normal", 0.0f);
        float probClicking = probs.getOrDefault("clicking", 0.0f);
        float probBeeping = probs.getOrDefault("beeping", 0.0f);
        float probScratching = probs.getOrDefault("scratching", 0.0f);

        binding.progressNormal.setProgress((int) (probNormal * 100.0f));
        binding.tvProbNormal.setText(String.format(Locale.US, "%.1f%%", probNormal * 100.0f));

        binding.progressClicking.setProgress((int) (probClicking * 100.0f));
        binding.tvProbClicking.setText(String.format(Locale.US, "%.1f%%", probClicking * 100.0f));

        binding.progressBeeping.setProgress((int) (probBeeping * 100.0f));
        binding.tvProbBeeping.setText(String.format(Locale.US, "%.1f%%", probBeeping * 100.0f));

        binding.progressScratching.setProgress((int) (probScratching * 100.0f));
        binding.tvProbScratching.setText(String.format(Locale.US, "%.1f%%", probScratching * 100.0f));

        // Podgląd wygenerowanego spektrogramu Mel
        if (state.getSpectrogramBitmap() != null) {
            binding.ivSpectrogramPreview.setImageBitmap(state.getSpectrogramBitmap());
        }
        if (model.getSpectrogramPath() != null) {
            binding.tvSpectrogramPath.setText(model.getSpectrogramPath());
        }
    }

    private void showClearHistoryConfirmationDialog() {
        new MaterialAlertDialogBuilder(this)
            .setTitle("Wyczyścić historię badań?")
            .setMessage("Spowoduje to trwałe usunięcie wszystkich wpisów z lokalnej bazy SQLite oraz powiązanych plików spektrogramów.")
            .setPositiveButton("Usuń wszystko", (dialog, which) -> viewModel.clearAllHistory())
            .setNegativeButton("Anuluj", null)
            .show();
    }

    private void showHistoryItemDialog(DiagnosticModel item) {
        String statusText;
        if (item.hasAnomaly()) {
            statusText = "WYKRYTO AWARIĘ";
        } else if ("ambient".equalsIgnoreCase(item.getPredictedClass())) {
            statusText = "SZUM TŁA / BRAK DYSKU";
        } else {
            statusText = "DYSK SPRAWNY";
        }

        MaterialAlertDialogBuilder builder = new MaterialAlertDialogBuilder(this)
            .setTitle(item.getHddName())
            .setMessage(String.format(Locale.US,
                "Data badania: %s\nKlasa: %s\nPewność: %.1f%%\nStatus: %s\nPlik spektrogramu:\n%s",
                item.getFormattedDate(),
                item.getPredictedClass().toUpperCase(Locale.US),
                item.getConfidence() * 100.0f,
                statusText,
                item.getSpectrogramPath() != null ? item.getSpectrogramPath() : "Brak"
            ));

        if (item.getSpectrogramPath() != null) {
            File imgFile = new File(item.getSpectrogramPath());
            if (imgFile.exists()) {
                Bitmap bmp = BitmapFactory.decodeFile(imgFile.getAbsolutePath());
                if (bmp != null) {
                    ImageView iv = new ImageView(this);
                    iv.setPadding(32, 16, 32, 16);
                    iv.setImageBitmap(bmp);
                    iv.setScaleType(ImageView.ScaleType.FIT_XY);
                    builder.setView(iv);
                }
            }
        }

        builder.setPositiveButton("Zamknij", null)
            .setNeutralButton("Usuń wpis", (dialog, which) -> viewModel.deleteHistoryItem(item))
            .show();
    }

    @Override
    protected void onStop() {
        super.onStop();
        if (viewModel != null) {
            viewModel.stopRecording();
        }
        getWindow().clearFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        stopPulseAnimation();
        if (pulseAnimatorX != null) {
            pulseAnimatorX.cancel();
            pulseAnimatorX = null;
        }
        if (pulseAnimatorY != null) {
            pulseAnimatorY.cancel();
            pulseAnimatorY = null;
        }
    }

    private static String formatFaultClassName(String rawClass) {
        if (rawClass == null) return "NIEZNANA AWARIA";
        switch (rawClass.toLowerCase(Locale.US)) {
            case "clicking": return "STUKANIE GŁOWIC (CLICK OF DEATH)";
            case "beeping": return "ZABLOKOWANE WRZECIONO (STICTION)";
            case "scratching": return "TARCIE GŁOWIC O TALERZ (HEAD CRASH)";
            default: return rawClass.toUpperCase(Locale.US);
        }
    }
}
