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

        switch (state.getStatus()) {
            case IDLE:
                stopPulseAnimation();
                binding.btnHeroShazam.setEnabled(true);
                binding.tvHeroHint.setText("Stuknij, aby zbadać stan HDD");
                binding.progressBarAudio.setVisibility(View.GONE);
                binding.progressBarAudio.setIndeterminate(false);
                binding.tvRecordingStatus.setText("Gotowy do badania (Shazam Mode)");
                break;

            case RECORDING:
                binding.cardResult.setVisibility(View.GONE);
                startPulseAnimation();
                binding.btnHeroShazam.setEnabled(true);
                binding.tvHeroHint.setText("Słucham dźwięków dysku... Zbliż telefon do HDD");
                binding.progressBarAudio.setVisibility(View.VISIBLE);
                binding.progressBarAudio.setIndeterminate(false);
                int progress = state.getRecordingTotalMs() > 0
                    ? (state.getRecordingElapsedMs() * 100) / state.getRecordingTotalMs()
                    : 0;
                binding.progressBarAudio.setProgress(progress);
                binding.tvRecordingStatus.setText(state.getStatusMessage());
                break;

            case PROCESSING:
                binding.cardResult.setVisibility(View.GONE);
                stopPulseAnimation();
                binding.btnHeroShazam.setEnabled(false);
                binding.tvHeroHint.setText("Analizowanie sygnatury akustycznej...");
                binding.progressBarAudio.setVisibility(View.VISIBLE);
                binding.progressBarAudio.setIndeterminate(true);
                binding.tvRecordingStatus.setText(state.getStatusMessage());
                break;

            case SUCCESS:
                stopPulseAnimation();
                binding.btnHeroShazam.setEnabled(true);
                binding.tvHeroHint.setText("Stuknij, aby zbadać ponownie");
                binding.progressBarAudio.setVisibility(View.GONE);
                binding.progressBarAudio.setIndeterminate(false);
                binding.tvRecordingStatus.setText("Diagnoza zakończona pomyślnie.");

                renderSuccessCard(state);
                break;

            case ERROR:
                stopPulseAnimation();
                binding.btnHeroShazam.setEnabled(true);
                binding.tvHeroHint.setText("Stuknij, aby spróbować ponownie");
                binding.progressBarAudio.setVisibility(View.GONE);
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
        binding.tvResultClass.setText(model.getPredictedClass().toUpperCase(Locale.US));

        if (model.hasAnomaly()) {
            binding.ivResultIcon.setImageResource(R.drawable.ic_warning);
            binding.tvResultBadge.setText("AWARIA");
            binding.tvResultBadge.setBackground(ContextCompat.getDrawable(this, R.drawable.bg_badge_anomaly));
            binding.tvResultBadge.setTextColor(ContextCompat.getColor(this, R.color.status_fault_text));
        } else {
            binding.ivResultIcon.setImageResource(R.drawable.ic_check_circle);
            binding.tvResultBadge.setText("SPRAWNY");
            binding.tvResultBadge.setBackground(ContextCompat.getDrawable(this, R.drawable.bg_badge_normal));
            binding.tvResultBadge.setTextColor(ContextCompat.getColor(this, R.color.status_normal_text));
        }

        binding.tvResultConfidence.setText(String.format(Locale.US,
            "Pewność: %.1f%% (czas inferencji: %d ms)",
            model.getConfidence() * 100.0f, result.getInferenceTimeMs()));

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
        MaterialAlertDialogBuilder builder = new MaterialAlertDialogBuilder(this)
            .setTitle(item.getHddName())
            .setMessage(String.format(Locale.US,
                "Data badania: %s\nKlasa: %s\nPewność: %.1f%%\nStatus: %s\nPlik spektrogramu:\n%s",
                item.getFormattedDate(),
                item.getPredictedClass().toUpperCase(Locale.US),
                item.getConfidence() * 100.0f,
                item.hasAnomaly() ? "WYKRYTO AWARIĘ" : "DYSK SPRAWNY",
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
}
