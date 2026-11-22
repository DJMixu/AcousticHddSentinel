package com.acoustichddsentinel.ui;

import android.app.Activity;
import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageButton;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.core.content.ContextCompat;
import androidx.recyclerview.widget.DiffUtil;
import androidx.recyclerview.widget.RecyclerView;

import com.acoustichddsentinel.R;
import com.acoustichddsentinel.audio.AudioPlaybackHelper;
import com.acoustichddsentinel.data.DiagnosticModel;
import com.acoustichddsentinel.dsp.SpectrogramImageSaver;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.widget.ImageView;

import java.io.File;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * Adapter widoku listy historii diagnoz (RecyclerView.Adapter).
 * Obsługuje:
 * - Odtwarzanie zarejestrowanych wycinków audio (Play/Stop za pomocą AudioPlaybackHelper).
 * - Rozwijany panel szczegółów (Crest Factor, RMS, czas badania, rozkład prawdopodobieństw 4 klas, rekomendacje).
 * - Spójne zarządzanie zasobami audio.
 */
public class DiagnosticAdapter extends RecyclerView.Adapter<DiagnosticAdapter.ViewHolder> {

    public interface OnItemActionListener {
        void onDeleteClick(DiagnosticModel item);
        void onItemClick(DiagnosticModel item);
        default void onPlayAudioClick(DiagnosticModel item) {}
    }

    private final List<DiagnosticModel> items = new ArrayList<>();
    private final Set<Long> expandedItemIds = new HashSet<>();
    private final OnItemActionListener listener;
    private AudioPlaybackHelper playbackHelper;

    public DiagnosticAdapter(OnItemActionListener listener) {
        this(listener, null);
    }

    public DiagnosticAdapter(OnItemActionListener listener, @Nullable AudioPlaybackHelper playbackHelper) {
        this.listener = listener;
        this.playbackHelper = playbackHelper;
    }

    public void setAudioPlaybackHelper(AudioPlaybackHelper playbackHelper) {
        this.playbackHelper = playbackHelper;
        try {
            notifyDataSetChanged();
        } catch (NullPointerException ignored) {
            // W środowisku testów jednostkowych JVM bez zainicjalizowanego mObservers
        }
    }

    public void submitList(List<DiagnosticModel> newItems) {
        DiffUtil.DiffResult diffResult = DiffUtil.calculateDiff(new DiffUtil.Callback() {
            @Override
            public int getOldListSize() {
                return items.size();
            }

            @Override
            public int getNewListSize() {
                return newItems != null ? newItems.size() : 0;
            }

            @Override
            public boolean areItemsTheSame(int oldItemPosition, int newItemPosition) {
                return items.get(oldItemPosition).getId() == newItems.get(newItemPosition).getId();
            }

            @Override
            public boolean areContentsTheSame(int oldItemPosition, int newItemPosition) {
                DiagnosticModel oldItem = items.get(oldItemPosition);
                DiagnosticModel newItem = newItems.get(newItemPosition);
                return oldItem.getId() == newItem.getId() &&
                       oldItem.getTimestamp() == newItem.getTimestamp() &&
                       oldItem.hasAnomaly() == newItem.hasAnomaly() &&
                       Float.compare(oldItem.getConfidence(), newItem.getConfidence()) == 0 &&
                       String.valueOf(oldItem.getAudioPath()).equals(String.valueOf(newItem.getAudioPath()));
            }
        });

        items.clear();
        if (newItems != null) {
            items.addAll(newItems);
        }
        try {
            diffResult.dispatchUpdatesTo(this);
        } catch (NullPointerException ignored) {
            // W środowisku testów jednostkowych JVM bez zainicjalizowanego mObservers
        }
    }

    @NonNull
    @Override
    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
            .inflate(R.layout.item_diagnostic_history, parent, false);
        return new ViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
        DiagnosticModel item = items.get(position);
        Context context = holder.itemView.getContext();

        holder.tvHddName.setText(item.getHddName());
        holder.tvDate.setText(item.getFormattedDate());
        holder.tvConfidence.setText(String.format(Locale.US, "%.1f%%", item.getConfidence() * 100.0f));

        if (item.hasAnomaly()) {
            holder.tvBadge.setText(item.getPredictedClass().toUpperCase(Locale.US));
            holder.tvBadge.setBackground(ContextCompat.getDrawable(context, R.drawable.bg_badge_anomaly));
            holder.tvBadge.setTextColor(ContextCompat.getColor(context, R.color.status_fault_text));
        } else if ("ambient".equalsIgnoreCase(item.getPredictedClass())) {
            holder.tvBadge.setText("SZUM TŁA / BRAK DYSKU");
            holder.tvBadge.setBackground(ContextCompat.getDrawable(context, R.drawable.badge_ambient_background));
            holder.tvBadge.setTextColor(ContextCompat.getColor(context, R.color.status_ambient_text));
        } else {
            holder.tvBadge.setText("SPRAWNY");
            holder.tvBadge.setBackground(ContextCompat.getDrawable(context, R.drawable.bg_badge_normal));
            holder.tvBadge.setTextColor(ContextCompat.getColor(context, R.color.status_normal_text));
        }

        // Konfiguracja przycisku odtwarzania audio WAV
        String audioPath = item.getAudioPath();
        boolean hasAudio = audioPath != null && !audioPath.trim().isEmpty() && new File(audioPath).exists();

        if (hasAudio) {
            holder.btnPlayAudio.setVisibility(View.VISIBLE);
            boolean isThisPlaying = playbackHelper != null && playbackHelper.isPlaying(audioPath);
            holder.btnPlayAudio.setImageResource(isThisPlaying ? R.drawable.ic_audio_stop : R.drawable.ic_audio_play);

            holder.btnPlayAudio.setOnClickListener(v -> {
                if (playbackHelper != null) {
                    playbackHelper.togglePlayback(audioPath, new AudioPlaybackHelper.PlaybackEventListener() {
                        @Override
                        public void onPlaybackStarted(String filePath) {
                            notifyDataSetChanged();
                        }

                        @Override
                        public void onPlaybackStopped(String filePath) {
                            notifyDataSetChanged();
                        }

                        @Override
                        public void onPlaybackCompleted(String filePath) {
                            notifyDataSetChanged();
                        }

                        @Override
                        public void onPlaybackError(String filePath, String errorMessage) {
                            notifyDataSetChanged();
                            if (context instanceof Activity) {
                                Toast.makeText(context, errorMessage, Toast.LENGTH_SHORT).show();
                            }
                        }
                    });
                }
                if (listener != null) {
                    listener.onPlayAudioClick(item);
                }
            });
        } else {
            holder.btnPlayAudio.setVisibility(View.GONE);
        }

        // Rozwijana sekcja szczegółów
        boolean isExpanded = expandedItemIds.contains(item.getId());
        holder.layoutDetails.setVisibility(isExpanded ? View.VISIBLE : View.GONE);

        holder.tvCrestFactor.setText(String.format(Locale.US, "Crest: %.2f", item.getCrestFactor()));
        holder.tvDuration.setText(String.format(Locale.US, "Czas: %.1fs", item.getScanDurationS()));
        holder.tvPeakRms.setText(String.format(Locale.US, "RMS: %.3f", item.getPeakRms()));

        holder.tvProbNormal.setText(String.format(Locale.US, "Normal: %.0f%%", item.getProbNormal() * 100.0f));
        holder.tvProbClicking.setText(String.format(Locale.US, "Click: %.0f%%", item.getProbClicking() * 100.0f));
        holder.tvProbBeeping.setText(String.format(Locale.US, "Beep: %.0f%%", item.getProbBeeping() * 100.0f));
        holder.tvProbScratching.setText(String.format(Locale.US, "Scratch: %.0f%%", item.getProbScratching() * 100.0f));

        if (item.getRecommendation() != null && !item.getRecommendation().trim().isEmpty()) {
            holder.tvRecommendation.setVisibility(View.VISIBLE);
            holder.tvRecommendation.setText("Zalecenie: " + item.getRecommendation());
        } else {
            holder.tvRecommendation.setVisibility(View.GONE);
        }

        // Konfiguracja podglądu spektrogramu Mel i eksportu do Galerii
        String specPath = item.getSpectrogramPath();
        boolean hasSpec = specPath != null && !specPath.trim().isEmpty() && new File(specPath).exists();

        if (hasSpec) {
            Bitmap bm = BitmapFactory.decodeFile(specPath);
            if (bm != null) {
                holder.ivSpectrogram.setImageBitmap(bm);
                holder.ivSpectrogram.setVisibility(View.VISIBLE);
                holder.tvNoSpectrogram.setVisibility(View.GONE);
                holder.btnSaveSpectrogram.setVisibility(View.VISIBLE);

                holder.btnSaveSpectrogram.setOnClickListener(v -> {
                    Uri uri = SpectrogramImageSaver.exportSpectrogramToGalleryFromPath(
                            v.getContext(), specPath, "spec_" + item.getHddName() + "_" + item.getPredictedClass()
                    );
                    if (uri != null) {
                        Toast.makeText(v.getContext(), "Zapisano spektrogram w Galerii (Pictures/AcousticHddSentinel)", Toast.LENGTH_SHORT).show();
                    } else {
                        Toast.makeText(v.getContext(), "Błąd zapisu spektrogramu w Galerii", Toast.LENGTH_SHORT).show();
                    }
                });
            } else {
                holder.ivSpectrogram.setVisibility(View.GONE);
                holder.tvNoSpectrogram.setVisibility(View.VISIBLE);
                holder.btnSaveSpectrogram.setVisibility(View.GONE);
            }
        } else {
            holder.ivSpectrogram.setVisibility(View.GONE);
            holder.tvNoSpectrogram.setVisibility(View.VISIBLE);
            holder.btnSaveSpectrogram.setVisibility(View.GONE);
        }

        holder.btnDelete.setOnClickListener(v -> {
            if (listener != null) {
                listener.onDeleteClick(item);
            }
        });

        holder.itemView.setOnClickListener(v -> {
            long itemId = item.getId();
            if (expandedItemIds.contains(itemId)) {
                expandedItemIds.remove(itemId);
            } else {
                expandedItemIds.add(itemId);
            }
            notifyItemChanged(holder.getAdapterPosition());

            if (listener != null) {
                listener.onItemClick(item);
            }
        });
    }

    @Override
    public void onViewRecycled(@NonNull ViewHolder holder) {
        super.onViewRecycled(holder);
        holder.btnPlayAudio.setOnClickListener(null);
        holder.btnDelete.setOnClickListener(null);
        holder.itemView.setOnClickListener(null);
    }

    @Override
    public void onDetachedFromRecyclerView(@NonNull RecyclerView recyclerView) {
        super.onDetachedFromRecyclerView(recyclerView);
        if (playbackHelper != null) {
            playbackHelper.stop();
        }
    }

    @Override
    public int getItemCount() {
        return items.size();
    }

    static class ViewHolder extends RecyclerView.ViewHolder {
        final TextView tvHddName;
        final TextView tvDate;
        final TextView tvConfidence;
        final TextView tvBadge;
        final ImageButton btnPlayAudio;
        final ImageButton btnDelete;

        final View layoutDetails;
        final TextView tvCrestFactor;
        final TextView tvDuration;
        final TextView tvPeakRms;
        final TextView tvProbNormal;
        final TextView tvProbClicking;
        final TextView tvProbBeeping;
        final TextView tvProbScratching;
        final TextView tvRecommendation;
        final ImageView ivSpectrogram;
        final TextView tvNoSpectrogram;
        final View btnSaveSpectrogram;

        ViewHolder(@NonNull View itemView) {
            super(itemView);
            tvHddName = itemView.findViewById(R.id.tvHistoryHddName);
            tvDate = itemView.findViewById(R.id.tvHistoryDate);
            tvConfidence = itemView.findViewById(R.id.tvHistoryConfidence);
            tvBadge = itemView.findViewById(R.id.tvHistoryBadge);
            btnPlayAudio = itemView.findViewById(R.id.btnHistoryPlayAudio);
            btnDelete = itemView.findViewById(R.id.btnHistoryDelete);

            layoutDetails = itemView.findViewById(R.id.layoutHistoryDetails);
            tvCrestFactor = itemView.findViewById(R.id.tvHistoryCrestFactor);
            tvDuration = itemView.findViewById(R.id.tvHistoryDuration);
            tvPeakRms = itemView.findViewById(R.id.tvHistoryPeakRms);
            tvProbNormal = itemView.findViewById(R.id.tvHistoryProbNormal);
            tvProbClicking = itemView.findViewById(R.id.tvHistoryProbClicking);
            tvProbBeeping = itemView.findViewById(R.id.tvHistoryProbBeeping);
            tvProbScratching = itemView.findViewById(R.id.tvHistoryProbScratching);
            tvRecommendation = itemView.findViewById(R.id.tvHistoryRecommendation);
            ivSpectrogram = itemView.findViewById(R.id.ivHistorySpectrogram);
            tvNoSpectrogram = itemView.findViewById(R.id.tvHistoryNoSpectrogram);
            btnSaveSpectrogram = itemView.findViewById(R.id.btnHistorySaveSpectrogram);
        }
    }
}
