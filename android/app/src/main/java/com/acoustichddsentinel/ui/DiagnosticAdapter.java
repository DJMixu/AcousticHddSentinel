package com.acoustichddsentinel.ui;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageButton;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.core.content.ContextCompat;
import androidx.recyclerview.widget.DiffUtil;
import androidx.recyclerview.widget.RecyclerView;

import com.acoustichddsentinel.R;
import com.acoustichddsentinel.data.DiagnosticModel;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Adapter widoku listy historii diagnoz (RecyclerView.Adapter).
 */
public class DiagnosticAdapter extends RecyclerView.Adapter<DiagnosticAdapter.ViewHolder> {

    public interface OnItemActionListener {
        void onDeleteClick(DiagnosticModel item);
        void onItemClick(DiagnosticModel item);
    }

    private final List<DiagnosticModel> items = new ArrayList<>();
    private final OnItemActionListener listener;

    public DiagnosticAdapter(OnItemActionListener listener) {
        this.listener = listener;
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
                       oldItem.getConfidence() == newItem.getConfidence();
            }
        });

        items.clear();
        if (newItems != null) {
            items.addAll(newItems);
        }
        diffResult.dispatchUpdatesTo(this);
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
        } else {
            holder.tvBadge.setText("SPRAWNY");
            holder.tvBadge.setBackground(ContextCompat.getDrawable(context, R.drawable.bg_badge_normal));
            holder.tvBadge.setTextColor(ContextCompat.getColor(context, R.color.status_normal_text));
        }

        holder.btnDelete.setOnClickListener(v -> {
            if (listener != null) {
                listener.onDeleteClick(item);
            }
        });

        holder.itemView.setOnClickListener(v -> {
            if (listener != null) {
                listener.onItemClick(item);
            }
        });
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
        final ImageButton btnDelete;

        ViewHolder(@NonNull View itemView) {
            super(itemView);
            tvHddName = itemView.findViewById(R.id.tvHistoryHddName);
            tvDate = itemView.findViewById(R.id.tvHistoryDate);
            tvConfidence = itemView.findViewById(R.id.tvHistoryConfidence);
            tvBadge = itemView.findViewById(R.id.tvHistoryBadge);
            btnDelete = itemView.findViewById(R.id.btnHistoryDelete);
        }
    }
}
