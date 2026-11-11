package com.acoustichddsentinel.ui

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageButton
import android.widget.TextView
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import com.acoustichddsentinel.R
import com.acoustichddsentinel.data.local.DiagnosticEntity
import java.text.SimpleDateFormat
import java.util.*

/**
 * Adapter listy RecyclerView prezentujący historię wykonanych diagnoz.
 */
class DiagnosticHistoryAdapter(
    private val onDeleteClick: (Long) -> Unit
) : ListAdapter<DiagnosticEntity, DiagnosticHistoryAdapter.ViewHolder>(DiffCallback) {

    private val dateFormat = SimpleDateFormat("dd.MM.yyyy HH:mm", Locale.getDefault())

    class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val tvHddName: TextView = view.findViewById(R.id.tvHistoryHddName)
        val tvDate: TextView = view.findViewById(R.id.tvHistoryDate)
        val tvBadge: TextView = view.findViewById(R.id.tvHistoryBadge)
        val tvConfidence: TextView = view.findViewById(R.id.tvHistoryConfidence)
        val btnDelete: ImageButton = view.findViewById(R.id.btnHistoryDelete)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_diagnostic_history, parent, false)
        return ViewHolder(view)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val item = getItem(position)
        holder.tvHddName.text = item.hddName
        holder.tvDate.text = dateFormat.format(Date(item.timestamp))
        holder.tvConfidence.text = String.format("%.1f%%", item.confidence * 100f)

        if (item.hasAnomaly) {
            holder.tvBadge.text = item.predictedClass.uppercase(Locale.getDefault())
            holder.tvBadge.setBackgroundResource(R.drawable.bg_badge_anomaly)
            holder.tvBadge.setTextColor(ContextCompat.getColor(holder.itemView.context, R.color.status_danger_text))
        } else {
            holder.tvBadge.text = "NORMAL"
            holder.tvBadge.setBackgroundResource(R.drawable.bg_badge_normal)
            holder.tvBadge.setTextColor(ContextCompat.getColor(holder.itemView.context, R.color.status_normal_text))
        }

        holder.btnDelete.setOnClickListener {
            onDeleteClick(item.id)
        }
    }

    object DiffCallback : DiffUtil.ItemCallback<DiagnosticEntity>() {
        override fun areItemsTheSame(oldItem: DiagnosticEntity, newItem: DiagnosticEntity): Boolean =
            oldItem.id == newItem.id

        override fun areContentsTheSame(oldItem: DiagnosticEntity, newItem: DiagnosticEntity): Boolean =
            oldItem == newItem
    }
}
