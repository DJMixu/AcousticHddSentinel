package com.acoustichddsentinel.ui;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.acoustichddsentinel.R;
import com.google.android.material.bottomsheet.BottomSheetDialogFragment;

/**
 * Interaktywne okno Bottom Sheet zawierające kompletną bazę wiedzy i FAQ
 * dotyczące akustycznej diagnostyki dysków twardych, procedur pomiarowych oraz interpretacji awarii.
 */
public class FaqBottomSheetDialog extends BottomSheetDialogFragment {

    public static final String TAG = "FaqBottomSheetDialog";

    public static FaqBottomSheetDialog newInstance() {
        return new FaqBottomSheetDialog();
    }

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container,
                             @Nullable Bundle savedInstanceState) {
        return inflater.inflate(R.layout.dialog_faq, container, false);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        View btnClose = view.findViewById(R.id.btnFaqClose);
        if (btnClose != null) {
            btnClose.setOnClickListener(v -> dismiss());
        }
    }
}
