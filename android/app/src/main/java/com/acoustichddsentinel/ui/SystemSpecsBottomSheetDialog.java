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
 * Dialog specyfikacji technicznej systemu Acoustic HDD Sentinel:
 * Parametry DSP (22050 Hz, N_FFT=1024, Hop=256, N_MELS=128),
 * Model ONNX INT8 (SMPC Option A, Dual-Model Strategy) oraz Fizyczni Strażnicy DSP RPM.
 */
public class SystemSpecsBottomSheetDialog extends BottomSheetDialogFragment {

    public static final String TAG = "SystemSpecsBottomSheetDialog";

    public static SystemSpecsBottomSheetDialog newInstance() {
        return new SystemSpecsBottomSheetDialog();
    }

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container,
                             @Nullable Bundle savedInstanceState) {
        return inflater.inflate(R.layout.dialog_system_specs, container, false);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        View btnClose = view.findViewById(R.id.btnSpecsClose);
        if (btnClose != null) {
            btnClose.setOnClickListener(v -> dismiss());
        }
    }
}
