# Kontekst Projektu i Kompendium Wiedzy Inżynierskiej — Acoustic HDD Sentinel

> **Data aktualizacji:** 13 listopada 2026 r.  
> **Cel dokumentu:** Pełne, usystematyzowane podsumowanie wszystkich dotychczasowych etapów rozwoju, sesji badawczych, audytów forensicznych, decyzji architektonicznych (ADR-001, ADR-002), modułu mobilnego Android (czysta Java 17, UI Shazam), silnika hybrydowego z analizą harmonicznych RPM, strażników DSP i odporności OOD/Adversarial oraz testów laboratoryjnych na 4 fizycznych dyskach twardych.

---

## 1. Cel i Wymagania Główne Projektu

Projekt **Acoustic HDD Sentinel** to system wibroakustycznej detekcji i wieloklasowej klasyfikacji uszkodzeń mechanicznych dysków twardych (HDD) przy użyciu głębokich sieci konwolucyjnych (2D CNN) oraz fizycznych strażników cyfrowego przetwarzania sygnałów (DSP). System klasyfikuje stan nośnika na podstawie krótkich, 3-sekundowych okien dźwiękowych do jednej z 4 klas:
1. **`normal`** – sprawny dysk (laminarny szum łożyska hydrodynamicznego FDB, stabilne obroty talerzy 5400/7200 RPM, miękkie pozycjonowanie głowic),
2. **`clicking`** – uderzanie głowic o ogranicznik (*Click of Death*, awaria przedwzmacniacza, zerwanie zawieszenia głowicy lub uszkodzenie ścieżek servo),
3. **`scratching`** – fizyczne tarcie i żłobienie powierzchni talerzy magnetycznych przez zniszczone głowice (*head crash*, katastrofalne mikro-skrawanie nośnika),
4. **`beeping`** – zablokowanie osi silnika wrzeciona BLDC (*seized spindle motor*) lub przylepienie głowic do powierzchni talerza (*stiction*).
5. **Stan otoczenia (`ambient` w silniku hybrydowym / Androidzie):** brak kontaktu z pracującym dyskiem twardym, szum wentylatorów, mowa lub tło akustyczne.

---

## 2. Niezmienniki Techniczne (Invariants — ADR-001 & ADR-002)

1. **Brak Data Leakage (Zero Data Leakage):**
   Podział na zbiory `train`, `validate` i `test` **musi** odbywać się wyłącznie na poziomie całych nagrań źródłowych przed ich podziałem na 3-sekundowe okna czasowe. Zabrania się losowego `train_test_split` na pociętych wycinkach `.npy`.
2. **Kształt tensora wejściowego:**
   - Na dysku w plikach `.npy`: tensor `float32` o stałych wymiarach **`(128, 259, 3)`** (Channel-Last).
   - W pamięci modelu PyTorch: tensor `(B, 3, 128, 259)` (Channel-First). Klasa `AcousticHddSentinelCNN` transparentnie obsługuje oba układy.
3. **Parametry potoku DSP:**
   - Częstotliwość próbkowania: `TARGET_SR = 22050` Hz
   - Długość okna FFT: `N_FFT = 1024` (okno Hanna, szerokość pasma prążka $\Delta f = 21.53$ Hz)
   - Krok przemieszczenia: `HOP_LENGTH = 256` (krok $\approx 11.6$ ms – krytyczny dla zachowania transjentów o wysokim współczynniku szczytu *Crest Factor*)
   - Liczba filtrów Mel: `N_MELS = 128` w paśmie od $50.0$ Hz do $10000.0$ Hz
   - Czas okna ewaluacyjnego: `CHUNK_DURATION = 3.0` s ($66 150$ próbek)
   - Augmentacja nakładania: `OVERLAP = 1.5` s (50%) **wyłącznie dla zbioru `train`**; dla `validate` i `test` overlap wynosi bezwzględnie `0.0` s (krok co 3.0 s).
   - Noise Gate: odrzucanie ramek o energii $RMS < 0.002$ (eliminacja szumu własnego przetwornika).
4. **Izolacja normalizacji dwustopniowej:**
   - Krok 1 (Waveform): Odcięcie składowej stałej (DC offset) + normalizacja szczytowa Peak do $[-1.0, 1.0]$.
   - Krok 2 (Tensor): Standaryzacja Z-Score niezależnie per kanał ($\mu = 0.0$, $\sigma = 1.0$) z zabezpieczeniem $\epsilon = 10^{-6}$.
5. **Niezmiennik VRAM (16 GB dGPU + iGPU):**
   Środowisko graficzne i pulpit działają na procesorze iGPU (Ryzen 7 7700), dzięki czemu karta AMD Radeon RX 9060 XT (16 GB) ma 100% wolnej pamięci VRAM na zadania uczenia maszynowego ROCm oraz lokalne LLM. Modele LLM są wyładowywane (`keep_alive=0`) przed rozpoczęciem treningu PyTorch.

---

## 3. Chronologiczna Historia Rozwoju i Ewolucja Systemu

### Faza 1: Architektura Bazowa i Pipeline DSP (TensorFlow / Keras)
*(Identyfikatory sesji: `d7cdbb65`, `cb4488fd`)*
- Zdefiniowano reprezentację czasowo-częstotliwościową w postaci 3-kanałowego tensora `(128, 259, 3)` w [ADR-001](docx/adr/adr.md):
  - **Kanał 0:** Log-Mel Spektrogram (rezonanse, harmoniczne obrotowe silnika, piski),
  - **Kanał 1:** Delta – 1. pochodna czasowa Savitzky'ego-Golaya (transjenty, impulsowe klikanie głowic),
  - **Kanał 2:** Delta-Delta – 2. pochodna czasowa (dynamika przyspieszeń i zmian kątowych).
- Stworzono bazowy enkoder audio [ml/hdd_audio_encoder.py](ml/hdd_audio_encoder.py), kategoryzator [ml/prepare_audio_dataset.py](ml/prepare_audio_dataset.py) oraz profiler [tests/profile_normalization_quality.py](tests/profile_normalization_quality.py).
- Wdrożono pierwszy model 2D CNN w TensorFlow/Keras.

---

### Faza 2: Rygorystyczny Audyt Zewnętrzny, Stress-Testy i Analiza Forensyczna
*(Identyfikator sesji: `ad8cb723`)*
- Przygotowano pakiet audytowy [docx/AUDIT_REVIEW_BUNDLE.md](docx/AUDIT_REVIEW_BUNDLE.md) i poddano kod niezależnej weryfikacji.
- **Wykryte wady i anomalie (Acoustic Forensics):**
  - **DC Offset:** Wykryto silne przesunięcie składowej stałej ($+0.3766$), zaburzające Peak Normalization. Wdrożono centrowanie sygnału (`audio - np.mean(audio)`).
  - **Artefakty zapętlania (`np.tile`):** Zapętlanie krótkich nagrań generowało sztuczne skoki fazowe na stykach ramek, które sieć błędnie interpretowała jako stukanie głowicy (`clicking`).
  - **Shortcut Learning w klasie `normal`:** Syntetyczny generator zdrowych dysków filtrował szum powyżej 2400 Hz, przez co sieć uczyła się odróżniać sprawny dysk po braku wysokich częstotliwości.
- Opracowano [tests/stress_test.py](tests/stress_test.py) oraz [tests/stress_test_vram.py](tests/stress_test_vram.py) do weryfikacji stabilności pamięci RAM/VRAM i latencji p95 podczas wielogodzinnej inferencji.

---

### Faza 3: Pełna Migracja na PyTorch i Akceleracja AMD ROCm (RX 9060 XT)
*(Identyfikator sesji: `75303e14`)*
- Przepisano cały stos na czysty PyTorch (`AcousticHddSentinelCNN` w [ml/model_cnn.py](ml/model_cnn.py) z 1 206 756 parametrami).
- Usunięto TensorFlow/Keras ze środowiska i repozytorium.
- Skonfigurowano akcelerację AMD ROCm 7.2 / HIP dla architektury ISA `gfx1200` (RDNA 4) w dedykowanym środowisku `.venv_rocm`.
- **Wynik wydajnościowy:** Czas epoki treningowej skrócił się z ~16 s na CPU do **~1.9 s na GPU (ponad 8-krotne przyspieszenie)**.

---

### Faza 4: Rozwiązanie Problemu Scratching/Beeping i Fine-Tuning
*(Identyfikatory sesji: `0c62a67c`, `3c821c7b`)*
- Zdiagnozowano przyczyny załamania metryk ($F1 = 0.00$ dla `scratching` i $F1 \approx 0.20$ dla `beeping`):
  - Utrata oryginalnych nazw 82 plików z bazy Northwind doprowadziła do fałszywych heurystyk spektralnych.
  - Plik `Audio_Player_57.wav` okazał się duplikatem klikającego Samsunga SP8002 oznaczonym jako `beeping`.
  - Z-score na bardzo cichym szumie ($RMS < 0.002$) sztucznie rozciągał szum mikrofonu.
- Pobrano autentyczne pliki MP3 z serwera Northwind z oryginalnymi nazwami oraz pozyskano 11 nowych nagrań laboratoryjnych uszkodzeń.
- Zbalansowano zbiór `train` (1389 tensorów, pitch-shift $\pm 1.5$ półtonu dla klas rzadkich) i zachowano w 100% czysty zbiór `test` (117 tensorów bez augmentacji).
- Zastosowano `AdamW` (`weight_decay=1e-4`), `LabelSmoothingCrossEntropy(0.03)` oraz `ReduceLROnPlateau`.
- **Wynik:** Wszystkie klasy przekroczyły próg $F1 \ge 0.70$, a średnia **Macro F1 osiągnęła 84.63%** przy luki generalizacyjnej równej **2.04 p.p.**

---

### Faza 5: Usunięcie Nadmiarowości i Wdrożenie Seed Ensembling
*(Identyfikator sesji: `4f1178fe`)*
- Usunięto zduplikowany katalog roboczy `CNN/` i stare pliki aliasowe, zwalniając ponad 840 MB.
- Zaimplementowano technikę Seed Ensembling ([ml/train_ensemble.py](ml/train_ensemble.py)) tworzącą komitet modeli (`FusedEnsemblePyTorch`) o ziarnach 42, 123 i 777.
- Zintegrowano moduł eksportu modeli [ml/export_model.py](ml/export_model.py) wspierający ONNX FP32, skwantyzowany ONNX INT8 oraz TorchScript.

---

### Faza 6: Architektura Mobilna i Standard ADR-002
*(Identyfikator sesji: `3cd6e5ea`)*
- Sformułowano decyzję architektoniczną [docx/adr/adr-002-mobile-architecture.md](docx/adr/adr-002-mobile-architecture.md).
- Dokonano wyboru silnika inferencji brzegowej: **ONNX Runtime Mobile INT8** (redukcja rozmiaru o 74.6% z 4.61 MB do **1.17 MB**, czas inferencji 18–32 ms, parzystość z PyTorch $< 10^{-8}$).
- Dodano automatyczną synchronizację wyeksportowanych wag do katalogu assets aplikacji mobilnej (`android/app/src/main/assets/`).

---

### Faza 7: Pełna Implementacja Natywnej Aplikacji Android w Czystej Javie 17
*(Identyfikatory sesji: `d48d1a3`, `7a9fb66`, `5878593`)*
- Zgodnie z wymaganiami programowymi studiów i ADR-002, całą aplikację Android zaimplementowano w **100% w czystej Javie 17** (bez narzutu bibliotek uruchomieniowych Kotlina i Room):
  - **`audio/AudioRecorder.java`:** Pętla rejestracji dźwięku z buforem kołowym, wsparciem dla 22050 Hz oraz awaryjnego 44100 Hz.
  - **`dsp/AudioPreprocessor.java`:** Wierna implementacja Radix-2 FFT, okna Hanna, trójkątnego banku filtrów Mel i wag Savitzky'ego-Golaya w Javie dająca tensory tożsame z Librosa.
  - **`ml/AcousticClassifier.java`:** Inferencja `OrtSession` z optymalizacją pamięciową i mapowaniem etykiet.
  - **`data/DiagnosticDbHelper.java`:** Czysty `SQLiteOpenHelper` z pełną historią diagnoz i eksportem (wraz z `DiagnosticModel.java` i `DiagnosticRepository.java`).
  - **`ui/MainActivity.java` & `MainViewModel.java`:** Wzorzec MVVM z `LiveData`, interfejs w stylu Shazam z pulsującym radarem, animowanymi falami oraz mechanizmem **Early Stopping** (natychmiastowe zatrzymanie przy pewności $\ge 75-80\%$ lub konsensusie okien).
- Utworzono 8 kompleksowych klas testowych obejmujących 37 testów jednostkowych w czystej Javie (`AudioChunkTest`, `AudioRecorderDecimationTest`, `AudioPreprocessorTest`, `RpmHarmonicsAnalyzerTest`, `DspEndToEndStressTest`, `DspPhysicalGuardTest`, `DiagnosticModelTest`, `ClassificationResultTest`) – 37/37 (100%) zdanych w 2 s.

---

### Faza 8: Strażnik Fizyczny (DSP Guard), Odporność OOD i Ataki Adwersarialne
*(Identyfikatory sesji: `d088918`, `efd4929`)*
- Utworzono laboratorium weryfikacji odporności [tests/adversarial_lab/](tests/adversarial_lab/):
  - **Szum wentylatorów chłodzenia PC:** Analiza widma $1/f$ oraz częstotliwości przejścia łopatek (Blade Passing Frequency – BPF). Wentylatory generują łagodne, opadające widmo bez transjentów, które surowa sieć mogła mylić z tarciem talerza (`scratching`). Wdrożono detektor `is_aerodynamic_fan`.
  - **Przesterowania mikrofonów MEMS (Flat-Top Clipping):** Tanie mikrofony smartfonów w bliskim kontakcie z obudową dysku wchodzą w nasycenie (obcinanie wierzchołków fali). Obcięcie generuje sztuczne nieparzyste harmoniczne (wysokie THD), które sieć brała za tarcie. Wdrożono detekcję `clipping_ratio > 0.015` w strażniku DSP.
  - **Mowa ludzka w tle:** Zintegrowano model **Silero VAD (16 kHz)**, automatycznie odrzucający fragmenty zanieczyszczone głosem serwisanta.
  - **Bramkowanie Open-Set (OOD Prototype Rejection):** Wbudowano w warstwę sieci `AcousticHddSentinelCNN` i wyeksportowano bezpośrednio do grafu ONNX bufory prototypów cech klas uszkodzeń (`fault_prototypes`, `fault_thresholds`, `gamma`). Jeśli wektor cech odległy jest od prototypów znanych awarii, logity uszkodzeń są tłumione funkcją sigmoid w dół, uniemożliwiając fałszywy alarm na dźwiękach spoza rozkładu (OOD).

---

### Faza 9: Analizator Harmonicznych Rotacji Silnika Wrzeciona (RPM Harmonics Analyzer)
*(Moduły: [ml/rpm_harmonics_analyzer.py](ml/rpm_harmonics_analyzer.py) oraz w Javie w `AudioPreprocessor.java`)*
- Każdy obracający się dysk HDD emituje ściśle określoną częstotliwość podstawową wirowania silnika bezszczotkowego BLDC:
  - **5400 RPM:** $f_0 = 90.0$ Hz (dopuszczalne pasmo $86 - 94$ Hz)
  - **7200 RPM:** $f_0 = 120.0$ Hz (dopuszczalne pasmo $116 - 124$ Hz)
- Algorytm estymuje gęstość widmową mocy metodą Welcha (okno 4096 próbek, rozdzielczość $\approx 5.38$ Hz).
- Wykrywa prążek $f_0$, jego wielokrotności $2f_0, 3f_0, 4f_0$ oraz harmoniczne stojana silnika dla dysków 2.5" ($810 - 900$ Hz).
- Potwierdzenie rotacji wrzeciona stanowi bezdyskusyjny fizyczny dowód, że dysk jest włączony i wiruje, co wyklucza zablokowanie wrzeciona (`beeping`) i nakazuje tłumienie niepewnych wskazań awarii do klasy `normal`.

---

### Faza 10: Skalowanie Bazy Audio (Ground Truth) do 353 Nagrań i 6840 Tensorów
*(Identyfikator sesji: `e7b374f`)*
- Baza surowych plików laboratoryjnych w `data/raw/` została rozszerzona o nowe katalogi:
  - `archive_hdd/` (Internet Archive, laboratoryjne zbiory uszkodzeń),
  - `freesound_hdd/` (Freesound nagrania dysków),
  - `independent_web_hdd/` (nagrania z forów i serwisów odzyskiwania danych),
  - `verified_real/` (nagrania z rzeczywistych weryfikacji sprzętowych),
  - `computer_ambient/` i `outliers_oe/` (próbki dźwięków otoczenia do treningu odporności).
- Utworzono rejestr [data/raw/PROCESSED_SOURCES_REGISTRY.json](data/raw/PROCESSED_SOURCES_REGISTRY.json) śledzący unikalne hashe MD5 i metadane próbek.
- Liczba surowych nagrań osiągnęła **353 pliki audio** w `data/raw/`, a łączna liczba poprawnych tensorów `.npy` w `dataset_encoded/` wynosi **6 840 tensorów** (6 045 w train z 496 nagrań źródłowych, 279 w validate z 52 nagrań, 516 w test z 74 nagrań — 100% Zero Data Leakage).

---

### Faza 11: Rzeczywista Weryfikacja Laboratoryjna na 4 Fizycznych Dyskach HDD
*(Dokument: [docx/RAPORT_TESTOW_RZECZYWISTYCH.md](docx/RAPORT_TESTOW_RZECZYWISTYCH.md))*
- Przeprowadzono fizyczny eksperyment laboratoryjny na 4 sprawnych dyskach HDD z mikrofonem kontaktowym przetwornika Realtek ALC1220 ADC:
  1. **Toshiba P300 6.0 TB 3.5"** (`HDWD260`, 5400 RPM, 2525 h)
  2. **WD Blue 500 GB 2.5"** (`WD5000LPVX`, 5400 RPM, 14360 h)
  3. **Seagate Momentus 500 GB 2.5"** (`ST9500325AS`, 5400 RPM, 3866 h)
  4. **HGST Travelstar 500 GB 2.5"** (`HTS545050A7E680`, 5400 RPM, 6098 h)
- Zarejestrowano 16 minut autentycznego dźwięku (po 240 s na dysk w 4 fazach: IDLE, WRITE z `fdatasync`, READ seek bez cache, MIXED I/O) — łącznie **320 okien ewaluacyjnych po 3.0 s**.
- **Odkrycia fizyczne i wnioski:**
  - **WD Blue (2.5"):** Osiągnął **90.0% celności** (100% w spoczynku IDLE) dzięki szerokopasmowemu, gładkiemu szumowi łożyska FDB bez ostrych rezonansów.
  - **Seagate i HGST (2.5"):** Wykazały silne rezonanse obudowy na częstotliwościach stojana BLDC (Seagate: szpilka 823.6 Hz skupiająca 87.7% energii; HGST: 899.0 Hz skupiająca 61.4% energii). Sieć CNN, która w bazie uczącej widziała energię 800–1200 Hz głównie przy tarciu talerza, wahała się i wskazywała `scratching`, lecz z **bardzo niską pewnością (28% do 38%)**.
  - **Crest Factor:** We wszystkich 320 oknach Crest Factor wynosił $3.2 - 4.3$, co w **100% wyeliminowało fałszywe alarmy stukania głowic (`clicking`)**.
  - **Confidence Gating $\ge 50\%$:** Żaden fałszywy alarm na żadnym dysku nie przekroczył 40% pewności. Wprowadzenie progu $\ge 50\%$ **wyeliminowało 100% fałszywych alarmów** w całym badaniu.

---

### Faza 12: Filtr Antyaliasingowy Butterwortha, Bramka Otoczenia i Kompendium Inżynierskie
- **Kaskadowy filtr Butterwortha 4. rzędu w `AudioRecorder.java`:**
  W smartfonach nieobsługujących sprzętowo częstotliwości 22050 Hz przetwornik rejestruje z 44100 Hz. Zwykła decymacja 2:1 powodowałaby zjawisko aliasingu (nakładania się składowych $>11025$ Hz). Zaimplementowano 2-sekcyjny kaskadowy filtr IIR Butterwortha ($f_c = 9500$ Hz) działający w locie na buforze audio przed decymacją.
- **Bramka obecności dysku (`ambient`):**
  W `MainViewModel.java` wprowadzono zasadę: jeśli sieć nie wykrywa ewidentnej awarii i jednocześnie analizator RPM nie potwierdza rotacji 5400/7200 RPM, aplikacja zamiast `normal` wyświetla neutralny stan **„Dźwięk otoczenia / brak dysku”**, instruując użytkownika o konieczności przyłożenia mikrofonu do metalowej obudowy dysku.
- **Kompendium inżynierskie dla prowadzącego ([PROMPT_DOKUMENTACJA_DLA_PROWADZACEGO.md](PROMPT_DOKUMENTACJA_DLA_PROWADZACEGO.md)):**
  Opracowano skompresowany, 9000-liniowy dokument łączący kompendium fizyczne i sygnałowe, historię ewolucji modeli od wczesnego Kerasa do ONNX INT8 oraz kompletny audyt bezpieczeństwa numerycznego.

---

### Faza 13: Audyt Algorytmiczny DSP, Eliminacja Kotlina i Poprawki Parzystości (Milestone 1 Retry & Przygotowanie SMPC)
- **Rygorystyczny audyt ADR-002 (Czysta Java 17):**
  Wyeliminowano wszelkie pozostałości Kotlina z repozytorium (szablon `AcousticClassifierAndroid.kt` zastąpiony przez `AcousticClassifierAndroid.java`). Moduł Android kompiluje się w 100% bez bibliotek uruchomieniowych Kotlina (0 plików `.kt`, 0 runtime overhead).
- **Dynamiczne skalowanie okna tła harmonicznych (`RpmHarmonicsAnalyzer`):**
  Zdiagnozowano i usunięto błąd sztywnego okna $\pm 30$ Hz wokół harmonicznych: dla częstotliwości $> 1000$ Hz (np. 1440 Hz przy 7200 RPM) tolerancja $h\_tol = h \times f0 \times 0.025$ przekraczała 30 Hz, powodując wyzerowanie liczby binów tła i fałszywą detekcję szumu numerycznego FFT ($9.74 \times 10^{-6}$) jako 3. harmonicznej. Wprowadzono adaptacyjne okno $\pm \max(40.0, h\_tol \times 2.0)$ Hz w Pythonie i Javie.
- **Bramka bezwzględnej energii harmonicznych:**
  Dodano warunek, że pik harmoniczny musi przewyższać lokalne tło ($> 1.6\times$) ORAZ posiadać bezwzględną energię $\ge 5\%$ średniej widma (`h_peak > np.mean(spec) * 0.05` / `maxHVal > meanEnergy * 0.05f`), co definitywnie eliminuje artefakty numeryczne na zerowym poziomie tła.
- **Symetryczne wykluczenie tła lokalnego fundamentalnej w Javie:**
  Zawężono wykluczenie wokół $f_0$ do $\pm 1$ bin (`Math.abs(b - bestF0Bin) > 1`) symetrycznie w promieniu 5 binów, eliminując sztuczne zaniżenie mediany tła i redukując $f_0$ prominence głosu z 3.65 do $< 2.0$.
- **Kryteria detekcji grzebienia dla autentycznych silników 2-tonowych (FDB):**
  Umożliwiono detekcję rotacji przy $harmonic\_hits \ge 1$, $f0\_prominence \ge 3.0$, $comb\_score \ge 3.0$ pod warunkiem, że pojedyncze trafienie przypada ściśle na 2. harmoniczną ($2f_0$, mnożnik 2.0), przywracając 100% czułość dla cichych silników 5400/7200 RPM o dominującej 2. harmonicznej przy dowolnym SNR (od Clean do 0 dB).
- **Jednostkowe testy odrzucania głosu i 2-tonowego rotora w module Android:**
  Dodano `testHumanVoiceRejection` do `RpmHarmonicsAnalyzerTest.java` (sygnał $130 \pm 15$ Hz z wibrato 5 Hz) oraz `testTwoTonePureMotor5400RpmDetection`.
- **Mobilność urządzeń w `HybridAcousticSentinel`:**
  Dodano wsparcie dla parametru `device: Optional[Union[torch.device, str]] = None` oraz dynamiczne dopasowanie tensora wejściowego do urządzenia parametrów modelu (`t_input.to(model_device)`), zapobiegając błędom typu `RuntimeError: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor)`.
- **Wprowadzenie do Milestone 2 (Architektura SMPC — Sub-Center Multi-Prototype Cosine):**
  W ramach badań nad paradygmatem Fault-Only Learning (OOD) przygotowano specyfikację modułów SMPC pod kątem orkiestracji modeli lokalnych (Qwen 3.8 Architekt i Qwen 2.5 Coder przez most `scripts/local_llm_bridge.py`).

---

### Faza 14: Wdrożenie Milestone 2 — Architektura Wielomodalnych Podcentrów (SMPC) i OOD Hardening
- **Front-Loaded orkiestracja lokalnych modeli LLM na AMD Radeon RX 9060 XT:**
  Jako pierwszy krok etapu uruchomiono most `scripts/local_llm_bridge.py`:
  1. *Architekt Qwen 3.8 27B* (`--preset smpc-arch`): przygotowanie rygorystycznej specyfikacji Option A w ONNX (`MatMul` + `view` + `ReduceMax` bez `einsum`) oraz walidacji prototypów scratching w Gałęzi D.
  2. *Programista Qwen 2.5 Coder 14B* (`--coder-only`): implementacja kodów integracji dla PyTorcha, funkcji straty ArcMargin i ortogonalności.
  3. Pełne zwolnienie pamięci VRAM (`--free-vram`) przed wykonaniem obliczeń GPU w PyTorch.
- **Implementacja SMPC w `ml/model_cnn.py`:**
  - Wzbogacono klasę `AcousticHddSentinelCNN` o transparentne bufory prototypów `fault_prototypes` wspierające zarówno układ unimodalny `(3, 128)`, jak i wielomodalny SMPC `(3, M, 128)`.
  - Wdrożono Option A dla ONNX: rzutowanie cech L2, iloczyn macierzowy `torch.matmul(z_norm, p_flat.t())`, `view(-1, K, M)` i `torch.max(dim=-1)` eliminujące `einsum` i gwarantujące 100% kompatybilność z ONNX Runtime Mobile INT8 na Androidzie.
  - Zastosowano mechanizm `_load_from_state_dict` pre-hook automatycznie dopasowujący bufor do checkpointu.
  - Bezwzględnie zachowano niezmiennik liczby trenowalnych parametrów bazowych: **dokładnie 1,206,756 parametrów**.
- **Wzmocnienie Gałęzi D w `ml/hybrid_acoustic_sentinel.py`:**
  - Zastąpiono ślepy fallback `cnn_top_class in ["scratching"] -> ambient_pc` weryfikacją podobieństwa cosinusowego do wielomodalnych prototypów tarcia talerza.
  - Sygnały o $\cos(\theta) \ge \tau_{scratching}$ są klasyfikowane jako autentyczne uszkodzenie `scratching` (`is_anomaly=True`), podczas gdy szumy wentylatorów PC o niskim podobieństwie są bezpiecznie odrzucane do `ambient_pc` (`is_anomaly=False`).
- **Integracja w potoku treningowym `ml/train_cnn.py`:**
  - Dodano flagi CLI: `--use-smpc`, `--smpc-subcenters 3`, `--smpc-margin 0.25`, `--smpc-scale 16.0`, `--smpc-ortho-lambda 0.10`, `--smpc-ortho-threshold 0.30`.
  - Wdrożono funkcje `SubCenterArcMarginLoss`, `compute_orthogonality_loss` oraz `spherical_kmeans` z separacją grup wag w optymalizatorze (`weight_decay=0.0` dla wektorów jednostkowych prototypów).
- **Zestaw testów `tests/test_smpc_architecture.py`:**
  - Opracowano 7 testów jednostkowych weryfikujących: Channel-First/Last, Option A numeryczną ekwiwalencję ($<10^{-6}$), niezmiennik 1,206,756 parametrów, ArcMargin loss i przepływ gradientów, ortogonalność podcentrów i hybrydową fuzję decyzyjną (100% sukcesu).

---

## 4. Zestawienie Wyników Ewaluacji Modeli

### Porównanie Metryki F1-Score na Niezależnym Zbiorze Testowym (117 próbek):

| Klasa Uszkodzenia | Stan Początkowy (TF/PyTorch Base) | **Stan Końcowy (Produkcyjny)** | Wzrost | Warunek ($\ge 0.70$) |
| :--- | :---: | :---: | :---: | :---: |
| **`beeping`** (stiction / zablokowane wrzeciono) | 0.2000 (20.0%) | **0.8636 (86.4%)** | **+66.4 p.p.** | **SPEŁNIONY** |
| **`clicking`** (uderzanie głowic / servo failure) | 0.7200 (72.0%) | **0.7848 (78.5%)** | **+6.5 p.p.** | **SPEŁNIONY** |
| **`scratching`** (tarcie głowic o talerze) | 0.0000 (0.0%) | **0.7368 (73.7%)** | **+73.7 p.p.** | **SPEŁNIONY** |
| **`normal`** (dysk sprawny) | 1.0000 (100.0%) | **1.0000 (100.0%)** | 0.0 p.p. | **SPEŁNIONY** |
| **Średnia Makro (Macro F1)** | **0.4800** | **0.8463** | **+36.6 p.p.** | **SPEŁNIONY** |
| **Dokładność ogólna (Accuracy)** | **~67.0%** | **83.76%** | **+16.8 p.p.** | — |

- **Luka generalizacyjna:** Train $\approx 85.8\%$, Test $83.76\%$ $\to$ **$2.04$ p.p.** (brak przeuczenia).

---

## 5. Architektura Silnika Hybrydowego (Hybrid Acoustic Sentinel)

```mermaid
flowchart TD
    RawAudio["Sygnał Audio z Mikrofonu\n(3.0s PCM 22050 Hz Mono)"] --> RMS{"Bramka Energii RMS\nRMS < 0.002 ?"}
    RMS -- "Tak" --> Silence["Cisza / Martwy sygnał\n(Odrzucenie ramki)"]
    RMS -- "Nie" --> VAD{"Silero VAD\nGłos ludzki w tle?"}
    VAD -- "Tak" --> VoiceReject["Zanieczyszczenie mową\n(Odrzucenie)"]
    VAD -- "Nie" --> DSPGuard{"Strażnik DSP\nClipping MEMS / BPF Wentylator?"}
    DSPGuard -- "Wykryto fałsz" --> FanSuppression["Tłumienie do AMBIENT\n(Ochrona przed szumem PC)"]
    DSPGuard -- "Czysty sygnał" --> Crest{"Analiza Transjentów\nCrest Factor >= 4.5 ?"}
    Crest -- "Tak" --> Transients["Wzmocnienie cech uderzeń\n(Kanał Delta - Clicking)"]
    Crest -- "Nie" --> RPM{"Analizator Harmonicznych\nRotacja 90 / 120 Hz ?"}
    Transients --> CNN["Inferencja 2D CNN (ONNX INT8)\nLog-Mel + Delta + Delta-Delta"]
    RPM --> CNN
    CNN --> OpenSet{"Bramka Open-Set OOD\nOdległość od prototypów?"}
    OpenSet -- "Obcy dźwięk" --> AmbientState["Stan: AMBIENT\n(Szum otoczenia / brak dysku)"]
    OpenSet -- "Znany profil" --> Decision{"Pewność predykcji\nConfidence >= 50% ?"}
    Decision -- "Nie (Entropia)" --> CheckRPM{"Czy wykryto rotację RPM?"}
    CheckRPM -- "Tak" --> NormalRPM["Dysk Sprawny (Normal)\nPotwierdzone obroty silnika"]
    CheckRPM -- "Nie" --> AmbientLow["Stan: AMBIENT\nBrak pewnych sygnatur"]
    Decision -- "Tak" --> OutputFinal["Ostateczna Klasyfikacja\n(Normal / Clicking / Scratching / Beeping)"]
```

---

## 6. Kompletna Mapa Plików Repozytorium

```
AcousticHddSentinel/
├── AGENTS.md                          # Instrukcja i niezmienniki techniczne dla agentów AI (SSOT)
├── README.md                          # Główna dokumentacja projektu z opisem architektury i quickstartem
├── kontekst.md                        # Niniejsze całościowe kompendium rozwoju projektu
├── fineTuning.md                      # Szczegółowa dokumentacja prac optymalizacyjnych i Acoustic Forensics
├── NOTES.md                           # Notatki badawcze dotyczące Seed Ensembling
├── PROMPT_DOKUMENTACJA_DLA_PROWADZACEGO.md # Skompresowana dokumentacja inżynierska i głęboki audyt
├── requirements.txt                   # Zależności środowiska Python (PyTorch, Librosa, Scikit-learn, ONNX)
├── .gitignore                         # Wykluczenia git (pliki .npy, artefakty Gradle, modele tymczasowe)
│
├── ml/                                # 🧠 Pipeline uczenia maszynowego (PyTorch 2D CNN, DSP, eksport)
│   ├── model_cnn.py                   # Architektura 2D CNN z buforami OOD Prototype Rejection i FusedEnsemble
│   ├── dataset_pytorch.py             # PyTorch Dataset z preloadem RAM i augmentacją SpecAugment
│   ├── hdd_audio_encoder.py           # Enkoder audio do tensorów .npy zgodny z ADR-001
│   ├── prepare_audio_dataset.py       # Kategoryzator bazy audio i generator próbek 'normal'
│   ├── train_cnn.py                   # Pipeline treningowy w PyTorch (AdamW, Label Smoothing, ROCm HIP)
│   ├── train_ensemble.py              # Pipeline Seed Ensembling (komitet modeli o ziarnach 42, 123, 777)
│   ├── export_model.py                # Eksport do formatów ONNX (FP32/INT8), TorchScript z auto-sync do Androida
│   ├── compare_ensembles.py           # Porównanie komitetu z pojedynczym modelem bazowym
│   ├── rpm_harmonics_analyzer.py      # Analizator harmonicznych rotacji silnika wrzeciona (5400/7200 RPM)
│   ├── dsp_adversarial_guard.py       # Strażnik adwersarialny (detekcja szumu wentylatorów PC i clippingu)
│   ├── hybrid_acoustic_sentinel.py    # Hybrydowy potok łączący DSP, VAD, RPM i CNN
│   └── hybrid_sentinel_engine.py      # Silnik inferencyjny zintegrowany ze strażnikami fizycznymi
│
├── tests/                             # 🧪 Testy jednostkowe, integracyjne, weryfikacyjne i stress-testy
│   ├── test_project_logic.py          # Automatyczny test integralności projektu i niezmienników ADR-001 (6/6)
│   ├── test_smpc_architecture.py      # Testy jednostkowe architektury Sub-Center Multi-Prototype Cosine (SMPC Option A, 7/7)
│   ├── test_rpm_harmonics_analyzer.py # Test jednostkowy detektora prążków obrotowych 90/120 Hz i wibrato mowy (9/9)
│   ├── test_computer_sound_rejection.py # Weryfikacja odrzucania szumów wentylatorów PC i cichego sygnału 45 dB (6/6)
│   ├── verify_cnn_model.py            # 6-etapowy test weryfikacyjny poprawności modelu PyTorch
│   ├── verify_binary_verdict_and_real_drives.py # Test 320 okien rzeczywistych dysków i wentylatorów
│   ├── verify_dsp_parity_python_java.py # Weryfikacja parzystości numerycznej DSP (Java vs Python)
│   ├── profile_normalization_quality.py # Profiler jakości dwuetapowej normalizacji DSP
│   ├── stress_test.py                 # Ciągły test inferencji potoku DSP + CNN (latencja, pamięć, stabilność)
│   ├── stress_test_vram.py            # Test obciążeniowy pamięci VRAM karty GPU AMD (2149 chunków/s)
│   ├── testEpok.py                    # Diagnostyka stabilności wag w zależności od liczby epok
│   ├── test_onnx_ood_rejection.py     # Test weryfikacyjny tłumienia dźwięków spoza rozkładu (OOD)
│   ├── run_toshiba_4min_study.py      # Skrypt ewaluacyjny 4-fazowego testu Toshiba P300 (6TB)
│   ├── run_wd_4min_study.py           # Skrypt ewaluacyjny 4-fazowego testu WD Blue (500GB)
│   ├── run_seagate_4min_study.py      # Skrypt ewaluacyjny 4-fazowego testu Seagate Momentus (500GB)
│   ├── run_hgst_4min_study.py         # Skrypt ewaluacyjny 4-fazowego testu HGST Travelstar (500GB)
│   ├── adversarial_lab/               # Laboratorium testów ataków adwersarialnych i szumów MEMS
│   └── recordings_4min_study/         # Zapis 16 minut rzeczywistych nagrań audio 4 dysków w 4 fazach
│
├── scripts/                           # 🛠️ Narzędzia i most orkiestracji lokalnych modeli LLM
│   ├── local_llm_bridge.py            # Tandem LLM: Qwen 3.8 27B (Architekt) + Qwen 2.5 Coder 14B (Coder) z 7 presetami
│   ├── start_services.sh              # Uruchamianie usług Ollama i Open WebUI
│   └── stop_services.sh               # Zatrzymywanie usług Ollama i Open WebUI
│
├── android/                           # 📱 Natywna aplikacja mobilna Android (Czysta Java 17, Material 3)
│   ├── app/src/main/assets/           # Skwantyzowany model ONNX INT8 (1.17 MB), classes.json, model_card.json
│   ├── app/src/main/java/             # Pakiety: audio (AudioRecord), data (SQLiteOpenHelper), dsp (ADR-001 DSP), ml (ONNX), ui (MVVM)
│   ├── app/src/main/res/              # Layouty UI w stylu Shazam z pulsującym radarem, ikony, motywy M3
│   ├── app/src/test/java/             # Zestaw 8 klas testowych (37 testów jednostkowych: DSP, Decymacja, RPM, Strażnik fizyczny, Audio, Baza SQLite, Klasyfikacja)
│   ├── build.gradle.kts               # Konfiguracja Gradle (Java 17, ONNX Runtime Mobile, AndroidX, 0 Kotlina)
│   └── README.md                      # Instrukcja uruchomienia i testowania w Android Studio
│
├── data/raw/                          # Baza 353 surowych nagrań laboratoryjnych (Ground Truth)
│   ├── northwind/                     # 82 pliki laboratoryjne Northwind Data Recovery
│   ├── additional/                    # 13 nagrań uszkodzeń HDD
│   ├── aesonlabs/                     # 9 nagrań Aesonlabs Data Recovery
│   ├── archive_hdd/                   # Nagrania z Internet Archive
│   ├── freesound_hdd/                 # Nagrania z Freesound
│   ├── independent_web_hdd/           # Nagrania z niezależnych serwisów napraw
│   ├── verified_real/                 # Próbki z fizycznych dysków testowych
│   ├── computer_ambient/              # Szumy wentylatorów i otoczenia komputerowego
│   └── PROCESSED_SOURCES_REGISTRY.json# Rejestr unikalnych próbek i sum kontrolnych MD5
│
├── dataset_categorized/               # Nagrania źródłowe podzielone na 4 klasy
├── dataset_encoded/                   # 6 840 tensorów .npy o kształcie (128, 259, 3) float32 (Train: 6045, Val: 279, Test: 516)
│
├── training_output/                   # Wagi referencyjne, metryki, raporty i macierz pomyłek
│   ├── best_hdd_cnn_model.pt          # Wagi najlepszego modelu PyTorch (seed 777)
│   ├── fused_ensemble_model.pt        # Wagi komitetu modeli
│   ├── classification_report.txt      # Raport Precision, Recall, F1
│   ├── test_evaluation_summary.json   # Metryki testowe w formacie JSON
│   ├── confusion_matrix.png           # Wykres macierzy pomyłek
│   └── classes.json                   # Mapowanie etykiet klas
│
├── exported_models/                   # Modele wdrożeniowe (ONNX FP32, ONNX INT8, TorchScript)
│
└── docx/                              # Dokumentacja techniczna i specyfikacje
    ├── adr/
    │   ├── adr.md                     # ADR-001: Architektura DSP i niezmienniki techniczne
    │   └── adr-002-mobile-architecture.md # ADR-002: Architektura aplikacji mobilnej Android
    ├── RAPORT_TESTOW_RZECZYWISTYCH.md # Raport z badań 4 fizycznych dysków HDD (16 minut audio, SMART)
    ├── diagramOverView.svg            # Schemat architektury systemu
    └── specs.txt                      # Specyfikacja techniczna wymagań
```

---

## 7. Status Integralności i Testów Automatycznych

1. **Test logiczny projektu (`python tests/test_project_logic.py`):**
   - [TEST 1/6] Składnia AST skryptów Python: **ZDANY (8 kluczowych modułów)**
   - [TEST 2/6] Brak Data Leakage (Train: 496 nagrań, Val: 52 nagrań, Test: 74 nagrań — łącznie 622 unikalne stems): **ZDANY (Zero przecieków)**
   - [TEST 3/6] Wymiary tensorów: **ZDANY (6 840 tensorów o idealnym kształcie (128, 259, 3) float32)**
   - [TEST 4/6] Weryfikacja bazy audio: **ZDANY (353 prawidłowe pliki w data/raw/)**
   - [TEST 5/6] Artefakty treningowe: **ZDANY (Model 4.63 MB, 4 klasy, classes.json)**
   - [TEST 6/6] Moduł mobilny Android: **ZDANY (27/27 kluczowych plików, model INT8 1.18 MB)**
   - **Wynik ogólny: 6/6 ZDANYCH (100% sukcesu)**

2. **Testy jednostkowe modułu Android (`cd android && ./gradlew test`):**
   - `AudioChunkTest`: walidacja bufora PCM i konwersji próbek — **ZDANY (1 test)**
   - `AudioRecorderDecimationTest`: kaskadowy filtr Butterwortha 4. rzędu i decymacja 2:1 (passband/stopband) — **ZDANY (5 testów)**
   - `AudioPreprocessorTest`: walidacja FFT, okna Hanna, banku Mel, Savitzky-Golay i Z-Score — **ZDANY (8 testów)**
   - `RpmHarmonicsAnalyzerTest`: weryfikacja prążków 90 Hz, 120 Hz, odrzucanie wibrato mowy 90/120/130 Hz, pisku stojana i 2-tone motor — **ZDANY (9 testów)**
   - `DspEndToEndStressTest`: 100 powtórzeń potoku bez wycieków pamięci i NaN — **ZDANY (3 testy)**
   - `DspPhysicalGuardTest`: strażnik fizyczny, stall beep, veto clippingu, fan noise vs scratching — **ZDANY (6 testów)**
   - `DiagnosticModelTest`: operacje CRUD bazy SQLiteOpenHelper i model diagnostyczny — **ZDANY (1 test)**
   - `ClassificationResultTest`: generowanie zaleceń inżynierskich i klasyfikacji — **ZDANY (4 testy)**
   - **Wynik ogólny: BUILD SUCCESSFUL (37/37 testów jednostkowych w 8 klasach zdanych w 2 s)**

3. **Testy architektury wielomodalnej SMPC (`python tests/test_smpc_architecture.py`):**
   - Weryfikacja układów Channel-First (B, 3, 128, 259) i Channel-Last (B, 128, 259, 3) — **ZDANY**
   - Option A ekwiwalencja numeryczna ONNX (MatMul + view + ReduceMax) vs definicja referencyjna ($< 10^{-6}$) — **ZDANY**
   - Niezmiennik liczby parametrów bazowych (dokładnie 1,206,756 parametrów przed i po SMPC) — **ZDANY**
   - SubCenterArcMarginLoss i przepływ gradientów w PyTorch — **ZDANY**
   - compute_orthogonality_loss (kara 0.0 dla wektorów ortogonalnych, 2.10 dla skolapsowanych) — **ZDANY**
   - spherical_kmeans klastrowanie M podcentrów na hipersferze S^127 — **ZDANY**
   - Weryfikacja hybrydowej Gałęzi D (walidacja cech scratching vs ambient noise) — **ZDANY**
   - **Wynik ogólny: 7/7 ZDANYCH (100% sukcesu)**

4. **Remediacja Milestone 1 Iteracja 3 (Temporal Pitch Stationarity & Zero FPR):**
   - **Weryfikacja stacjonarności czasowej pitch (Python & Java 17):** Implementacja testu niespójnych przesunięć czasowych (incommensurate offsets: 0.0s, 0.37s, 0.83s, 1.41s, 2.11s) z interpolacją paraboliczną częstotliwości piku w pasmie +/- 5.0 Hz. Sztywny silnik HDD zachowuje $\Delta f_0 \le 0.1$ Hz, podczas gdy wibrato głosu ludzkiego (3-8 Hz FM) wykazuje $\Delta f_0 \ge 3.0$ Hz, co eliminuje fałszywe alarmy rotacji w 100%.
   - **Bariera ostrości prążka (`f0_sharpness`):** Obliczanie stosunku energii piku do ramion (+/- 2 biny widma). Silniki mechaniczne wykazują ostrość $> 10.0$, podczas gdy szerokie wybrzuszenia modulacji głosu mają ostrość $< 6.0$. Wymóg `f0_sharpness >= 6.0` dla 2-tonowych rotorów FDB całkowicie odcina podszywanie się pod rotację.
   - **Zaokrąglenia binów i symetria tła w Javie:** Poprawiono zaokrąglanie granic pasma `(int) Math.ceil` / `(int) Math.floor` w `RpmHarmonicsAnalyzer.java` oraz wyrównano okno szacowania tła lokalnego do `targetF0 \pm 25.0` Hz.
   - **Adversarial Audit:** 0% fałszywych alarmów (0/96 w Pythonie, 0/54 w Javie) na pełnej siatce sweepów wokalnych, syren i tonów czystych.

---

## 8. Kluczowe Polecenia Szybkiego Startu

> **Uwaga dotyczące środowiska:** Do uruchamiania skryptów PyTorch/DSP projekt posiada gotowy wirtualny interpreter w `.venv` (automatycznie wykrywany lub poprzez `source .venv/bin/activate`).

```bash
# 1. Kompletny zestaw 5 oficjalnych testów systemu:
python tests/test_project_logic.py
python tests/test_smpc_architecture.py
python tests/test_rpm_harmonics_analyzer.py
python tests/test_computer_sound_rejection.py
cd android && ./gradlew test && cd ..
python tests/verify_cnn_model.py
python tests/test_rpm_harmonics_analyzer.py
python tests/test_computer_sound_rejection.py
python tests/test_smpc_architecture.py

# 2. Uruchomienie testów jednostkowych Androida (czysta Java 17):
cd android && ./gradlew test && cd ..

# 3. Kategoryzacja i enkodowanie tensorów audio:
python ml/prepare_audio_dataset.py
python ml/hdd_audio_encoder.py

# 4. Trening modelu bazowego PyTorch na GPU (AMD ROCm) z opcjonalnym SMPC:
python ml/train_cnn.py --epochs 35 --batch-size 32 --output-dir ./training_output
python ml/train_cnn.py --use-smpc --smpc-subcenters 3 --epochs 35 --batch-size 32 --output-dir ./training_output_smpc

# 5. Eksport modeli ONNX FP32, ONNX INT8 i synchronizacja do Androida:
python ml/export_model.py --weights training_output/best_hdd_cnn_model.pt --output-dir exported_models

# 6. Uruchomienie ewaluacji fizycznego badania 4 dysków HDD:
python tests/run_wd_4min_study.py
python tests/run_seagate_4min_study.py
python tests/run_hgst_4min_study.py
python tests/run_toshiba_4min_study.py
```
