#!/usr/bin/env bash
# ==============================================================================
# Skrypt instalacji aplikacji mobilnej Acoustic HDD Sentinel na telefonie Android
# Single Source of Truth: ADR-002, AGENTS.md
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
APK_PATH="${PROJECT_ROOT}/android/app/build/outputs/apk/debug/app-debug.apk"
PACKAGE_NAME="com.acoustichddsentinel"
MAIN_ACTIVITY="${PACKAGE_NAME}/.ui.MainActivity"

# Kolory do komunikatów terminala
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}   INSTALACJA APLIKACJI ACOUSTIC HDD SENTINEL NA SMARTFONIE ANDROID   ${NC}"
echo -e "${BLUE}======================================================================${NC}"

# 1. Lokalizacja narzędzia ADB
ADB_CMD="adb"
if ! command -v adb &> /dev/null; then
    if [ -x "${HOME}/Android/Sdk/platform-tools/adb" ]; then
        ADB_CMD="${HOME}/Android/Sdk/platform-tools/adb"
    else
        echo -e "${RED}[BŁĄD] Narzędzie adb nie zostało znalezione w PATH ani w ~/Android/Sdk/platform-tools/adb!${NC}"
        echo -e "Zainstaluj Android platform-tools lub dodaj ścieżkę do zmiennej PATH."
        exit 1
    fi
fi
echo -e "-> Narzędzie ADB: ${GREEN}${ADB_CMD}${NC}"

# 2. Sprawdzenie podłączonych urządzeń
DEVICE_COUNT=$("${ADB_CMD}" devices | grep -v "List" | grep -w "device" | wc -l)
if [ "${DEVICE_COUNT}" -eq 0 ]; then
    echo -e "${RED}[BŁĄD] Nie wykryto podłączonego urządzenia Android z włączonym debugowaniem USB!${NC}"
    echo -e "${YELLOW}Kroki rozwiązywania problemu:${NC}"
    echo -e " 1. Podłącz telefon kablem USB do komputera."
    echo -e " 2. W telefonie upewnij się, że Opcje Programisty i 'Debugowanie USB' są WŁĄCZONE."
    echo -e " 3. Sprawdź ekran telefonu i zaakceptuj monit 'Zezwalaj na debugowanie USB z tego komputera'."
    echo -e "\nAktualna lista urządzeń:"
    "${ADB_CMD}" devices -l
    exit 1
fi

DEVICE_INFO=$("${ADB_CMD}" devices -l | grep -w "device" | head -n 1)
echo -e "-> Wykryto urządzenie: ${GREEN}${DEVICE_INFO}${NC}"

# 3. Sprawdzenie lub przebudowa pliku APK
if [ ! -f "${APK_PATH}" ] || [ "$1" == "--build" ] || [ "$1" == "-b" ]; then
    echo -e "\n${YELLOW}-> Budowanie najnowszej wersji APK za pomocą Gradle...${NC}"
    cd "${PROJECT_ROOT}/android"
    ./gradlew assembleDebug
    cd "${PROJECT_ROOT}"
fi

if [ ! -f "${APK_PATH}" ]; then
    echo -e "${RED}[BŁĄD] Nie znaleziono pliku APK: ${APK_PATH}${NC}"
    exit 1
fi

APK_SIZE=$(du -h "${APK_PATH}" | cut -f1)
echo -e "-> Plik APK gotowy: ${GREEN}${APK_PATH}${NC} (${APK_SIZE})"

# 4. Instalacja aplikacji
echo -e "\n${YELLOW}-> Przesyłanie i instalowanie pakietu na telefonie...${NC}"
"${ADB_CMD}" install -r "${APK_PATH}"
echo -e "${GREEN}✓ Aplikacja zainstalowana pomyślnie!${NC}"

# 5. Nadanie uprawnień do mikrofonu (dla wygody testów deweloperskich)
echo -e "-> Nadawanie uprawnienia RECORD_AUDIO..."
"${ADB_CMD}" shell pm grant "${PACKAGE_NAME}" android.permission.RECORD_AUDIO 2>/dev/null || true

# 6. Uruchomienie aplikacji na ekranie smartfona
echo -e "-> Uruchamianie aplikacji..."
"${ADB_CMD}" shell am start -n "${MAIN_ACTIVITY}" > /dev/null

# 7. Odczyt diagnostyczny z Logcat
sleep 1
echo -e "\n${BLUE}--- Diagnostyka startowa silnika AI (Logcat) ---${NC}"
"${ADB_CMD}" logcat -d -s "AcousticClassifier:*" "MainViewModel:*" | tail -n 8

echo -e "\n${GREEN}======================================================================${NC}"
echo -e "${GREEN}🎉 SUKCES: Aplikacja Acoustic HDD Sentinel działa na Twoim telefonie!${NC}"
echo -e "${GREEN}======================================================================${NC}"
