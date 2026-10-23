
from pathlib import Path
from datetime import datetime
import threading
import time

import av
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import (
    AudioProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

# ==================================================
# 1. PATHS AND SETTINGS
# ==================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_PATH = DATA_DIR / "noise_data.csv"
ARTIFACT_PATH = ROOT / "artifacts" / "noise_kmeans_model.joblib"
MICROPHONE_CSV = DATA_DIR / "live_microphone_readings.csv"

RECORD_INTERVAL_SECONDS = 5

st.set_page_config(
    page_title="Noise Pollution Monitor",
    page_icon="🔊",
    layout="wide",
)

st.title("🔊 Noise Pollution Monitor")
st.caption(
    "Noise monitoring dashboard | K-Means model | "
    "Live microphone recording"
)

# ==================================================
# 2. CLASSIFICATION
# ==================================================

def classify_zone(sound_level_db):
    """Classify a calibrated environmental sound level in dB SPL."""
    if sound_level_db < 44:
        return "Quiet Zone"
    elif sound_level_db <= 60:
        return "Moderate Zone"
    return "Noisy Zone"


# ==================================================
# 3. PRESERVE AND LOAD THE K-MEANS ARTIFACT
# ==================================================

@st.cache_resource
def load_kmeans_artifact():
    if not ARTIFACT_PATH.exists():
        return None

    return joblib.load(ARTIFACT_PATH)


try:
    kmeans_artifact = load_kmeans_artifact()
except Exception as exc:
    kmeans_artifact = None
    st.warning(
        "The saved K-Means artifact could not be loaded. "
        "Threshold-based noise classification and microphone "
        f"recording can still work. Details: {exc}"
    )

if kmeans_artifact is not None:
    st.caption("Saved K-Means model found. The original model file is unchanged.")
else:
    st.caption(
        "K-Means artifact not found or unavailable. "
        "Threshold-based classification remains available."
    )


# ==================================================
# 4. LOAD EXISTING MEASUREMENTS
# ==================================================

@st.cache_data
def load_existing_data():
    if not DATA_PATH.exists():
        return pd.DataFrame(
            columns=["timestamp", "location", "sound_level_db", "category"]
        )

    df = pd.read_csv(DATA_PATH)

    required = {"timestamp", "location", "sound_level_db"}
    if not required.issubset(df.columns):
        return pd.DataFrame(
            columns=["timestamp", "location", "sound_level_db", "category"]
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["location"] = df["location"].astype("string").str.strip()
    df["sound_level_db"] = pd.to_numeric(
        df["sound_level_db"], errors="coerce"
    )

    df = df.dropna(
        subset=["timestamp", "location", "sound_level_db"]
    ).copy()
    df = df[df["location"] != ""]

    # Fixed zone thresholds for calibrated dB SPL data.
    df["category"] = df["sound_level_db"].apply(classify_zone)

    return df


data = load_existing_data()


# ==================================================
# 5. MICROPHONE HISTORY STORAGE
# ==================================================

MIC_COLUMNS = [
    "timestamp",
    "location",
    "measured_level",
    "unit",
    "digital_level_dbfs",
    "calibration_offset_db",
    "zone",
]


def save_microphone_reading(
    timestamp,
    location,
    dbfs,
    calibration_offset,
):
    """
    Append one five-second audio-window reading.

    dBFS remains a digital measurement. An SPL value is produced
    only when the user enables calibrated measurement and provides
    a calibration offset obtained using a suitable reference meter.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if calibration_offset is None:
        measured_level = dbfs
        unit = "dBFS"
        zone = "Uncalibrated"
        offset_to_save = ""
    else:
        measured_level = dbfs + calibration_offset
        unit = "dB SPL (calibrated)"
        zone = classify_zone(measured_level)
        offset_to_save = calibration_offset

    row = pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "location": location,
                "measured_level": round(float(measured_level), 2),
                "unit": unit,
                "digital_level_dbfs": round(float(dbfs), 2),
                "calibration_offset_db": offset_to_save,
                "zone": zone,
            }
        ],
        columns=MIC_COLUMNS,
    )

    write_header = not MICROPHONE_CSV.exists()

    row.to_csv(
        MICROPHONE_CSV,
        mode="a",
        header=write_header,
        index=False,
        encoding="utf-8",
    )


@st.cache_data
def load_microphone_history():
    if not MICROPHONE_CSV.exists():
        return pd.DataFrame(columns=MIC_COLUMNS)

    try:
        history = pd.read_csv(MICROPHONE_CSV)

        for column in MIC_COLUMNS:
            if column not in history.columns:
                history[column] = ""

        return history[MIC_COLUMNS].copy()

    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame(columns=MIC_COLUMNS)


# ==================================================
# 6. LIVE AUDIO PROCESSOR
# ==================================================

class LiveAudioProcessor(AudioProcessorBase):
    """
    Collect audio samples and calculate an RMS digital level.

    A snapshot resets the accumulation window, allowing the app
    to save approximately one measurement per five-second window.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.sum_squares = 0.0
        self.sample_count = 0
        self.peak = 0.0
        self.last_dbfs = None

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray()

        if audio.size:
            if np.issubdtype(audio.dtype, np.integer):
                info = np.iinfo(audio.dtype)
                scale = float(max(abs(info.min), abs(info.max)))
                samples = audio.astype(np.float32) / scale
            else:
                samples = audio.astype(np.float32)

            samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)

            square_sum = float(np.sum(samples ** 2))
            count = int(samples.size)
            peak = float(np.max(np.abs(samples)))

            with self.lock:
                self.sum_squares += square_sum
                self.sample_count += count
                self.peak = max(self.peak, peak)

                rms = np.sqrt(
                    self.sum_squares / max(self.sample_count, 1)
                )
                self.last_dbfs = float(
                    20 * np.log10(max(float(rms), 1e-8))
                )

        # Return the original frame so WebRTC can continue processing.
        return frame

    def read_window(self):
        """Return and reset the accumulated window's RMS and peak levels."""
        with self.lock:
            if self.sample_count == 0:
                return None, None

            rms = np.sqrt(
                self.sum_squares / max(self.sample_count, 1)
            )

            dbfs = float(20 * np.log10(max(float(rms), 1e-8)))
            peak_dbfs = float(
                20 * np.log10(max(self.peak, 1e-8))
            )

            self.sum_squares = 0.0
            self.sample_count = 0
            self.peak = 0.0

            return dbfs, peak_dbfs

    def current_level(self):
        with self.lock:
            return self.last_dbfs


# ==================================================
# 7. MAIN TABS
# ==================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Dashboard",
        "🔍 Predict a Zone",
        "📁 Upload Data",
        "🎙️ Live Microphone",
        "🕘 Microphone History",
    ]
)


# ==================================================
# 8. DASHBOARD
# ==================================================

with tab1:
    st.subheader("Noise Monitoring Dashboard")

    if not data.empty:
        col1, col2, col3 = st.columns(3)

        col1.metric("Total Readings", f"{len(data):,}")
        col2.metric("Locations", str(data["location"].nunique()))
        col3.metric(
            "Average Sound Level",
            f"{data['sound_level_db'].mean():.1f} dB",
        )

        st.subheader("Average Sound Level by Location")
        st.bar_chart(
            data.groupby("location")["sound_level_db"]
            .mean()
            .sort_values()
        )

        st.subheader("Noise Zone Distribution")
        zone_order = ["Quiet Zone", "Moderate Zone", "Noisy Zone"]
        zone_counts = (
            data["category"]
            .value_counts()
            .reindex(zone_order, fill_value=0)
        )
        st.bar_chart(zone_counts)

        st.subheader("Recent Measurements")
        st.dataframe(
            data[
                ["timestamp", "location", "sound_level_db", "category"]
            ].sort_values("timestamp", ascending=False),
            use_container_width=True,
        )

        st.download_button(
            "Download Categorized Measurements",
            data=data.to_csv(index=False).encode("utf-8"),
            file_name="noise_readings_with_zones.csv",
            mime="text/csv",
        )
    else:
        st.info(
            "No valid measurements found in data/noise_data.csv. "
            "Upload a CSV to get started."
        )

    st.caption(
        "The displayed categories use fixed thresholds. "
        "Environmental classification assumes measurements are in calibrated dB SPL."
    )


# ==================================================
# 9. MANUAL PREDICTION
# ==================================================

with tab2:
    st.subheader("Classify a Sound-Level Reading")

    existing_locations = (
        sorted(data["location"].astype(str).unique().tolist())
        if not data.empty
        else []
    )

    location_options = existing_locations + ["Other / new location"]

    selected_location = st.selectbox(
        "Measurement Location",
        location_options,
        key="prediction_location",
    )

    if selected_location == "Other / new location":
        location = st.text_input(
            "Enter location name",
            value="New Location",
            key="new_prediction_location",
        ).strip()
    else:
        location = selected_location

    sound_level = st.number_input(
        "Calibrated Sound Level (dB SPL)",
        min_value=0.0,
        max_value=200.0,
        value=50.0,
        step=1.0,
    )

    reading_date = st.date_input(
        "Measurement Date",
        value=pd.Timestamp.now().date(),
    )

    reading_time = st.time_input(
        "Measurement Time",
        value=pd.Timestamp.now().time().replace(
            second=0, microsecond=0
        ),
    )

    if st.button("Classify Noise Zone", type="primary"):
        if not location:
            st.error("Please enter a location.")
        else:
            zone = classify_zone(sound_level)
            timestamp = pd.Timestamp.combine(
                reading_date, reading_time
            )

            st.subheader(f"Result: {zone}")
            st.write(f"**Location:** {location}")
            st.write(f"**Sound level:** {sound_level:.1f} dB SPL")
            st.write(f"**Time:** {timestamp}")

            if zone == "Quiet Zone":
                st.success("Below 44 dB SPL.")
            elif zone == "Moderate Zone":
                st.info("From 44 to 60 dB SPL, inclusive.")
            else:
                st.error("Above 60 dB SPL.")

            st.caption(
                "Ensure the entered measurement is calibrated before "
                "interpreting it as an environmental sound level."
            )


# ==================================================
# 10. CSV UPLOAD
# ==================================================

with tab3:
    st.subheader("Upload Noise Measurements")

    st.write(
        "Required columns: timestamp, location, sound_level_db."
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        key="noise_csv_upload",
    )

    if uploaded_file is not None:
        try:
            uploaded_data = pd.read_csv(uploaded_file)

            required = {
                "timestamp",
                "location",
                "sound_level_db",
            }

            missing = required - set(uploaded_data.columns)

            if missing:
                st.error(
                    "Missing columns: " + ", ".join(sorted(missing))
                )
            else:
                uploaded_data["timestamp"] = pd.to_datetime(
                    uploaded_data["timestamp"], errors="coerce"
                )
                uploaded_data["location"] = (
                    uploaded_data["location"]
                    .astype("string")
                    .str.strip()
                )
                uploaded_data["sound_level_db"] = pd.to_numeric(
                    uploaded_data["sound_level_db"], errors="coerce"
                )

                uploaded_data = uploaded_data.dropna(
                    subset=[
                        "timestamp",
                        "location",
                        "sound_level_db",
                    ]
                ).copy()

                uploaded_data = uploaded_data[
                    uploaded_data["location"] != ""
                ]

                uploaded_data["category"] = (
                    uploaded_data["sound_level_db"].apply(classify_zone)
                )

                st.success(
                    f"Classified {len(uploaded_data):,} valid readings."
                )

                st.dataframe(
                    uploaded_data[
                        [
                            "timestamp",
                            "location",
                            "sound_level_db",
                            "category",
                        ]
                    ],
                    use_container_width=True,
                )

                st.download_button(
                    "Download Classified CSV",
                    data=uploaded_data.to_csv(index=False).encode("utf-8"),
                    file_name="uploaded_noise_zones.csv",
                    mime="text/csv",
                )

        except Exception as exc:
            st.error(f"Could not process the CSV: {exc}")


# ==================================================
# 11. LIVE MICROPHONE AND AUTOMATIC RECORDING
# ==================================================

with tab4:
    st.subheader("🎙️ Live Microphone Monitoring")

    st.info(
        "When the microphone is running, the app saves one reading "
        "every five seconds to data/live_microphone_readings.csv."
    )

    microphone_location = st.text_input(
        "Monitoring Location",
        value="Computer Microphone",
        key="microphone_location",
        help="For example: Radio Studio, Office, Classroom, or Workshop.",
    ).strip()

    calibrated_mode = st.checkbox(
        "I have calibrated this microphone against a suitable reference sound-level meter",
        value=False,
        key="microphone_calibrated",
    )

    calibration_offset = None

    if calibrated_mode:
        calibration_offset = st.number_input(
            "Calibration offset (dB): dB SPL = dBFS + offset",
            min_value=-200.0,
            max_value=200.0,
            value=100.0,
            step=0.1,
            key="microphone_calibration_offset",
            help=(
                "Enter the offset determined from a known reference measurement. "
                "Do not use the default value without performing calibration."
            ),
        )

        st.caption(
            "Only enable calibrated mode and enter an offset after "
            "performing a valid calibration. A single offset may not "
            "fully correct frequency response or microphone limitations."
        )

    st.warning(
        "Without calibration, the app stores dBFS readings with the zone "
        "'Uncalibrated'. Digital dBFS values are not environmental dB SPL."
    )

    @st.fragment(run_every=1)
    def live_microphone():
        context = webrtc_streamer(
            key="noise-monitor-live-microphone",
            mode=WebRtcMode.SENDONLY,
            audio_processor_factory=LiveAudioProcessor,
            media_stream_constraints={
                "audio": True,
                "video": False,
            },
            async_processing=True,
        )

        processor = context.audio_processor

        if not context.state.playing:
            st.session_state["mic_last_save_time"] = time.monotonic()
            st.session_state["mic_window_started"] = False
            st.info(
                "Start the microphone using the control above. "
                "Automatic recording runs only while it is active."
            )
            return

        if processor is None:
            st.info("Connecting to the microphone...")
            return

        if not st.session_state.get("mic_window_started", False):
            st.session_state["mic_last_save_time"] = time.monotonic()
            st.session_state["mic_window_started"] = True

        current_dbfs = processor.current_level()

        if current_dbfs is not None:
            st.metric(
                "Current Digital Audio Level",
                f"{current_dbfs:.1f} dBFS",
            )

        now = time.monotonic()
        last_save = st.session_state.get("mic_last_save_time", now)

        if now - last_save >= RECORD_INTERVAL_SECONDS:
            dbfs, peak_dbfs = processor.read_window()

            # Advance the interval even if no usable samples arrived.
            st.session_state["mic_last_save_time"] = now

            if dbfs is not None and microphone_location:
                timestamp = datetime.now().astimezone().isoformat(
                    timespec="seconds"
                )

                try:
                    save_microphone_reading(
                        timestamp=timestamp,
                        location=microphone_location,
                        dbfs=dbfs,
                        calibration_offset=calibration_offset,
                    )

                    if calibration_offset is None:
                        st.success(
                            f"Saved reading: {dbfs:.1f} dBFS "
                            "(zone Uncalibrated)"
                        )
                    else:
                        measured_spl = dbfs + calibration_offset
                        zone = classify_zone(measured_spl)
                        st.success(
                            f"Saved reading: {measured_spl:.1f} dB SPL — {zone}"
                        )

                    st.caption(
                        f"Five-second window peak: {peak_dbfs:.1f} dBFS. "
                        "The saved level is an RMS average for the audio "
                        "samples accumulated in that window."
                    )

                    # Clear the cached history so the History tab can reload it.
                    load_microphone_history.clear()

                except OSError as exc:
                    st.error(f"Could not save microphone reading: {exc}")

            elif not microphone_location:
                st.error("Enter a location to save microphone readings.")

    live_microphone()


# ==================================================
# 12. MICROPHONE HISTORY
# ==================================================

with tab5:
    st.subheader("🕘 Saved Microphone History")

    st.write(f"**CSV file:** `data/live_microphone_readings.csv`")

    history = load_microphone_history()

    if history.empty:
        st.info(
            "No microphone readings have been saved yet. "
            "Open Live Microphone, enter a location, and start recording."
        )
    else:
        total_readings = len(history)
        saved_locations = history["location"].nunique()

        col1, col2 = st.columns(2)
        col1.metric("Saved Microphone Readings", f"{total_readings:,}")
        col2.metric("Recorded Locations", str(saved_locations))

        st.dataframe(
            history.sort_values("timestamp", ascending=False),
            use_container_width=True,
        )

        st.download_button(
            "Download Microphone History",
            data=history.to_csv(index=False).encode("utf-8"),
            file_name="live_microphone_readings.csv",
            mime="text/csv",
        )

        calibrated_history = history[
            history["unit"] == "dB SPL (calibrated)"
        ].copy()

        if not calibrated_history.empty:
            calibrated_history["measured_level"] = pd.to_numeric(
                calibrated_history["measured_level"], errors="coerce"
            )

            st.subheader("Calibrated Noise Zone Distribution")
            st.bar_chart(
                calibrated_history["zone"].value_counts().reindex(
                    ["Quiet Zone", "Moderate Zone", "Noisy Zone"],
                    fill_value=0,
                )
            )

        st.caption(
            "Uncalibrated digital readings are retained for analysis but "
            "are not counted as Quiet, Moderate, or Noisy SPL classifications."
        )


# ==================================================
# 13. FOOTER
# ==================================================

st.divider()
st.caption(
    "Noise Pollution Monitor | Research and demonstration project. "
    "Calibrate measurements and validate the system before using it "
    "for environmental or regulatory decisions."
)
