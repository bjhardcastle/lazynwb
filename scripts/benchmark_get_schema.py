from wcmatch.glob import Z
import threading
import time

import lazynwb
from polars import Schema, Array, List, Boolean, Float32, Float64, Int64, String

# These benchmark paths all live in the same public S3 bucket.
# Providing the region avoids an extra probe on first access.
lazynwb.config.fsspec_storage_options = {"region": "us-west-2"}
lazynwb.config.anon = True
lazynwb.config.use_obstore = False
lazynwb.config.use_remfile = True

correct_schema = Schema(
    [
        ("activity_drift", Float64),
        ("amplitude", Float64),
        ("amplitude_cutoff", Float64),
        ("amplitude_cv_median", Float64),
        ("amplitude_cv_range", Float64),
        ("amplitude_median", Float64),
        ("ccf_ap", Float64),
        ("ccf_dv", Float64),
        ("ccf_ml", Float64),
        ("channels", List(Int64)),
        ("cluster_id", Int64),
        ("d_prime", Float64),
        ("decoder_label", String),
        ("decoder_probability", Float64),
        ("default_qc", Boolean),
        ("device_name", String),
        ("drift_mad", Float64),
        ("drift_ptp", Float64),
        ("drift_std", Float64),
        ("electrode_group", String),
        ("electrode_group_name", String),
        ("electrodes", List(Int64)),
        ("exp_decay", Float64),
        ("firing_range", Float64),
        ("firing_rate", Float64),
        ("half_width", Float64),
        ("id", Int64),
        ("is_not_drift", Boolean),
        ("isi_violations_count", Float64),
        ("isi_violations_ratio", Float64),
        ("isolation_distance", Float64),
        ("l_ratio", Float64),
        ("location", String),
        ("nn_hit_rate", Float64),
        ("nn_miss_rate", Float64),
        ("num_negative_peaks", Int64),
        ("num_positive_peaks", Int64),
        ("num_spikes", Float64),
        ("obs_intervals", List(Array(Float64, shape=(2,)))),
        ("peak_channel", Int64),
        ("peak_electrode", Int64),
        ("peak_to_valley", Float64),
        ("peak_trough_ratio", Float64),
        ("peak_waveform_index", Int64),
        ("presence_ratio", Float64),
        ("recovery_slope", Float64),
        ("repolarization_slope", Float64),
        ("rp_contamination", Float64),
        ("rp_violations", Float64),
        ("silhouette", Float64),
        ("sliding_rp_violation", Float64),
        ("snr", Float64),
        ("spike_amplitudes", List(Float32)),
        ("spike_times", List(Float64)),
        ("spread", Float64),
        ("structure", String),
        ("sync_spike_2", Float64),
        ("sync_spike_4", Float64),
        ("sync_spike_8", Float64),
        ("unit_id", String),
        ("velocity_above", Float64),
        ("velocity_below", Float64),
        ("waveform_mean", Array(Float64, shape=(210, 384))),
        ("waveform_sd", Array(Float64, shape=(210, 384))),
    ]
)
paths = [
    "s3://aind-scratch-data/dynamic-routing/cache/nwb/v0.0.272/620263_2022-07-26.nwb",
    "s3://aind-scratch-data/dynamic-routing/cache/nwb/v0.0.277/620263_2022-07-26.nwb",
]


def print_heartbeat(
    label: str,
    stop_event: threading.Event,
    t0: float,
    interval_seconds: float = 10.0,
) -> None:
    while not stop_event.wait(interval_seconds):
        print(
            f"{label} still running after {time.time() - t0:.1f}s...",
            flush=True,
        )


files_to_check = paths[:2]
print(
    f"Benchmarking get_table_schema on {len(files_to_check)} files in table 'units'...",
    flush=True,
)
print(
    f"obstore config: anon={lazynwb.config.anon}, region={lazynwb.config.fsspec_storage_options.get('region')!r}",
    flush=True,
)
for path in files_to_check:
    print(f"  - {path}", flush=True)

t0 = time.time()
stop_event = threading.Event()
heartbeat = threading.Thread(
    target=print_heartbeat,
    args=("get_table_schema", stop_event, t0),
    daemon=True,
)
heartbeat.start()
try:
    schema = lazynwb.get_table_schema(
        file_paths=files_to_check,
        table_path="units",
        first_n_files_to_infer_schema=None,
        exclude_array_columns=False,
        exclude_internal_columns=True,
        raise_on_missing=False,
    )
finally:
    stop_event.set()
    heartbeat.join(timeout=0.1)

assert (
    schema == correct_schema
), f"Incorrect schema returned from `get_table_schema()`: {schema}"
print(f"Correct schema returned. Time elapsed: {time.time() - t0:.2f}s", flush=True)
