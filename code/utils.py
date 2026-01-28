import json
import pandas as pd
from pynwb import NWBFile

from parse_raw_data import MetricsVrForaging

import pandas as pd
from typing import Literal


def extract_site_entry_exit(
    df: pd.DataFrame,
    site_col: str = "label",
    stop_col: str = "stop_time",
) -> pd.DataFrame:
    """
    Extract site entry and exit events

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame indexed by start time, with stop_time and site columns.
        Site index should be NaN when not at a site.
    site_col : str
        Column name for site indices.
    stop_col : str
        Column name for stop time.
    
    Returns
    -------
    pd.DataFrame
        Event table with columns:
        - timestamp
        - site
        - event ('site_entry' or 'site_exit')
    """

    events = []

    for start_time, row in df.iterrows():
        site = row[site_col]
        stop_time = row[stop_col]

        if pd.isna(site) or pd.isna(stop_time):
            continue

        # Entry event at start_time
        events.append({
            "timestamp": start_time,
            "site": site,
            "event": "site entry"
        })

        # Exit event at stop_time
        events.append({
            "timestamp": stop_time,
            "site": site,
            "event": "site exit"
        })

    event_table = pd.DataFrame(events)
    event_table = event_table.sort_values("timestamp").reset_index(drop=True)
    return event_table

def extract_patch_entry_exit(df: pd.DataFrame, patch_label_col:str="patch_label", 
    patch_onset_col:str="patch_onset"):
    """
    Extract patch entry and exit events from a state table.

    Assumes:
    - index is start_time
    - patch_label is the current patch
    - patch_onset is the start time of that patch visit (constant within patch)

    Returns:
    - event table with patch_entry and patch_exit events
    """
    df = df.copy()

    # Identify when the patch changes (or starts/ends)
    df["prev_patch"] = df[patch_label_col].shift(1)
    df["prev_onset"] = df[patch_onset_col].shift(1)

    events = []

    for ts, row in df.iterrows():
        patch = row[patch_label_col]
        onset = row[patch_onset_col]
        prev_patch = row["prev_patch"]
        prev_onset = row["prev_onset"]

        # Patch entry (first time we see a patch)
        if pd.isna(prev_patch) and pd.notna(patch):
            events.append({"timestamp": ts, "patch": patch, "event": "patch entry"})

        # Patch exit (patch disappears)
        elif pd.notna(prev_patch) and pd.isna(patch):
            events.append({"timestamp": ts, "patch": prev_patch, "event": "patch exit"})

        # Patch switch (exit previous + entry new)
        elif pd.notna(prev_patch) and pd.notna(patch) and patch != prev_patch:
            events.append({"timestamp": ts, "patch": prev_patch, "event": "patch exit"})
            events.append({"timestamp": ts, "patch": patch, "event": "patch entry"})

    return pd.DataFrame(events).sort_values("timestamp").reset_index(drop=True)

def extract_binary_events(df: pd.DataFrame, flag_col: str, time_col: str, event_name: str):
    """
    Extract discrete events from a snapshot-style table using a boolean flag.

    Rows represent task state at a given time, not events. An event is emitted
    for rows where `flag_col` is True. Event timestamps are taken from
    `time_col` when available, otherwise the row index is used.

    Parameters
    ----------
    df : pandas.DataFrame
        Snapshot/state table indexed by time.
    flag_col : str
        Boolean column indicating event occurrence (True = event).
    time_col : str
        Column containing precise event timestamps (may be NaN).
    event_name : str
        Name assigned to the extracted event.

    Returns
    -------
    pandas.DataFrame
        Event table with columns: `timestamp`, `event`.
    """
    events = []

    for ts, row in df.iterrows():
        if row.get(flag_col) is True:
            t = row.get(time_col)
            event_time = t if pd.notna(t) else ts
            events.append({"timestamp": event_time, "event": event_name})

    return (
        pd.DataFrame(events)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

def extract_odor_events(odor_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand odor onset/offset intervals into odor on/off events.

    Parameters
    ----------
    odor_df : pandas.DataFrame
        Contains ``odor_onset``, ``odor_offset``, and ``patch_type``.

    Returns
    -------
    pandas.DataFrame
        Event table with ``time``, ``event`` (``odor_on``/``odor_off``),
        and ``patch_type``.
    """
    onset_events = (
        odor_df[["odor_onset", "patch_type"]]
        .rename(columns={"odor_onset": "time"})
        .assign(event="odor onset")
    )

    offset_events = (
        odor_df[["odor_offset", "patch_type"]]
        .rename(columns={"odor_offset": "time"})
        .assign(event="odor offset")
    )

    events = (
        pd.concat([onset_events, offset_events], ignore_index=True)
        .sort_values("time")
        .reset_index(drop=True)
    )

    return events

def extract_lick_events(lick_series: pd.Series) -> pd.DataFrame:
    """
    Extract lick events from a boolean time series.

    Parameters
    ----------
    lick_series : pandas.Series
        Boolean series indexed by time; True indicates a lick.

    Returns
    -------
    pandas.DataFrame
        Event table with columns ``time`` and ``event`` (``lick``).
    """
    times = lick_series[lick_series].index

    return pd.DataFrame({
        "time": times,
        "event": "lick",
    })

import pandas as pd

def merge_event_tables(event_tables: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge multiple event tables into a single time-sorted table.

    Parameters
    ----------
    event_tables : list of pandas.DataFrame
        Each table must contain a timestamp column (``time`` or ``timestamp``)
        and an ``event`` column.

    Returns
    -------
    pandas.DataFrame
        Merged event table with columns ``timestamp`` and ``event``.
    """
    normalized = []

    for df in event_tables:
        if df is None or df.empty:
            continue

        df = df.copy()

        if "timestamp" in df.columns:
            pass
        elif "time" in df.columns:
            df = df.rename(columns={"time": "timestamp"})
        else:
            raise ValueError("Event table missing time/timestamp column")

        normalized.append(df[["timestamp", "event"]])

    return (
        pd.concat(normalized, ignore_index=True)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

def add_odor_times_to_trials(
    trials: pd.DataFrame,
    odors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign odor onset and offset times to trials.

    For each trial, the first odor whose onset falls within
    [start_time, stop_time) is assigned. Trials without an
    odor receive NaNs.
    """
    trials = trials.copy()

    trials["odor_onset"] = np.nan
    trials["odor_offset"] = np.nan

    for i, trial in trials.iterrows():
        mask = (
            (odors["odor_onset"] >= trial["start_time"]) &
            (odors["odor_onset"] < trial["stop_time"])
        )

        trial_odors = odors.loc[mask]

        if not trial_odors.empty:
            first = trial_odors.iloc[0]
            trials.at[i, "odor_onset"] = first["odor_onset"]
            trials.at[i, "odor_offset"] = first["odor_offset"]

    return trials

def add_lick_flag_to_trials(trials: pd.DataFrame, lick_series: pd.Series) -> pd.DataFrame:
    """
    Add a boolean indicating whether any lick occurred during each trial.

    Parameters
    ----------
    trials : pandas.DataFrame
        Must contain ``start_time`` and ``stop_time``.
    lick_series : pandas.Series
        Boolean series indexed by time; True indicates a lick.

    Returns
    -------
    pandas.DataFrame
        Trials table with added ``has_lick`` column.
    """
    trials = trials.copy()

    lick_times = lick_series[lick_series].index.values

    trials["has_lick"] = False

    for i, trial in trials.iterrows():
        trials.at[i, "has_lick"] = (
            (lick_times >= trial["start_time"]) &
            (lick_times < trial["stop_time"])
        ).any()

    return trials

def construct_trials_table(metrics_from_raw: MetricsVrForaging) -> pd.DataFrame:
    metrics_df = metrics_from_raw.active_site_add

    trials_table = pd.DataFrame()

    trials_table["start_time"] = metrics_df.index.values
    trials_table["stop_time"] = metrics_df["stop_time"]
    trials_table["start_position"] = metrics_df["start_position"]
    trials_table["length"] = metrics_df["length"]
    trials_table["site_label"] = metrics_df["label"]
    trials_table["friction"] = metrics_df["friction"]
    trials_table["patch_label"] = metrics_df["patch_label"]
    trials_table["odor_label"] = metrics_df["odor_label"]
   
    trials_table = add_odor_times_to_trials(trials_table, metrics_from_raw.stream_data.odor_triggers)
    trials_table["reward_onset_time"] = metrics_df["reward_onset_time"]
    trials_table["reward_amount"] = metrics_df["reward_amount"]
    trials_table["reward_probability"] = metrics_df["reward_probability"]
    trials_table["reward_available"] = metrics_df["reward_available"]
    trials_table["has_reward"] = metrics_df["is_reward"]
    trials_table["choice_cue_time"] = metrics_df["choice_cue_time"]
    trials_table["has_choice"] = metrics_df["is_choice"]

    trials_table["has_lick"] = add_lick_flag_to_trials(trials_table, metrics_from_raw.stream_data.lick_onset)
    trials_table["reward_delay_duration"] = trials_table["reward_onset_time"] - trials_table["choice_cue_time"]
    trials_table["has_waited_reward_delay"] = metrics_df["succesful_wait_time"]

    return trials_table




    

