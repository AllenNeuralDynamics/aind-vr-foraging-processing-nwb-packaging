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

def extract_patch_entry_exit(df, patch_label_col="patch_label", patch_onset_col="patch_onset"):
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

import pandas as pd

def extract_reward_events(df,
                           is_reward_col="is_reward",
                           reward_time_col="reward_onset_time"):
    """
    Extract reward delivery events from a snapshot/state table.

    Assumes:
    - index is time (start_time)
    - rows with is_reward == True correspond to reward events
    - reward_onset_time gives the precise event time when available

    Returns
    -------
    pd.DataFrame
        columns: timestamp, event
    """
    events = []

    for ts, row in df.iterrows():
        if row.get(is_reward_col) is True:
            reward_time = row.get(reward_time_col)

            # Use reward_onset_time if available, otherwise fallback to index
            event_time = reward_time if pd.notna(reward_time) else ts

            events.append({
                "timestamp": event_time,
                "event": "reward"
            })

    return pd.DataFrame(events).sort_values("timestamp").reset_index(drop=True)




    

