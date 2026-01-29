import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import scipy
from aind_data_schema.core.processing import DataProcess
from hdmf_zarr import NWBZarrIO
from ndx_events import EventsTable, MeaningsTable
from pydantic import Field
from pydantic_settings import BaseSettings
from pynwb.base import DynamicTable, ProcessingModule, TimeSeries

import utils
import parse_raw_data

logger = logging.getLogger(__name__)


class VRForagingSettings(BaseSettings, cli_parse_args=True):
    """
    Settings for VR Foraging Primary Data NWB Packaging
    """

    input_directory: Path = Field(
        default=Path("/data/vr_foraging_raw_nwb"),
        description="Directory where data is",
    )
    output_directory: Path = Field(
        default=Path("/results/"), description="Output directory"
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    settings = VRForagingSettings()
    paths = tuple(settings.input_directory.glob("*"))
    raw_nwb_path = [path for path in paths if path.is_dir()]

    if not raw_nwb_path:
        raise FileNotFoundError("No raw nwb found")

    logger.info(f"Found raw nwb at path {raw_nwb_path[0]}")

    source_io = NWBZarrIO(raw_nwb_path[0].as_posix(), "r")
    nwb = source_io.read()

    start_process_time = datetime.now()
    metrics_from_nwb = parse_raw_data.MetricsVrForaging(nwb)
    odor_events = utils.extract_odor_events(metrics_from_nwb.stream_data.odor_triggers)
    site_events = utils.extract_site_entry_exit(metrics_from_nwb.active_site_add)
    patch_events = utils.extract_patch_entry_exit(metrics_from_nwb.active_site_add)
    cue_events = utils.extract_binary_events(metrics_from_nwb.active_site_add, "is_choice", "choice_cue_time", "Cue")
    reward_events = utils.extract_binary_events(metrics_from_nwb.active_site_add, "is_reward", "reward_onset_time", "reward")
    lick_events = utils.extract_lick_events(metrics_from_nwb.stream_data.lick_onset)

    event_table_df = utils.merge_event_tables(
        [
            odor_events, site_events, patch_events, cue_events, reward_events, lick_events
        ]
    )
    trial_table_df = utils.construct_trials_table(metrics_from_nwb)

    event_table = EventsTable.from_dataframe(
        event_table_df,
        name="events",
        table_description="Events for VR Foraging task",
    )

    nwb_output_path = (
        settings.output_directory / f"{raw_nwb_path[0].stem}-processed.zarr"
    ).as_posix()
    logger.info("Finished packaging processed timeseries and events.")
    logger.info(f"Writing to disk now at path {nwb_output_path}")

    with NWBZarrIO(nwb_output_path, "w") as io:
        io.export(src_io=source_io, nwbfile=nwb)
    logger.info("Successfully wrote processed NWB")

    with open(settings.output_directory / "data_process.json", "w") as f:
        f.write(data_process.model_dump_json(indent=4))
