import json
import logging
from pathlib import Path

import pandas as pd
import utils
from hdmf_zarr import NWBZarrIO
from ndx_events import EventsTable, MeaningsTable, NdxEventsNWBFile
from pydantic import Field
from pydantic_settings import BaseSettings
from pynwb.base import ProcessingModule, TimeSeries

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

    event_timeseries_classification_dict = utils.get_event_timeseries_classifications(
        utils.VR_FORAGING_MAPPING, nwb
    )
    event_table_dict = {
        "timestamp": [],
        "event_name": [],
        "processed_event_data": [],
        "raw_event_data": [],
    }
    if "behavior" not in nwb.processing:
        processing_module = ProcessingModule(
            name="behavior",
            description="behavioral timeseries data for VR Foraging task",
        )
    else:
        processing_module = nwb.processing["behavior"]

    for key, items in event_timeseries_classification_dict.items():
        for item in items:
            is_event = item[1][0]
            column = item[0]

            if not is_event:  # classified as event, skip timeseries
                logger.info(f"Processing timeseries {column} from device {key}")
                timestamps = nwb.acquisition[key][:]["Time"].to_numpy()

                if column == "Encoder":
                    data = utils.get_processed_encoder(nwb)[
                        "filtered_velocity"
                    ].to_numpy()
                elif column == "RawVoltage":
                    data = utils.get_breathing_from_sniff_detector(nwb)
                else:
                    data = nwb.acquisition[key][:][column].to_numpy()
                ts = TimeSeries(
                    name=f"{key}.{column}", data=data, timestamps=timestamps, unit="V"
                )

                processing_module.add(ts)
            else:
                logger.info(f"Processing event {column} from device {key}")
                data = nwb.acquisition[key][:]
                event_table_dict["timestamp"].extend(data["Time"].tolist())
                event_table_dict["event_name"].extend([key for i in range(len(data))])
                event_table_dict["processed_event_data"].extend(data[column].tolist())
                event_table_dict["raw_event_data"].extend(
                    ["" for i in range(len(data))]
                )

    software_event_keys = [
        key for key in list(nwb.acquisition.keys()) if "SoftwareEvents" in key
    ]
    for software_event in software_event_keys:
        data = nwb.acquisition[software_event][:]
        event_table_dict["timestamp"].extend(data["timestamp"].tolist())
        event_table_dict["event_name"].extend(data["name"].tolist())
        event_table_dict["raw_event_data"].extend(data["data"].tolist())
        event_table_dict["processed_event_data"].extend(["" for i in range(len(data))])

    event_table = EventsTable.from_dataframe(
        pd.DataFrame(event_table_dict),
        name="events",
        table_description="Events for VR Foraging task",
    )
    nwb.add_processing_module(processing_module)
    nwb.add_events_table(event_table)

    nwb_output_path = (
        settings.output_directory / f"{raw_nwb_path[0].stem}_processed"
    ).as_posix()
    logger.info(
        f"Finished packaging processed timeseries and events. Writing to disk now at path {nwb_output_path}"
    )

    with NWBZarrIO(nwb_output_path, "w") as io:
        io.export(src_io=source_io, nwbfile=nwb, write_args=dict(link_data=False))
    logger.info("Successfully wrote processed NWB")
