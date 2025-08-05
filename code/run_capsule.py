import json
import logging
from pathlib import Path

import pandas as pd
import utils
from hdmf_zarr import NWBZarrIO
from ndx_events import EventsTable, MeaningsTable
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

    event_timeseries_classification_dict = (
        utils.get_event_timeseries_classifications(
            utils.VR_FORAGING_MAPPING, nwb
        )
    )
    event_table_dict = {
        "timestamp": [],
        "event_name": [],
        "event_data": [],
    }
    meanings_table_dict = {"value": [], "meaning": []}

    if "behavior" not in nwb.processing:
        processing_module = ProcessingModule(
            name="behavior",
            description="behavioral timeseries data for VR Foraging task",
        )
    else:
        processing_module = nwb.processing["behavior"]

    for key, items in event_timeseries_classification_dict.items():
        for item in items:
            is_event = item[1]
            column = item[0]
            name = item[2]
            description = item[3]

            name_for_nwb = None
            # if name != column:
            #     name_for_nwb = f"{name}_{column}"
            # else:
            name_for_nwb = name

            if not is_event:  # classified as timeseries
                if column != "Encoder": # only processing done on Encoder
                    continue

                logger.info(
                    f"Processing timeseries {column} from device {key}"
                )
                timestamps = nwb.acquisition[key][:]["Time"].to_numpy()

                data = utils.get_processed_encoder(nwb)[
                    "filtered_velocity"
                ].to_numpy()

                ts = TimeSeries(
                    name=name_for_nwb,
                    data=data,
                    timestamps=timestamps,
                    unit="s",
                    description=f"{name_for_nwb} - {description} with a FIR filter applied",
                )

                processing_module.add(ts)
            else:  # event
                logger.info(f"Processing event {column} from device {key}")
                data = nwb.acquisition[key][:]
                # Generate mask based on column type
                if data[column].dtype == bool and name != "Lick" and name != "Odor":
                    mask = data[column].tolist()
                else:
                    mask = [True] * len(data)

                # Filtered rows
                filtered_rows = data[mask].reset_index(drop=True)
                filtered_column_values = filtered_rows[column].tolist()

                # Unique values for meanings
                unique_values = pd.Series(filtered_column_values).unique()
                for value in unique_values:
                    if f"{name_for_nwb} - {value}" not in meanings_table_dict["value"]:
                        meanings_table_dict["value"].append(f"{name_for_nwb} - {value}")
                        meanings_table_dict["meaning"].append(description)

                # Fill event table with filtered rows only
                event_table_dict["timestamp"].extend(filtered_rows["Time"].tolist())
                event_table_dict["event_name"].extend([name_for_nwb] * len(filtered_rows))
                event_table_dict["event_data"].extend(
                    [json.dumps(d) for d in filtered_column_values]
                )

    software_event_keys = [
        key for key in list(nwb.acquisition.keys()) if "SoftwareEvents" in key
    ]
    for software_event in software_event_keys:
        data = nwb.acquisition[software_event][:]
        event_table_dict["timestamp"].extend(data["timestamp"].tolist())
        event_table_dict["event_name"].extend(
            name.split('.')[-1] for name in data["name"]
        )
        event_table_dict["event_data"].extend(data["data"].apply(utils.normalize_to_json_string).tolist())
        for name in data["name"].unique():
            meanings_table_dict["value"].append(f"{name.split('.')[-1]}")
            meanings_table_dict["meaning"].append(nwb.acquisition[software_event].description)


    meanings_table = MeaningsTable.from_dataframe(
        pd.DataFrame(meanings_table_dict),
        name="meanings",
        # probably better way, but violates flake8 if not like this
        table_description=(
            "Description of values in events table for VR Foraging task",
        )[0]
    )
    # sort by timestamps
    sorted_indices = sorted(range(len(event_table_dict["timestamp"])), key=lambda i: event_table_dict["timestamp"][i])
    # Apply the sort to all keys
    event_table_dict = {k: [v[i] for i in sorted_indices] for k, v in event_table_dict.items()}

    event_table = EventsTable.from_dataframe(
        pd.DataFrame(event_table_dict),
        name="events",
        table_description="Events for VR Foraging task",
    )
    event_table.add_meanings_tables(meanings_table)
    nwb.add_processing_module(processing_module)
    nwb.add_events_table(event_table)

    nwb_output_path = (
        settings.output_directory / f"{raw_nwb_path[0].stem}-processed"
    ).as_posix()
    logger.info(
        "Finished packaging processed timeseries and events."
    )
    logger.info(
        f"Writing to disk now at path {nwb_output_path}"
    )

    with NWBZarrIO(nwb_output_path, "w") as io:
        io.export(
            src_io=source_io, nwbfile=nwb, write_args=dict(link_data=False)
        )
    logger.info("Successfully wrote processed NWB")
