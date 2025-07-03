import numpy as np
import pynwb
from ndx_events import EventsTable, MeaningsTable

# TODO: get this from aind instrument/rig.json?
VR_FORAGING_MAPPING = {
    "Behavior.HarpBehavior.PwmStart": ["PwmDO2"],  # EVENT
    "Behavior.HarpBehavior.PwmStop": ["PwmDO2"],  # EVENT
    "Behavior.HarpBehavior.PulseSupplyPort0": ["PulseSupplyPort0"],  # EVENT
    "Behavior.HarpOlfactometer.OdorValveState": ["Valve0", "Valve1", "Valve2"],  # EVENT
    "Behavior.HarpOlfactometer.EndValveState": ["EndValve0"],  # EVENT
    "Behavior.HarpLickometer.LickState": ["Channel0"],  # EVENT
    "Behavior.HarpBehavior.DigitalInputState": ["DIPort0"],  # CONTINUOUS
    "Behavior.HarpOlfactometer.Channel0ActualFlow": [
        "Channel0ActualFlow"
    ],  # CONTINUOUS
    "Behavior.HarpOlfactometer.Channel1ActualFlow": [
        "Channel1ActualFlow"
    ],  # CONTINUOUS
    "Behavior.HarpOlfactometer.Channel2ActualFlow": [
        "Channel2ActualFlow"
    ],  # CONTINUOUS
    "Behavior.HarpOlfactometer.Channel3ActualFlow": [
        "Channel3ActualFlow"
    ],  # CONTINUOUS
    "Behavior.HarpOlfactometer.Channel4ActualFlow": [
        "Channel4ActualFlow"
    ],  # CONTINUOUS
    "Behavior.HarpSniffDetector.RawVoltage": ["RawVoltage"],  # CONTINUOUS
    "Behavior.HarpStepperDriver.AccumulatedSteps": [
        "Motor0, Motor1, Motor2, Motor3"
    ],  # CONTINUOUS
    "Behavior.HarpTreadmill.SensorData": [
        "Encoder",
        "Torque",
        "TorqueLoadCurrent",
    ],  # CONTINUOUS
    "Behavior.HarpEnvironmentSensor.SensorData": [
        "Pressure",
        "Temperature",
        "Humidity",
    ],  # CONTINUOUS
}


def get_event_timeseries_classifications(
    device_mapping: dict[str, list[str]], nwb: pynwb.NWBFile
) -> dict[str, list[tuple[str, bool]]]:
    """
    Returns the classification of the register from the device provided in the mapping

    Parameters
    ----------
    device_mapping: dict[str, list[str]]
        The mapping of harp device to active registers that need to be classified

    Returns
    -------
    dict[str, list[tuple[str, bool]]]
        Dictionary where key is device and value is tuple with register and True for event-like,
        False for continuous-like
    """

    register_event_timseries_classification = {}

    for device, registers in device_mapping.items():
        for register in registers:
            if device not in nwb.acquisition.keys():
                continue

            data = nwb.acquisition[device][:]["Time"]
            is_this_event = is_event(data)
            if device in register_event_timseries_classification:
                register_event_timseries_classification[device].append(
                    (register, is_this_event)
                )
            else:
                register_event_timseries_classification[device] = [
                    (register, is_this_event)
                ]

    return register_event_timseries_classification


def is_event(times: np.ndarray, threshold: float = 1) -> bool:
    """
    Classifies where the timestamps are event/discrete or continuous-like. Takes mean of diff of times and checks against threshold.

    Parameters
    ----------
    times: np.ndarray
        Array of timestamps to classify

    threshold: float, default = 1
        The value to compare the mean of the diff of times

    Returns
    -------
    bool: True if event-like, False if continuous-like
    """
    if times.shape[0] < 3:
        return True, 0

    deltas = np.diff(times)
    mean_delta = np.mean(deltas)

    return bool(mean_delta > threshold), mean_delta


def add_event_from_raw_data(
    events_table: EventsTable,
    meanings_table: MeaningsTable,
    event_data: pynwb.core.DynamicTable,
) -> None:
    """
    Adds event data to the EventsTable and MeaningsTable based on the provided table from raw nwb.

    Parameters
    ----------
    events_table : EventsTable
        A table where event data is stored. This table will be updated with new event information.

    meanings_table : MeaningsTable
        A table containing the meanings of events. This table will be updated based on the data provided.

    event_data : pynwb.core.DynamicTable
        A table containing new event data. Each row corresponds to a new event entry and contains
        information that needs to be inserted into both the events_table and meanings_table.

    Returns
    -------
    None
    """
    data = event_data[:]
    for index, row in data.iterrows():
        events_table.add_row(
            timestamp=row["timestamp"], event_name=row["name"], event_data=row["data"]
        )
        meanings_table.add_row(
            value=row["data"],
            meaning=f"{row['name']} - {event_data.description}",
        )
