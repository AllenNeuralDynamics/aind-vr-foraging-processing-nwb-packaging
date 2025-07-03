import json

import numpy as np
import pandas as pd
import pynwb
from ndx_events import EventsTable, MeaningsTable
from packaging.version import Version
from scipy.signal import filtfilt, firwin

# TODO: get this from aind instrument/rig.json
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


def get_breathing_from_sniff_detector(nwb: pynwb.NWBFile) -> np.ndarray:
    """
    Gets the breating from the sniff detector raw data

    Parameters
    ----------
    nwb: pynwb.NWBFile
        The nwb with the raw data

    Returns
    -------
    np.ndarray
        The result array with the breathing data
    """
    if "Behavior.HarpSniffDetector.RawVoltage" in nwb.acquisition.keys():
        return nwb.acquisition["Behavior.HarpSniffDetector.RawVoltage"][:][
            "RawVoltage"
        ].to_numpy()

    else:
        return nwb.acquisition["Behavior.HarpBehavior.AnalogData"][:][
            "AnalogInput0"
        ].to_numpy()


def fir_filter(
    data: pd.DataFrame, col: str, cutoff_hz: float, num_taps=61, nyq_rate=1000 / 2.0
) -> pd.DataFrame:
    """
    Create a FIR filter and apply it to signal.

    Parameters
    ----------
    data: pd.DataFrame
        Input data to apply filter to

    col: str
        The column to be added to the result

    cutoff_hz: float
        The cuttoff frequency of the filter

    numtaps: int, default = 61
        Length of the filter (number of coefficients, i.e. the filter order + 1). Default to 61

    nyq_rate: float = The Nyquist rate of the signal.

    cutoff_hz: float, default = 500
        The cutoff frequency of the filter: 5KHz

    Returns
    -------
    pd.DataFrame
        The result dataframe with the filter applied
    """

    # Use firwin to create a lowpass FIR filter
    fir_coeff = firwin(num_taps, cutoff_hz / nyq_rate)

    # Use lfilter to filter the signal with the FIR filter
    data["filtered_" + col] = filtfilt(fir_coeff, 1.0, data[col].values)

    return data


def get_processed_encoder(nwb: pynwb.NWBFile, parser: str = "filter") -> pd.DataFrame:
    """
    Processes the raw encoder data to return filtered velocity

    Parameters
    ----------
    nwb: pynwb.NWBFile
        The raw nwb to pull the encoder data

    parser: str, default = "filter"
        Either apply a FIR filter or resample

    Returns
    -------
    pd.DataFrame
        The processed velocity from the encoder data
    """
    rig = json.loads(nwb.acquisition["Behavior.InputSchemas.Rig"].description)
    current_version = Version(rig["version"])

    ## Load data from encoder efficiently
    if current_version >= Version("0.4.0"):
        sensor_data = nwb.acquisition["Behavior.HarpTreadmill.SensorData"][:]

        wheel_size = rig["harp_treadmill"]["calibration"]["output"]["wheel_diameter"]
        pulses_per_revolution = rig["harp_treadmill"]["calibration"]["output"][
            "pulses_per_revolution"
        ]
        invert_direction = rig["harp_treadmill"]["calibration"]["output"][
            "invert_direction"
        ]

        converter = (
            wheel_size * np.pi / pulses_per_revolution * (-1 if invert_direction else 1)
        )
        sensor_data["Encoder"] = sensor_data.Encoder.diff()
        dispatch = 250

    elif current_version >= Version("0.3.0") and current_version < Version("0.4.0"):
        sensor_data = nwb.acquisition["Behavior.HarpTreadmill.SensorData"][:]

        wheel_size = rig["harp_treadmill"]["calibration"]["wheel_diameter"]
        pulses_per_revolution = data["InputSchemas"][rig]["harp_treadmill"][
            "calibration"
        ]["pulses_per_revolution"]
        invert_direction = rig["harp_treadmill"]["calibration"]["invert_direction"]

        sensor_data["Encoder"] = sensor_data.Encoder.diff()
        dispatch = 250

    else:
        sensor_data = nwb.acquisition["Behavior.HarpBehavior.AnalogData"][:]
        if "settings" in rig["treadmill"].keys():
            wheel_size = rig["treadmill"]["settings"]["wheel_diameter"]
            pulses_per_revolution = rig["treadmill"]["settings"][
                "pulses_per_revolution"
            ]
            invert_direction = rig["treadmill"]["settings"]["invert_direction"]
        else:
            if "wheel_diameter" in rig["treadmill"].keys():
                wheel_diameter = "wheel_diameter"
                pulses = "pulses_per_revolution"
                invert = "invert_direction"
            else:
                wheel_diameter = "wheelDiameter"
                pulses = "pulsesPerRevolution"
                invert = "invertDirection"

            wheel_size = data["InputSchemas"][rig]["treadmill"][wheel_diameter]
            pulses_per_revolution = data["InputSchemas"][rig]["treadmill"][pulses]
            invert_direction = data["InputSchemas"][rig]["treadmill"][invert]

        dispatch = 1000

    converter = (
        wheel_size * np.pi / pulses_per_revolution * (-1 if invert_direction else 1)
    )
    if parser == "filter":
        sensor_data["velocity"] = (
            sensor_data["Encoder"] * converter
        ) * dispatch  # To be replaced by dispatch rate whe it works
        sensor_data["distance"] = sensor_data["Encoder"] * converter
        sensor_data = fir_filter(sensor_data, "velocity", 50)
        encoder = sensor_data[["filtered_velocity"]]

    elif parser == "resampling":
        encoder = sensor_data[["Time", "Encoder"]]
        encoder["Encoder"] = encoder.apply(lambda x: x * converter)
        encoder["Time"] = pd.to_datetime(encoder["Time"], unit="s")
        encoder["Encoder"] = (
            encoder["Encoder"].resample("33ms").sum().interpolate(method="linear")
            / 0.033
        )
        encoder["Time"] = encoder["Time"] - pd.to_datetime(0)
        encoder["Time"] = encoder["Time"].total_seconds()
        encoder = encoder.to_frame()
        encoder.rename(columns={"Encoder": "filtered_velocity"}, inplace=True)

    encoder_data = encoder

    return encoder_data


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
