import json
import logging

import numpy as np
import pandas as pd
import pynwb
from packaging.version import Version
from scipy.signal import filtfilt, firwin

logger = logging.getLogger(__name__)

# this a mapping that Tiffany provided.
# TODO: get this from aind instrument/rig.json
VR_FORAGING_MAPPING = {
    "Behavior.HarpBehavior.PwmStart": (
        ["PwmDO2"],
        "sound",
        "Sound delivered from hardware",
        True
    ),  # EVENT
    "Behavior.HarpBehavior.PwmStop": (
        ["PwmDO2"],
        "sound_offset",
        "Hardware sound offset",
        True
    ),  # EVENT
    "Behavior.HarpBehavior.PulseSupplyPort0": (
        ["PulseSupplyPort0"],
        "water",
        "Water delivered from hardware",
        True
    ),  # EVENT
    "Behavior.HarpOlfactometer.OdorValveState": (
        ["Valve0", "Valve1", "Valve2"],
        "odor_line_load",
        (
            "Loading of odor to line. "
            "Doesn’t mean odor is presented to the mouse "
            "but needs to happen before EndValve trigger"
            "and defines what odor is "
            "being presented."
        ),
        True
    ),  # EVENT
    "Behavior.HarpOlfactometer.EndValveState": (
        ["EndValve0"],
        "Odor",
        "Odor Delivery",
        True
    ),  # EVENT
    "Behavior.HarpLickometer.LickState": (
        ["Channel0"],
        "Lick",
        "Lick",
        True
    ),  # EVENT
    "Behavior.HarpBehavior.DigitalInputState": (
        ["DIPort0"],
        "Photodiode",
        "Screen synchronization photodiode",
        False
    ),  # CONTINUOUS
    "Behavior.HarpOlfactometer.Channel0ActualFlow": (
        ["Channel0ActualFlow"],
        "Channel0ActualFlow",
        "Measure flow in channel 0",
        False
    ),  # CONTINUOUS
    "Behavior.HarpOlfactometer.Channel1ActualFlow": (
        ["Channel1ActualFlow"],
        "Channel1ActualFlow",
        "Measure flow in channel 1",
        False
    ),  # CONTINUOUS
    "Behavior.HarpOlfactometer.Channel2ActualFlow": (
        ["Channel2ActualFlow"],
        "Channel2ActualFlow",
        "Measure flow in channel 2",
        False
    ),  # CONTINUOUS
    "Behavior.HarpOlfactometer.Channel3ActualFlow": (
        ["Channel3ActualFlow"],
        "Channel3ActualFlow",
        "Measure flow in channel 3",
        False
    ),  # CONTINUOUS
    "Behavior.HarpOlfactometer.Channel4ActualFlow": (
        ["Channel4ActualFlow"],
        "Channel4ActualFlow",
        "Measure flow in channel 4",
        False
    ),  # CONTINUOUS
    "Behavior.HarpSniffDetector.RawVoltage": (
        ["RawVoltage"],
        "Breathing",
        "Breathing signal",
        False
    ),  # CONTINUOUS
    "Behavior.HarpStepperDriver.AccumulatedSteps": (
        ["Motor0, Motor1, Motor2, Motor3"],
        "MotorPositions",
        "The position of x, y1, y2, and z  of the lickspout and oder tube",
        False
    ),  # CONTINUOUS
    "Behavior.HarpTreadmill.SensorData": (
        [
            "Encoder",
            "Torque",
            "TorqueLoadCurrent",
        ],
        "Treadmill",
        "Continuous signal from treadmill",
        False
    ),  # CONTINUOUS
    "Behavior.HarpEnvironmentSensor.SensorData": (
        [
            "Pressure",
            "Temperature",
            "Humidity",
        ],
        "Environment",
        "Continuous signal from environment sensor",
        False
    ),  # CONTINUOUS
}

HED_TAG_MAPPING = {
    "Lick": "Lick",
    "UpdaterRewardDelayOffset": "Reward/Offset",
    "RngSeed": "Random/Quantitative-value",
    "StopVelocityThreshold": "Run/Varying/Speed",
    "water": "Controller-agent/onset",
    "Odor": "Smell",
    "Block": "Virtual-world//Experiment-structure",
    "DepletionVariable": "Quantitative-Value",
    "ActivePatch": "Virtual-world/Experiment-structure",
    "PatchRewardAvailable": "Reward/Quantitative-Value",
    "PatchRewardAmount": "Reward/Quantitative-Value",
    "PatchRewardProbability": "Reward/Probablity",
    "VisualCorridorSpecs": "Controller_Agent",
    "ActiveSite": "Virtual-world/Experiment-structure",
    "ArmOdor": "Experiment-structure/Controller-agent",
    "odor_line_load": "Controller-agent",
    "ChoiceFeedback": "Feedback",
    "sound": "Sound/onset/Controller-agent",
    "WaitRewardOutcome": "Waiting-for",
    "GiveReward": "Reward/Quantitative-value",
}


def normalize_to_json_string(x):
    """
    Normalizes input to a JSON-compatible string for NWB.

    Parameters
    ----------
    x : Any
        The input to normalize.
        Can be a dict, string, None, or other JSON-serializable types.

    Returns
    -------
    str
        A JSON-formatted string representing the input.
    """
    if isinstance(x, dict):
        return json.dumps(x)  # serialize dict (handles nesting)
    elif isinstance(x, str):
        try:
            json.loads(x)  # check if valid JSON string
            return x  # already a valid JSON string
        except json.JSONDecodeError:
            # Not a valid JSON string, re-encode
            return json.dumps(x)
    elif x is None:
        return "null"
    else:
        # fallback: try to serialize other types
        return json.dumps(x)


# ported from Tiffany's processing code
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


# ported from Tiffany's processing code
def fir_filter(
    data: pd.DataFrame,
    col: str,
    cutoff_hz: float,
    num_taps=61,
    nyq_rate=1000 / 2.0,
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
        Length of the filter (number of coefficients, the filter order + 1)
        Default to 61

    nyq_rate: float
        The Nyquist rate of the signal.
        default = 500

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


# ported from Tiffany's processing code
def get_processed_encoder(
    nwb: pynwb.NWBFile, parser: str = "filter"
) -> pd.DataFrame:
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

    # Load data from encoder efficiently
    if current_version >= Version("0.4.0"):
        sensor_data = nwb.acquisition["Behavior.HarpTreadmill.SensorData"][:]

        wheel_size = rig["harp_treadmill"]["calibration"]["output"][
            "wheel_diameter"
        ]
        pulses_per_revolution = rig["harp_treadmill"]["calibration"]["output"][
            "pulses_per_revolution"
        ]
        invert_direction = rig["harp_treadmill"]["calibration"]["output"][
            "invert_direction"
        ]

        converter = (
            wheel_size
            * np.pi
            / pulses_per_revolution
            * (-1 if invert_direction else 1)
        )
        sensor_data["Encoder"] = sensor_data.Encoder.diff()
        dispatch = 250

    elif current_version >= Version("0.3.0") and current_version < Version(
        "0.4.0"
    ):
        sensor_data = nwb.acquisition["Behavior.HarpTreadmill.SensorData"][:]

        wheel_size = rig["harp_treadmill"]["calibration"]["wheel_diameter"]
        pulses_per_revolution = rig["harp_treadmill"]["calibration"][
            "pulses_per_revolution"
        ]
        invert_direction = rig["harp_treadmill"]["calibration"][
            "invert_direction"
        ]

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

            wheel_size = rig["treadmill"][wheel_diameter]
            pulses_per_revolution = rig["treadmill"][pulses]
            invert_direction = rig["treadmill"][invert]

        dispatch = 1000

    converter = (
        wheel_size
        * np.pi
        / pulses_per_revolution
        * (-1 if invert_direction else 1)
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
            encoder["Encoder"]
            .resample("33ms")
            .sum()
            .interpolate(method="linear")
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
    Returns the classification of the register
    from the device provided in the mapping

    Parameters
    ----------
    device_mapping: dict[str, list[str]]
        The mapping of harp device to
        active registers that need to be classified

    Returns
    -------
    dict[str, list[tuple[str, bool]]]
        Dictionary where key is device and value is tuple with register.
        True for event-like, False for continuous-like
    """

    register_event_timseries_classification = {}

    for device, contents in device_mapping.items():
        registers = contents[0]
        name = contents[1]
        description = contents[2]
        is_event = contents[3]

        for register in registers:
            if device not in nwb.acquisition.keys():
                logger.warning(
                    f"No {device} found in acquisition field of nwb."
                )
                continue

            #data = nwb.acquisition[device][:]["Time"]
            is_this_event = is_event
            if device in register_event_timseries_classification:
                register_event_timseries_classification[device].append(
                    (register, is_this_event, name, description)
                )
            else:
                register_event_timseries_classification[device] = [
                    (register, is_this_event, name, description)
                ]

    return register_event_timseries_classification


def is_event(times: np.ndarray, threshold: float = 1) -> bool:
    """
    Classifies where the timestamps are event/discrete or continuous-like.
    Takes mean of diff of times and checks against threshold.

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

    return bool(mean_delta > threshold)
