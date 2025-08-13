# aind-vr-foraging-processing-nwb-packaging

Capsule that processes primary data from vr-foraging task. Capsule can be found here: [vr_foraging_processing_capsule](https://codeocean.allenneuraldynamics.org/capsule/5107215/tree).

This capsule uses the raw packaged data from the [vr_foraging_raw_nwb_capsule](https://codeocean.allenneuraldynamics.org/capsule/3265591/tree). The output of this capsule is a NWB file with the processed data appended to the NWB from the raw packaging. See details below.

### Input Json File Mapping
Currently, 2 json files are needed as input - `hardware_mapping.json` and `hed_tag_mapping.json`. 

**`hardware_mapping.json`**: This is a file with key as the name of the table in the acquisition module, and value as a list containing the **relevant registers** to look at, the name to go in either in the processing or events module, a description, and whether or not it is an event (true) or continuous (false).
An example is shown below
```json
{
  "Behavior.HarpBehavior.PwmStart": [
    ["PwmDO2"],
    "sound",
    "Sound delivered from hardware",
    true
  ],
  "Behavior.HarpBehavior.PwmStop": [
    ["PwmDO2"],
    "sound_offset",
    "Hardware sound offset",
    true
  ],
  "Behavior.HarpBehavior.PulseSupplyPort0": [
    ["PulseSupplyPort0"],
    "water",
    "Water delivered from hardware",
    true
  ],
  "Behavior.HarpOlfactometer.OdorValveState": [
    ["Valve0", "Valve1", "Valve2"],
    "odor_line_load",
    "Loading of odor to line. Doesn’t mean odor is presented to the mouse but needs to happen before EndValve trigger and defines what odor is being presented.",
    true
  ],
  "Behavior.HarpOlfactometer.EndValveState": [
    ["EndValve0"],
    "Odor",
    "Odor Delivery",
    true
  ],
  "Behavior.HarpLickometer.LickState": [
    ["Channel0"],
    "Lick",
    "Lick",
    true
  ],
  "Behavior.HarpBehavior.DigitalInputState": [
    ["DIPort0"],
    "Photodiode",
    "Screen synchronization photodiode",
    false
  ],
  "Behavior.HarpOlfactometer.Channel0ActualFlow": [
    ["Channel0ActualFlow"],
    "Channel0ActualFlow",
    "Measure flow in channel 0",
    false
  ],
  "Behavior.HarpOlfactometer.Channel1ActualFlow": [
    ["Channel1ActualFlow"],
    "Channel1ActualFlow",
    "Measure flow in channel 1",
    false
  ],
  "Behavior.HarpOlfactometer.Channel2ActualFlow": [
    ["Channel2ActualFlow"],
    "Channel2ActualFlow",
    "Measure flow in channel 2",
    false
  ],
  "Behavior.HarpOlfactometer.Channel3ActualFlow": [
    ["Channel3ActualFlow"],
    "Channel3ActualFlow",
    "Measure flow in channel 3",
    false
  ],
  "Behavior.HarpOlfactometer.Channel4ActualFlow": [
    ["Channel4ActualFlow"],
    "Channel4ActualFlow",
    "Measure flow in channel 4",
    false
  ],
  "Behavior.HarpSniffDetector.RawVoltage": [
    ["RawVoltage"],
    "Breathing",
    "Breathing signal",
    false
  ],
  "Behavior.HarpStepperDriver.AccumulatedSteps": [
    ["Motor0", "Motor1", "Motor2", "Motor3"],
    "MotorPositions",
    "The position of x, y1, y2, and z of the lickspout and odor tube",
    false
  ],
  "Behavior.HarpTreadmill.SensorData": [
    ["Encoder", "Torque", "TorqueLoadCurrent"],
    "Treadmill",
    "Continuous signal from treadmill",
    false
  ],
  "Behavior.HarpEnvironmentSensor.SensorData": [
    ["Pressure", "Temperature", "Humidity"],
    "Environment",
    "Continuous signal from environment sensor",
    false
  ]
}
```

**hed_tag_mapping.json**. This a file mapping the event name to the corresponding **HED tags** using the mouse schema defined [here](https://www.hedtags.org/display_hed_prerelease.html?schema=standard_prerelease). These tags are currently in development, an example is shown below:
```json
{
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
    "GiveReward": "Reward/Quantitative-value"
}
```

### Usage
The code snippet below can be used to read in the NWB and access the relevant data. For the processed data, the 2 most important containers are the **`processing`** and **`events`** modules. Raw data can be accessed using the **`acquisition`** module in the NWB file.

The processing module contains `timeseries` objects for the following **continuous** streams currently:
* Encoder data where the signal has a FIR filter applied to it

The events module contains a `DynamicTable` for the **event** streams currently. There are columns for event timestamp, event name, and data associated with the event:
* Water
* Licks
* Sound
* Events for block, site, patch, etc.
* Meanings table with HED tags (still in development)

```
import json
from hdmf_zarr import NWBZarrIO

# REPLACE WITH PATH TO NWB
with NWBZarrIO('path/to/nwb', 'r') as io:
  nwb = io.read()

events = nwb.get_events__events_tables()
event_table_df = events.to_dataframe()
# recover original data types from nwb
event_table_df["event_data"] = [json.loads(v) if v != "" else None for v in event_table_df["event_data"]]
event_meanings_table = events.meanings_tables["meanings"][:]
timeseries_streams = nwb.processing["behavior"].data_interfaces
stream = "Encoder"
# access one of streams from timeseries
data = timeseries_streams[stream].data[:] # array with data
timestamps = timeseries_streams[stream].timestamps[:] # array with timestamps for data
```


  
  
