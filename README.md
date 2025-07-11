# aind-vr-foraging-processing-nwb-packaging

Capsule that processes primary data from vr-foraging task. Capsule can be found here: [vr_foraging_processing_capsule](https://codeocean.allenneuraldynamics.org/capsule/5107215/tree).

This capsule uses the raw packaged data from the [vr_foraging_raw_nwb_capsule](https://codeocean.allenneuraldynamics.org/capsule/3265591/tree). The output of this capsule is a NWB file with the processed data appended to the NWB from the raw packaging. See details below.

### Usage
The code snippet below can be used to read in the NWB and access the relevant data. For the processed data, the 2 most important containers are the `processing` and `events` modules.

The processing module contains `timeseries` objects for the following **continuous** streams currently:
* HarpOlfactometer - Channel0ActualFlow
* HarpOlfactoemeter - Channel1ActualFlow
* HarpOlfactometer - Channel2ActualFlow
* HarpOlfactometer - Channel3ActualFlow
* HarpSniffDetector - RawVoltage
* HarpStepperDriver - AccumulatedSteps
* HarpTreadmill - SensorData (Encoder, Torque, TorqueLoadCurrent)
* HarpEnvironmentSensor - SensorData (Pressure, Temperature, Humidity)

The events module contains a `DynamicTable` for the **event** streams currently:
* HarpBehavior = PwmStart, Hardware sound onset
* HarpBehavior - PwmStop, Hardware sound offset
* HarpBehavior - PulseSupplyPort0, Hardware water onset
* HarpOlfactometer - OdorValveState (Valve0, Valve1, Valve2), Loading of odor to line
* HarpOlfactometer - EndValveState, Odor onset (True) and Odor offset (False)
* HarpLickometer - LickState, Lick onset (True) and Lick offset (False)
* HarpBehavior - DigitalInputState, Screen synchronization of photodiode

```
from hdmf_zarr import NWBZarrIO

# REPLACE WITH PATH TO NWB
with NWBZarrIO('path/to/nwb', 'r') as io:
  nwb = io.read()

events = nwb.get_events__events_tables()
event_table_df = events.to_dataframe()
# recover original data types from nwb
event_table_df["processed_event_data"] = [json.loads(v) if v != "" else None for v in event_table_df["processed_event_data"]]
event_meanings_table = events.meanings_tables["meanings"][:]
timeseries_streams = nwb.processing.data_interfaces
stream = "Treadmill_Encoder"
# access one of streams from timeseries
data = timeseries_streams[stream].data[:] # array with data
timestamps = timeseries_streams[stream].timestamps[:] # array with timestamps for data
```


  
  
