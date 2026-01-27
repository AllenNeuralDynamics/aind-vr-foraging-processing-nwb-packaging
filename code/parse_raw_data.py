from hdmf_zarr import NWBZarrIO
import numpy as np
from packaging.version import Version
import json 
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.signal import firwin, filtfilt

def extract_behavior_tables_as_dataframes(nwb):
    """
    Extract Behavior.* DynamicTables or JSON from NWB acquisition into a nested dictionary.

    Returns:
        dict: {'HarpBehavior': {'AnalogData': DataFrame, ...}, 'CurrentPosition': DataFrame, ..., 
               'InputSchemas': {'TaskLogic': dict, ...}}
    """
    behavior_data = {}

    for key in nwb.acquisition:
        if not key.startswith("Behavior."):
            continue

        # Remove 'Behavior.' prefix
        subpath = key[len("Behavior."):]

        # Split only on the first '.'
        if '.' in subpath:
            head, subkey = subpath.split('.', 1)
        else:
            head, subkey = subpath, None

        try:
            # Special case: InputSchemas.* keys are JSON in description
            if head == "InputSchemas":
                json_str = nwb.acquisition[key].description
                content = json.loads(json_str)
            else:
                content = nwb.acquisition[key][:]
                if 'Time' in content.columns:
                    content = content.set_index('Time')
                if 'Seconds' in content.columns:
                    content = content.set_index('Seconds')
        except Exception as e:
            print(f"Skipping {key} due to error: {e}")
            continue

        if subkey is None:
            # e.g., Behavior.CurrentPosition
            behavior_data[head] = content
        else:
            # e.g., Behavior.HarpBehavior.AnalogData
            if head not in behavior_data:
                behavior_data[head] = {}
            behavior_data[head][subkey] = content

    return behavior_data

def fir_filter(data, col, cutoff_hz, num_taps=61, nyq_rate=1000 / 2.0):
    """
    Create a FIR filter and apply it to signal.

    nyq_rate (int) = The Nyquist rate of the signal.
    cutoff_hz (float) = The cutoff frequency of the filter: 5KHz
    numtaps (int) = Length of the filter (number of coefficients, i.e. the filter order + 1)
    """

    # Use firwin to create a lowpass FIR filter
    fir_coeff = firwin(num_taps, cutoff_hz / nyq_rate)

    # Use lfilter to filter the signal with the FIR filter
    data['filtered_' + col] = filtfilt(fir_coeff, 1.0, data[col].values)

    return data

class AddExtraColumnsNWB:
    def __init__(self, all_epochs, run_on_init=True):
        self.all_epochs = all_epochs.copy()

        if run_on_init:
            self.add_main_info()
            self.cumulative_consecutive()
            self.patch_time_entry()
            self.skipped_sites()

    def get_odor_sites(self):
        odor_sites = self.all_epochs.loc[self.all_epochs.label == "OdorSite"]
        return odor_sites

    def get_all_epochs(self):
        return self.all_epochs
        
    def patch_time_entry(self):
        self.all_epochs['duration_epoch'] = self.all_epochs['stop_time'] - self.all_epochs.index
                
        patch_number = -1
        first_entry = True
        patch_onset = pd.DataFrame()
        for index, row in self.all_epochs.iterrows():
            if row['label'] == 'InterSite' and patch_number == row['patch_number'] and first_entry:
                new_rows = pd.DataFrame([
                {'patch_number': row['patch_number'], 'patch_onset': row.name}])
                patch_onset = pd.concat([patch_onset, new_rows])
                first_entry = False
                
            if patch_number != row['patch_number']:
                patch_number = row['patch_number']
                first_entry = True
        
        merged_df = pd.merge(self.all_epochs, patch_onset, on='patch_number', how='left')
        self.all_epochs['patch_onset'] = merged_df['patch_onset'].values
        self.all_epochs['time_since_entry'] = self.all_epochs.index - self.all_epochs['patch_onset']
        self.all_epochs['exit_epoch'] = self.all_epochs['time_since_entry'] + self.all_epochs['duration_epoch']

    def cumulative_consecutive(self):
        
        odor_sites = self.all_epochs.loc[self.all_epochs.label == "OdorSite"]
        
        previous_patch = -1
        cumulative_rewards = 0
        consecutive_rewards = 0
        cumulative_failures = 0
        consecutive_failures = 0
        after_choice_cumulative_rewards = 0

        for index, row in odor_sites.iterrows():
            # Total number of rewards in the current patch ( accumulated)
            if row["patch_number"] != previous_patch:
                previous_patch = row["patch_number"]
                cumulative_rewards = 0
                cumulative_failures = 0
                consecutive_failures = 0
                consecutive_rewards = 0
                after_choice_cumulative_rewards = 0

            odor_sites.loc[index, "cumulative_rewards"] = cumulative_rewards
            odor_sites.loc[index, "consecutive_rewards"] = consecutive_rewards
            odor_sites.loc[index, "cumulative_failures"] = cumulative_failures
            odor_sites.loc[index, "consecutive_failures"] = consecutive_failures

            if row["is_reward"] != 0:
                cumulative_rewards += 1
                consecutive_rewards += 1
                consecutive_failures = 0
                after_choice_cumulative_rewards += 1

            odor_sites.loc[index, "after_choice_cumulative_rewards"] = (
                after_choice_cumulative_rewards
            )

            if row["is_reward"] == 0 and row["is_choice"] == True:
                cumulative_failures += 1
                consecutive_failures += 1
                consecutive_rewards = 0

        self.all_epochs = pd.concat([self.all_epochs.loc[self.all_epochs.label != 'OdorSite'], odor_sites], axis=0)
        self.all_epochs = self.all_epochs.sort_index()
        
    def skipped_sites(self):
        odor_sites = self.all_epochs.loc[self.all_epochs.label == "OdorSite"]
        
        skipped_count = 0

        for index, row in odor_sites.iterrows():
            # Number of first sites without stopping - useful for filtering disengagement
            if (row["is_choice"] == False and row["site_number"] == 0):
                skipped_count += 1
            # elif row["is_choice"] == False and row["site_number"] == 1:
            #     skipped_count += 1
            elif row["is_choice"] == True:
                skipped_count = 0
            odor_sites.loc[index, "skipped_count"] = skipped_count
            
        self.all_epochs = pd.concat([self.all_epochs.loc[self.all_epochs.label != 'OdorSite'], odor_sites], axis=0)
        self.all_epochs = self.all_epochs.sort_index()
        
    def add_main_info(self):
        odor_sites = self.all_epochs.loc[
            self.all_epochs.label == "OdorSite"
        ].copy()
        
        # Add column for site number
        odor_sites["odor_sites"] = np.arange(len(odor_sites))

        odor_sites["collected"] = (
            odor_sites["is_reward"] * odor_sites["reward_amount"]
        )

        odor_sites["depleted"] = np.where(
            odor_sites["reward_available"] == 0, 1, 0
        )

        odor_sites["next_site_number"] = odor_sites[
            "site_number"
        ].shift(-2)
        odor_sites["last_visit"] = np.where(
            (odor_sites["next_site_number"] == 0)
            & (odor_sites["is_choice"] == True),
            1,
            0,
        )
        odor_sites.drop(columns=["next_site_number"], inplace=True)

        odor_sites["last_site"] = odor_sites["site_number"].shift(-1)
        odor_sites["last_site"] = np.where(
            odor_sites["last_site"] == 0, 1, 0
        )

        self.all_epochs = pd.concat(
            [self.all_epochs.loc[self.all_epochs.label != "OdorSite"], odor_sites], axis=0
        )
        self.all_epochs = self.all_epochs.sort_index()
        
    def add_time_previous_intersite_interpatch(self):
        self.all_epochs.loc[:, "total_sites"] = 0

        patch_number = -1
        total_sites = -1
        time_interpatch = 0
        time_intersite = 0
        for i, row in self.all_epochs.iterrows():
            if row["label"] == "InterPatch":
                patch_number += 1
                time_interpatch = i
                self.all_epochs.at[i, "patch_number"] = patch_number
            if row["label"] == "InterSite":
                total_sites += 1
                time_intersite = i
                self.all_epochs.at[i, "patch_number"] = patch_number
                self.all_epochs.at[i, "total_sites"] = total_sites
            if row["label"] == "OdorSite":
                if row["site_number"] == 0:
                    self.all_epochs.at[i, "previous_interpatch"] = time_interpatch
                    self.all_epochs.at[i, "previous_intersite"] = time_intersite
                else:
                    self.all_epochs.at[i, "previous_intersite"] = time_intersite

        self.all_epochs["total_sites"] = np.where(
            self.all_epochs["total_sites"] == -1, 0, self.all_epochs["total_sites"]
        )
    
    def add_previous_odor_info(self):
        odor_sites = self.all_epochs.loc[
            self.all_epochs.label == "OdorSite"
        ]
        # -------------------------------- Add previous and next site information ---------------------
        index = odor_sites.index[1:].tolist()
        index.append(0)
        odor_sites["next_odor"] = index

        index = odor_sites["odor_offset"].iloc[:-1].tolist()
        index.insert(0, 0)
        odor_sites["previous_odor"] = index

        self.all_epochs = pd.concat(
            [self.all_epochs.loc[self.all_epochs.label != "OdorSite"], odor_sites], axis=0
        )
        self.all_epochs = self.all_epochs.sort_index()
        
    def add_previous_patch_info(self):
        odor_sites = self.all_epochs.loc[
            self.all_epochs.label == "OdorSite"
        ]
       
        odor_sites["next_patch"] = odor_sites["patch_number"].shift(1)
        odor_sites["next_odor"] = odor_sites["odor_label"].shift(1)
        odor_sites["same_patch"] = np.where(
            (odor_sites["next_patch"] != odor_sites["patch_number"])
            & (odor_sites["odor_label"] == odor_sites["next_odor"]),
            1,
            0,
        )
        odor_sites.drop(columns=["next_patch", "next_odor"], inplace=True)
        
        self.all_epochs = pd.concat(
            [self.all_epochs.loc[self.all_epochs.label != "OdorSite"], odor_sites], axis=0
        )
        self.all_epochs = self.all_epochs.sort_index()   

class TaskSchemaPropertiesNWB:
    """This class is used to store the schema properties of the task configuration.

    tasklogic (str): The key used to access task logic data in the configuration.
    environment (str): The key used to access environment statistics in the configuration.
    reward_specification (str): The key used to access reward specifications in the configuration.
    odor_specifications (str): The key used to access odor specifications in the configuration.
    odor_index (str): The key used to access the odor index in the configuration.
    patches (list): A list of patches in the task configuration.
    """

    def __init__(self, data):
        self._data = data

        if "rig_input" in self._data["InputSchemas"].keys():
            self.rig = "rig_input"
        else:
            self.rig = "Rig"

        if "TaskLogic" in self._data["InputSchemas"].keys():
            self.tasklogic = "TaskLogic"
        else:
            self.tasklogic = "tasklogic_input"

        if "Session" in self._data["InputSchemas"].keys():
            self.session_log = "Session"
        else:
            self.session_log = "session_input"
            
        self.session = self._data["InputSchemas"][self.session_log]['date'][:10]
        self.mouse = int(self._data["InputSchemas"][self.session_log]['subject'])
        self.stage = self._data["InputSchemas"][self.tasklogic]['stage_name']
        self.rig_name = self._data["InputSchemas"][self.rig]['rig_name']
        self.experimenter = self._data["InputSchemas"][self.session_log]['experimenter'][0]
        self.updaters = self._data["InputSchemas"][self.tasklogic]['task_parameters']['updaters']
        try:
            version = Version(self._data["InputSchemas"][self.tasklogic]["version"])
        except KeyError:
            version = Version("0.0.0")
        
        if version >= Version("0.5.1"):
            self.environment = "environment"
            self.reward_specification = "reward_specification"
            self.odor_specifications = "odor_specification"
            self.odor_index = "index"
            
        elif (
            "environment_statistics" in self._data["InputSchemas"][self.tasklogic]
        ):
            self.environment = "environment_statistics"
            self.reward_specification = "reward_specification"
            self.odor_specifications = "odor_specification"
            self.odor_index = "index"
        else:
            self.environment = "environmentStatistics"
            self.reward_specification = "rewardSpecifications"
            self.odor_specifications = "odorSpecifications"
            self.odor_index = "odorIndex"
        
        if "task_parameters" in self._data["InputSchemas"][self.tasklogic]:
            if 'blocks' in self._data["InputSchemas"][self.tasklogic]["task_parameters"][self.environment].keys():
                patches = []
                for blocks in data["InputSchemas"][self.tasklogic]["task_parameters"]['environment']['blocks']:
                    patches.extend(blocks['environment_statistics']['patches'])
                self.patches = patches
            else:
                self.patches = (
                    self._data["InputSchemas"][self.tasklogic].data["task_parameters"][self.environment]["patches"]
                )
        else:
            self.patches = self._data["InputSchemas"][self.tasklogic].data[self.environment]["patches"]

class RewardFunctionsNWB:
    """
    This class is used to calculate and manage reward functions for amount, reward available or probability.

    Attributes:
        _data (dict): A dictionary containing the task InputSchemasuration.
        reward_sites (DataFrame): A pandas DataFrame containing the reward sites data.
    """

    def __init__(self, data, reward_sites):
        """
        The constructor for reward_functions class.

        Parameters:
            data (dict): A dictionary containing the task InputSchemasuration.
            reward_sites (DataFrame): A pandas DataFrame containing the reward sites data.
        """

        self._data = data.copy()
        self.reward_sites = reward_sites.copy()
        self.schema_properties = TaskSchemaPropertiesNWB(self._data)

    def calculate_reward_functions(self):
        self.add_cumulative_rewards()
        self.reward_amount()
        self.reward_probability()
        self.reward_available()
        self.reward_sites.drop(columns=["cumulative_rewards"], inplace=True)
        return self.reward_sites

    def add_cumulative_rewards(self):
        """
        This method calculates the cumulative rewards for each patch in the reward sites.
        """

        previous_patch = -1
        cumulative_rewards = 0

        for index, row in self.reward_sites.iterrows():
            # Total number of rewards in the current patch ( accumulated)
            if row["patch_number"] != previous_patch:
                previous_patch = row["patch_number"]
                cumulative_rewards = 0

            self.reward_sites.loc[index, "cumulative_rewards"] = cumulative_rewards

            if row["is_reward"] != 0:
                cumulative_rewards += 1

    def reward_amount(self):
        """
        This method calculates the reward amount for each reward site based on the reward function specified in the task InputSchemasuration.
        It creates a new column 'reward_amount' in the reward_sites DataFrame.

        Returns:
            DataFrame: The updated reward_sites DataFrame with the 'reward_amount' column.
        """

        # Create a curve for how the reward amount changes in time and create a column with the current value
        x = np.linspace(0, 500, 501)  # Generate 500 points between 0 and 500
        dict_odor = {}

        for patches in self.schema_properties.patches:
            if "reward_function" not in patches[self.schema_properties.reward_specification]:
                dict_odor[patches["label"]] = np.repeat(
                    patches[self.schema_properties.reward_specification]["amount"], 500
                )
                continue
            else:
                function_type = patches[self.schema_properties.reward_specification]["reward_function"]["amount"]["function_type"]
                if (
                    function_type
                    == "ConstantFunction"
                ):
                    odor_label = patches["label"]
                    y = np.repeat(
                        patches[self.schema_properties.reward_specification]["reward_function"]["amount"]["value"],
                        500,
                    )
                elif(
                    function_type
                    == 'LookupTableFunction' ):
                    odor_label = patches["label"]
                    y = np.array(
                        patches[self.schema_properties.reward_specification]['reward_function']['amount']['lut_values']
                    )
                elif(
                    function_type
                    == 'PowerFunction' ):
                    odor_label = patches["label"]
                    a = patches[self.schema_properties.reward_specification]["reward_function"]["amount"]["a"]
                    b = patches[self.schema_properties.reward_specification]["reward_function"]["amount"]["b"]
                    c = -patches[self.schema_properties.reward_specification]["reward_function"]["amount"]["c"]
                    d = patches[self.schema_properties.reward_specification]["reward_function"]["amount"]["d"]

                    # Generate x values
                    y = a * pow(b, -c * x) + d
                elif(
                    function_type == 'LinearFunction'):
                    odor_label = patches["label"]
                    a = patches[self.schema_properties.reward_specification]["reward_function"]["probability"]["a"]
                    b = patches[self.schema_properties.reward_specification]["reward_function"]["probability"]["b"]
                    y = b + a * x
                    
            dict_odor[odor_label] = y

        depletion_rule = patches[self.schema_properties.reward_specification]['reward_function']['depletion_rule']
        if depletion_rule == 'OnChoice':
            update = "site_number"
        elif depletion_rule == 'OnReward':
            update = "cumulative_rewards"
        for index, row in self.reward_sites.iterrows():           
            self.reward_sites.at[index, "reward_amount"] = np.around(
                dict_odor[row["patch_label"]][int(row[update])], 3
            )

        return self.reward_sites

    def reward_probability(self):
        """
        This method calculates the reward probability for each reward site based on the reward function specified in the task InputSchemasuration.
        It creates a new column 'reward_probability' in the reward_sites DataFrame.
        """

        # Create a curve for how the reward probability changes in time and create a column with the current value
        x = np.linspace(0, 500, 501)  # Generate 100 points between 0 and 5
        dict_odor = {}

        for patches in self.schema_properties.patches:
            if "reward_function" not in patches[self.schema_properties.reward_specification]:
                dict_odor[patches["label"]] = np.repeat(
                    patches[self.schema_properties.reward_specification]["probability"],
                    500,
                )
                continue
            else:
                function_type = patches[self.schema_properties.reward_specification]["reward_function"]["probability"]["function_type"]
                if (
                    function_type
                    == "ConstantFunction"
                ):
                    odor_label = patches["label"]
                    y = np.repeat(
                        patches[self.schema_properties.reward_specification]["reward_function"]["probability"]["value"],
                        500,
                    )
                elif(
                    function_type
                    == 'LookupTableFunction'):
                    odor_label = patches["label"]
                    y = np.array(
                        patches[self.schema_properties.reward_specification]['reward_function']['probability']['lut_values']
                    )
                elif(
                    function_type
                    == 'PowerFunction' ):
                    odor_label = patches["label"]
                    a = patches[self.schema_properties.reward_specification]["reward_function"]["probability"]["a"]
                    b = patches[self.schema_properties.reward_specification]["reward_function"]["probability"]["b"]
                    c = -patches[self.schema_properties.reward_specification]["reward_function"]["probability"]["c"]
                    d = patches[self.schema_properties.reward_specification]["reward_function"]["probability"]["d"]

                    # Generate x values
                    y = a * pow(b, -c * x) + d
                elif(
                    function_type == 'LinearFunction'):
                    odor_label = patches["label"]
                    a = patches[self.schema_properties.reward_specification]["reward_function"]["probability"]["a"]
                    b = patches[self.schema_properties.reward_specification]["reward_function"]["probability"]["b"]
                    y = b + a * x
                
            dict_odor[odor_label] = y

        depletion_rule = patches[self.schema_properties.reward_specification]['reward_function']['depletion_rule']
        if depletion_rule == 'OnChoice':
            update = "site_number"
        elif depletion_rule == 'OnReward':
            update = "cumulative_rewards"
            
        #### ----------- Need to add the modification for On Choice, right now specific for OnReward
        for index, row in self.reward_sites.iterrows():
            self.reward_sites.at[index, "reward_probability"] = np.around(
                dict_odor[row["patch_label"]][int(row[update])], 3
            )

    def reward_available(self):
        """
        This method calculates the reward availability for each reward site based on the reward function specified in the task InputSchemasuration.
        It creates a new column 'reward_available' in the reward_sites DataFrame.

        Returns:
            DataFrame: The updated reward_sites DataFrame with the 'reward_available' column.
        """
        # Create a curve for how the reward available changes in time and create a column with the current value
        x = np.linspace(0, 500, 501)  # Generate 100 points between 0 and 5
        dict_odor = {}

        for patches in self.schema_properties.patches:
            # Segment for when the conventions were different. It was always a linear decrease.
            if "reward_function" not in patches[self.schema_properties.reward_specification]:
                if patches["patchRewardFunction"]["initialRewardAmount"] >= 100:
                    dict_odor[patches["label"]] = np.repeat(100, 500)
                else:
                    odor_label = patches["label"]
                    initial = patches["patchRewardFunction"]["initialRewardAmount"]
                    amount = patches[self.schema_properties.reward_specification]["amount"]
                    y = initial - amount * x
                    dict_odor[odor_label] = y
                continue
            else:
                function_type = patches[self.schema_properties.reward_specification]["reward_function"]["available"][
                    "function_type"]
                if (
                    function_type == "ConstantFunction"
                ):
                    odor_label = patches["label"]
                    y = np.repeat(
                        patches[self.schema_properties.reward_specification]["reward_function"]['available']["value"],
                        500,
                    )
                elif(
                    function_type
                    == 'LookupTableFunction' ):
                    odor_label = patches["label"]
                    y = np.array(
                        patches[self.schema_properties.reward_specification]['reward_function']['available']['lut_values']
                    )
                elif(
                    function_type
                    == 'PowerFunction'):
                    odor_label = patches["label"]
                    a = patches[self.schema_properties.reward_specification]["reward_function"]["available"]["a"]
                    b = patches[self.schema_properties.reward_specification]["reward_function"]["available"]["b"]
                    c = -patches[self.schema_properties.reward_specification]["reward_function"]["available"]["c"]
                    d = patches[self.schema_properties.reward_specification]["reward_function"]["available"]["d"]

                    # Generate x values
                    y = a * pow(b, -c * x) + d

                elif(
                    function_type == 'LinearFunction'):
                    odor_label = patches["label"]
                    a = patches[self.schema_properties.reward_specification]["reward_function"]["available"]["a"]
                    b = patches[self.schema_properties.reward_specification]["reward_function"]["available"]["b"]
                    y = b + a * x
                    
            dict_odor[odor_label] = y
            
        depletion_rule = patches[self.schema_properties.reward_specification]['reward_function']['depletion_rule']
        if depletion_rule == 'OnChoice':
            update = "site_number"
        elif depletion_rule == 'OnReward':
            update = "cumulative_rewards"
        
        for index, row in self.reward_sites.iterrows():
            self.reward_sites.at[index, "reward_available"] = np.around(
                dict_odor[row["patch_label"]][int(row[update])], 3)

        return self.reward_sites

class ContinuousDataNWB:
    def __init__(self, data, load_continuous: bool = True):

        self.data = data.copy()

        if "rig_input" in self.data["InputSchemas"].keys():
            self.rig = "rig_input"
        else:
            self.rig = "Rig"

        if "schema_version" in self.data["InputSchemas"][self.rig].keys():
            self.current_version = Version(self.data["InputSchemas"][self.rig]["schema_version"])
        elif "version" in self.data["InputSchemas"][self.rig].keys():
            self.current_version = Version(self.data["InputSchemas"][self.rig]["version"])
        else:
            self.current_version = Version("0.0.0")

        if load_continuous == True:
            self.encoder_data = self.encoder_loading()
            self.choice_feedback = self.choice_feedback_loading()
            self.lick_onset, self.lick_offset = self.lick_onset_loading()
            self.give_reward, self.pulse_duration = self.water_valve_loading()
            self.sniff_data_loading()
            self.position_loading()
            # self.odor_triggers = odor_data_harp_olfactometer(self.data)

    def position_loading(self):
        position = self.data['OperationControl']['CurrentPosition']
        self.position_data = position
        
        return self.position_data
        
    def encoder_loading(self, parser: str = "filter"):
        ## Load data from encoder efficiently
        if self.current_version >= Version("0.4.0"):
            sensor_data = self.data["HarpTreadmill"]['SensorData'].copy()

            wheel_size = self.data["InputSchemas"][self.rig]["harp_treadmill"]["calibration"]['output']["wheel_diameter"]
            PPR = self.data["InputSchemas"][self.rig]["harp_treadmill"]["calibration"]['output']["pulses_per_revolution"]
            invert_direction = (
                self.data["InputSchemas"][self.rig]["harp_treadmill"]["calibration"]['output']["invert_direction"]
            )

            converter = wheel_size * np.pi / PPR * (-1 if invert_direction else 1)
            sensor_data["Encoder"] = sensor_data.Encoder.diff()
            dispatch = 250
            
        elif self.current_version >= Version("0.3.0") and self.current_version < Version("0.4.0"):
            sensor_data = self.data["HarpTreadmill"]['SensorData'].copy()
            
            wheel_size = self.data["InputSchemas"][self.rig]["harp_treadmill"]["calibration"]["wheel_diameter"]
            PPR = self.data["InputSchemas"][self.rig]["harp_treadmill"]["calibration"]["pulses_per_revolution"]
            invert_direction = (
                self.data["InputSchemas"][self.rig]["harp_treadmill"]["calibration"]["invert_direction"]
            )

            sensor_data["Encoder"] = sensor_data.Encoder.diff()
            dispatch = 250

        else:
            sensor_data = self.data['HarpBehavior']['AnalogData']
            if "settings" in self.data["InputSchemas"][self.rig]["treadmill"].keys():
                wheel_size = self.data["InputSchemas"][self.rig]["treadmill"]["settings"]["wheel_diameter"]
                PPR = self.data["InputSchemas"][self.rig]["treadmill"]["settings"]["pulses_per_revolution"]
                invert_direction = (
                    self.data["InputSchemas"][self.rig]["treadmill"]["settings"]["invert_direction"]
                )
            else:
                if "wheel_diameter" in self.data["InputSchemas"][self.rig].data["treadmill"].keys():
                    wheel_diameter = "wheel_diameter"
                    pulses = "pulses_per_revolution"
                    invert = "invert_direction"
                else:
                    wheel_diameter = "wheelDiameter"
                    pulses = "pulsesPerRevolution"
                    invert = "invertDirection"

                wheel_size = self.data["InputSchemas"][self.rig]["treadmill"][wheel_diameter]
                PPR = self.data["InputSchemas"][self.rig]["treadmill"][pulses]
                invert_direction = self.data["InputSchemas"][self.rig]["treadmill"][invert]

            dispatch = 1000
        
        converter = wheel_size * np.pi / PPR * (-1 if invert_direction else 1)
        if parser == "filter":
            sensor_data["velocity"] = (
                sensor_data["Encoder"] * converter
            ) * dispatch  # To be replaced by dispatch rate whe it works
            sensor_data["distance"] = sensor_data["Encoder"] * converter
            sensor_data = fir_filter(sensor_data, "velocity", 50)
            encoder = sensor_data[["filtered_velocity"]]

        elif parser == "resampling":
            encoder = sensor_data["Encoder"]
            encoder = encoder.apply(lambda x: x * converter)
            encoder.index = pd.to_datetime(encoder.index, unit="s")
            encoder = encoder.resample("33ms").sum().interpolate(method="linear") / 0.033
            encoder.index = encoder.index - pd.to_datetime(0)
            encoder.index = encoder.index.total_seconds()
            encoder = encoder.to_frame()
            encoder.rename(columns={"Encoder": "filtered_velocity"}, inplace=True)

        self.encoder_data = encoder

        return self.encoder_data

    def torque_loading(self, parser: str = "filter"):
        ## Load data from encoder efficiently
        if self.current_version >= Version("0.3.0"):
            torque_data = self.data["HarpTreadmill"]['SensorData'].data[["Torque", "TorqueLoadCurrent"]]
            brake_data = self.data["HarpTreadmill"]['BrakeCurrentSetPoint']
        return torque_data, brake_data

    def choice_feedback_loading(self):
        if self.current_version < Version("0.3.0"):
            # Find responses to Reward site
            choice_feedback = self.data['HarpBehavior']['PwmStart'].loc[
                self.data['HarpBehavior']['PwmStart'].data["PwmDO1"] == True
            ]
        else:
            choice_feedback = self.data['HarpBehavior']['PwmStart'].loc[
                self.data['HarpBehavior']['PwmStart']["PwmDO2"] == True
            ]
        return choice_feedback

    def lick_onset_loading(self):
        
        if "HarpLickometer" in self.data:          
            licks = self.data["HarpLickometer"]['LickState']["Channel0"] == True
            lick_onset = licks.loc[licks == True]
            lick_offset = licks.loc[licks == False]

        else:           
            di_state = self.data['HarpBehavior']['DigitalInputState']["DIPort0"]
            lick_onset = di_state.loc[di_state == True]
            lick_offset = di_state.loc[di_state == False]
        return lick_onset, lick_offset

    def water_valve_loading(self):        
        # Find hardware reward events
        give_reward = self.data['HarpBehavior']['OutputSet'][["SupplyPort0"]]
        self.give_reward = give_reward.loc[give_reward.SupplyPort0 == True]

        # Pulses delivered for water
        self.pulse_duration = self.data['HarpBehavior']['PulseSupplyPort0']["PulseSupplyPort0"]

        return self.give_reward, self.pulse_duration

    def sniff_data_loading(self):
        if "HarpSniffsensor" in self.data:
            self.breathing = pd.DataFrame(
                index=self.data["HarpSniffsensor"]['RawVoltage']["RawVoltage"].index,
                columns=["data"],
            )
            self.breathing["data"] = self.data["HarpSniffsensor"]['RawVoltage']["RawVoltage"].values

        else:
            ## Breathing
            self.breathing = pd.DataFrame(
                index=self.data['HarpBehavior']['AnalogData']["AnalogInput0"].index,
                columns=["data"],
            )
            self.breathing["data"] = self.data['HarpBehavior']['AnalogData']["AnalogInput0"].values
        return self.breathing

class MetricsVrForaging:
    def __init__(self, nwb):
        self.data = extract_behavior_tables_as_dataframes(nwb)
        self.active_site = parse_dataframe_NWB(self.data)
        self.active_site_add = AddExtraColumnsNWB(self.active_site).all_epochs
        self.active_site['patch_label'] = self.active_site['patch_label'].apply(get_condition_code)
        
        self.stream_data = ContinuousDataNWB(self.data)
        self.reward_sites = self.active_site.loc[self.active_site['label'] == 'OdorSite']
        
        self.df = self.retrieve_metrics()
        self.schemas = TaskSchemaPropertiesNWB(self.data)
        self.mouse = self.schemas.mouse
        self.session = self.schemas.session
        self.rig_name = self.schemas.rig_name
        self.stage = self.schemas.stage
        self.updaters =  self.schemas.updaters
        
    def retrieve_metrics(self) -> pd.DataFrame:
        reward_sites = self.reward_sites
        active_site = self.active_site
        data = self.data

        df = pd.DataFrame()
        # Summary of different relevants aspects -------------------------------------------------

        unrewarded_stops = reward_sites.loc[reward_sites.is_reward==0]['reward_amount'].count()
        rewarded_stops = reward_sites.loc[reward_sites.is_reward==1]['reward_amount'].count()
        water_collected = reward_sites.loc[(reward_sites['is_reward']==1)]['reward_amount'].sum()
        total_stops = reward_sites.loc[(reward_sites['is_choice']==True)]['reward_amount'].count()

        print('Total sites: ' ,len(reward_sites), ' | ', 'Total rewarded stops: ',rewarded_stops, '(',  np.round((rewarded_stops/total_stops)*100,2),'%) | ', 
            'Total unrewarded stops: ',unrewarded_stops,'(',  np.round((unrewarded_stops/total_stops)*100,2),'%) | ','Water consumed: ', water_collected, 'ul')

        print('Total travelled m: ', np.round(active_site.start_position.max()/100,2), ', current position (cm): ', data['OperationControl']['CurrentPosition'].max()
        )

        for odor_label in reward_sites.odor_label.unique():
            values = reward_sites.loc[(reward_sites['odor_label']==odor_label)&(reward_sites['is_reward']==1)]['reward_amount'].sum()
            print(f'{odor_label} {values} ul')
            
        df.at[0,'odor_sites_travelled'] = int(len(reward_sites))
        df.at[0,'distance_m'] = data['OperationControl']['CurrentPosition'].max().values[0]/100
        df.at[0,'water_collected_ul'] = water_collected
        df.at[0,'rewarded_stops'] = int(rewarded_stops)
        df.at[0,'total_stops'] = int(total_stops)
        df.at[0,'session_duration_min'] = (reward_sites.index[-1] - reward_sites.index[0])/60
        df.at[0, 'total_patches_visited'] = reward_sites.loc[reward_sites['site_number'] >= 1].patch_number.nunique()
        return df

    def retrieve_updater_values(self):
        # Initialize a pointer for the data values
        data_pointer = 0
        
        reward_sites = self.reward_sites
        data = self.data
        df = self.df
        
        # Helper function to safely extract stream data
        def get_stream_data(data, key):
            try:
                stream = data['UpdaterEvents'][key]['data']
                stream.reset_index(drop=True, inplace=True)
                return stream
            except (KeyError, AttributeError):
                return None

        # Load updater data safely
        stop_duration = get_stream_data(data, 'UpdaterStopDurationOffset')
        delay = get_stream_data(data, 'UpdaterRewardDelayOffset')
        velocity_threshold = get_stream_data(data, 'UpdaterStopVelocityThreshold')

        # Create new columns in reward_sites with default values
        reward_sites['delay_s'] = np.nan
        reward_sites['velocity_threshold_cms'] = np.nan
        reward_sites['stop_duration_s'] = np.nan

        data_pointer = 0
        try:
            for index, row in reward_sites.iterrows():
                if row['is_reward'] == 1:
                    if delay is not None and len(delay) > data_pointer:
                        reward_sites.at[index, 'delay_s'] = delay[data_pointer]
                    if velocity_threshold is not None and len(velocity_threshold) > data_pointer:
                        reward_sites.at[index, 'velocity_threshold_cms'] = velocity_threshold[data_pointer]
                    if stop_duration is not None and len(stop_duration) > data_pointer:
                        reward_sites.at[index, 'stop_duration_s'] = stop_duration[data_pointer]
                    data_pointer += 1
                else:
                    if delay is not None and len(delay) > data_pointer:
                        reward_sites.at[index, 'delay_s'] = delay[data_pointer]
                    if velocity_threshold is not None and len(velocity_threshold) > data_pointer:
                        reward_sites.at[index, 'velocity_threshold_cms'] = velocity_threshold[data_pointer]
                    if stop_duration is not None and len(stop_duration) > data_pointer:
                        reward_sites.at[index, 'stop_duration_s'] = stop_duration[data_pointer]
        except IndexError:
            if delay is not None:
                reward_sites.at[index, 'delay_s'] = delay.max()
            if velocity_threshold is not None:
                reward_sites.at[index, 'velocity_threshold_cms'] = velocity_threshold.max()
            if stop_duration is not None:
                reward_sites.at[index, 'stop_duration_s'] = stop_duration.max()

        # Summary of the training metrics
        reward_sites['odor_sites'] = np.arange(1, len(reward_sites) + 1)

        # Safely update df only if values exist
        if delay is not None:
            df.at[0, 'start_delay'] = reward_sites['delay_s'].min()
            df.at[0, 'end_delay'] = reward_sites['delay_s'].max()
            df.at[0, 'sites_to_max_delay'] = reward_sites[reward_sites['delay_s'] == reward_sites['delay_s'].max()].iloc[0]['odor_sites']

        if stop_duration is not None:
            df.at[0, 'start_stop_duration'] = reward_sites['stop_duration_s'].min()
            df.at[0, 'end_stop_duration'] = reward_sites['stop_duration_s'].max()
            df.at[0, 'sites_to_max_stop_duration'] = reward_sites[reward_sites['stop_duration_s'] == reward_sites['stop_duration_s'].max()].iloc[0]['odor_sites']
            df.at[0, 'rewarded_sites_in_max_stop'] = int(reward_sites[(reward_sites['stop_duration_s'] == reward_sites['stop_duration_s'].max()) & (reward_sites.is_choice == 1)]['odor_sites'].nunique())

        if velocity_threshold is not None:
            df.at[0, 'start_velocity_threshold'] = reward_sites['velocity_threshold_cms'].min()
            df.at[0, 'end_velocity_threshold'] = reward_sites['velocity_threshold_cms'].max()
            df.at[0, 'target_max_velocity_threshold'] = reward_sites['velocity_threshold_cms'].max()
            df.at[0, 'sites_to_min_velocity'] = reward_sites[reward_sites['velocity_threshold_cms'] == reward_sites['velocity_threshold_cms'].min()].iloc[0]['odor_sites']
            df.at[0, 'sites_to_max_velocity'] = reward_sites[reward_sites['velocity_threshold_cms'] == reward_sites['velocity_threshold_cms'].max()].iloc[0]['odor_sites']        
        
        self.reward_sites = reward_sites
        self.df = df

    def get_metrics(self):
        return self.df

    def get_reward_sites(self):
        return self.reward_sites
    
    def get_mouse_and_session(self):
        return self.mouse, self.session
    
    def run_pdf_summary(self):
        color1='#d95f02'
        color2='#1b9e77'
        color3='#7570b3'
        color4='#e7298a'

        color_dict_label = {'InterSite': '#808080',
            'InterPatch': '#b3b3b3', 
            'PatchZ': '#d95f02', 'PatchZB': '#d95f02', 
            'PatchB': '#d95f02','PatchA': '#7570b3', 
            'PatchC': '#1b9e77',
            'Alpha-pinene': '#1b9e77', 
            'Methyl Butyrate': '#7570b3', 
            'Amyl Acetate': '#d95f02', 
            'Fenchone': '#7570b3', 
            'S': color1,
            'D': color2,
            'N': color3,   
            'Do': color1
            }
        
        odor_sites = self.reward_sites.copy()
        encoder_data = self.stream_data.encoder_data
        active_site = self.active_site.copy()
        
        active_site['mouse'] = self.mouse
        active_site['session'] = self.session
        
        # Apply function
        active_site['long_patch_label'] = active_site['patch_label']
        active_site['patch_label'] = active_site['patch_label'].apply(get_condition_code)
        
        # odor_sites['odor_label'] = odor_sites['odor_label'].str.replace(' ', '_')
        
        # Remove segments where the mouse was disengaged
        # last_engaged_patch = odor_sites['patch_number'][odor_sites['skipped_count'] >= 10].min()
        # if pd.isna(last_engaged_patch):
        #     last_engaged_patch = odor_sites['patch_number'].max()
            
        # odor_sites['engaged'] = odor_sites['patch_number'] <= last_engaged_patch  
    
        try:
            odor_sites['block'] = odor_sites['patch_label'].str.extract(r'set(\d+)').astype(int)
        except ValueError: 
            odor_sites['block'] = 0

        # Apply function
        odor_sites['long_patch_label'] = odor_sites['patch_label']
        odor_sites['patch_label'] = odor_sites['patch_label'].apply(get_condition_code)
        odor_sites['odor_sites'] = np.arange(len(odor_sites))
        
        trial_summary = trial_collection(odor_sites[['is_choice', 'site_number', 'odor_label', 'depleted', 'odor_sites', 'is_reward','reward_probability','reward_amount','reward_available']], 
                                                  encoder_data, 
                                                  window=(-1,3)
                                                )
    
        # Save each figure to a separate page in the PDF
        pdf_filename = f'{self.mouse}_{self.session}_summary.pdf'
        with PdfPages("/results"+"/"+pdf_filename) as pdf:
            text1 = ('Mouse: ' + str(self.mouse) 
            + '\nSession: ' + str(self.session) 
            + '\nRig: ' + str(self.rig_name) 
            + '\nStage: ' + str(self.stage)
            + '\nTotal sites: '  + str(self.df.total_stops.iloc[0]) 
            + '\nTotal rewarded stops: ' + str(self.df.rewarded_stops.iloc[0]) + ' (' +str(np.round((self.df.rewarded_stops.iloc[0]/self.df.total_stops.iloc[0])*100,2)) + '%) \n' 
            + 'Water consumed: ' +  str(np.round(self.df.water_collected_ul.iloc[0], 2)) + 'ul\n' 
            + 'Session duration: ' + str(np.round(self.df.session_duration_min.iloc[0],2)) + 'min\n' 
            + 'Total travelled m: ' + str(np.round(active_site.start_position.max()/100,2))
            )
            
            # '(',  np.round((rewarded_stops/total_stops)*100,2),'%) | ', 
            text_to_figure = text1
            if self.stage[:7] == 'shaping':
                text2 = '\nTotal sites travelled: ' + str(self.df.odor_sites_travelled.iloc[0]) + '\nRewarded stops in max stop duration: ' + str(self.df.rewarded_sites_in_max_stop.iloc[0]) + '\nTotal patches visited: ' + str(self.df.total_patches_visited.iloc[0])
                text_to_figure = text1 + text2
            
            # Create a figure
            fig, ax = plt.subplots(figsize=(8.5, 11))  # Standard letter size
            ax.text(0.1, 0.9, text_to_figure, ha='left', va='center', fontsize=12)
            ax.axis('off')  # Hide the axes
            pdf.savefig(fig)
            plt.close(fig)
            
            # raster_with_velocity(active_site, stream_data, color_dict_label=color_dict_label, save=pdf)
            segmented_raster_vertical(odor_sites, 
                                            save=pdf, 
                                            color_dict_label=color_dict_label)
            raster_with_velocity(active_site, self.stream_data, color_dict_label=color_dict_label, save=pdf)
        
            summary_withinsession_values(odor_sites, 
                                    color_dict_label = color_dict_label, 
                                    save=pdf)
            speed_traces_efficient(trial_summary, self.mouse, self.session,  save=pdf)
            preward_estimates(odor_sites, 
                                    color_dict_label = color_dict_label, 
                                    save=pdf)
            speed_traces_value(trial_summary, self.mouse, self.session, condition = 'reward_probability', save=pdf) 
            velocity_traces_odor_entry(trial_summary, max_range = trial_summary.speed.max(), color_dict_label=color_dict_label, save=pdf)

            length_distributions(self.active_site, self.data, delay=True, save=pdf)
            if len(self.updaters):
                update_values(self.reward_sites, save=pdf)
                
        return pdf_filename

# Helper function to safely extract stream data
def get_stream_data(data, key):
    try:
        stream = data['UpdaterEvents'][key]['data']
        stream.reset_index(drop=True, inplace=True)
        return stream
    except (KeyError, AttributeError):
        return None

# Function to assign codes
def get_condition_code(text):
    if 'delayed' in text:
        return 'D'
    elif 'single' in text:
        return 'S'
    elif 'no_reward' in text or 'noreward' in text:
        return 'N'
    elif 'double' in text:
        return 'Do'
    else:
        return text

def parse_dataframe_NWB(data: dict) -> pd.DataFrame:
    """
    Parse the data from the session and return the reward sites, active sites and encoder data
    
    Inputs:
    data: dict
        Data from the session
    
    Returns:
    all_epochs: pd.DataFrame
        DataFrame containing the  active sites
    
    """
    # Use json_normalize to create a new DataFrame from the 'data' column
    active_site = pd.json_normalize(data["SoftwareEvents"]['ActiveSite']['data'].apply(lambda x: json.loads(x)).reset_index()['data'])
    active_site.index = data["SoftwareEvents"]['ActiveSite'].timestamp
    
    # Add the postpatch label
    active_site["previous_epoch"] = active_site["label"].shift(-1)
    active_site["label"] = np.where(
        active_site["label"] == active_site["previous_epoch"], "PostPatch", active_site["label"]
    )
    active_site.drop(columns=["previous_epoch"], inplace=True)
    
    active_site["label"] = active_site["label"].replace("Reward", "OdorSite")
    active_site["label"] = active_site["label"].replace("RewardSite", "OdorSite")
    
    if "treadmill_specification.friction.distribution_parameters.value" in active_site.columns:
        active_site.rename(
            columns={
                "startPosition": "start_position",
                "treadmill_specification.friction.distribution_parameters.value": "friction",
            },
            inplace=True,
        )
        # Crop and rename columns
        active_site = active_site[["label", "start_position", "length", "friction"]]
    
    else:
        active_site.rename(columns={"startPosition": "start_position"}, inplace=True)
        active_site = active_site[["label", "start_position", "length"]]
    
    # Add patch_number column
    group = (active_site["label"] == "InterPatch").cumsum()
    active_site["patch_number"] = group - 1
    
    # Patch initialization
    patches = data["SoftwareEvents"]['ActivePatch']
    patches.index = patches.timestamp

    # Instances where a patch gets defined but it's not really used. Happens during block transitions. 
    patches['real_diff'] = patches.index.to_series().diff().shift(-1).fillna(5)
    patches = patches[patches.real_diff >= 0.1]
    
    df_patch = pd.json_normalize(patches['data'].apply(lambda x: json.loads(x)).reset_index()['data'])
    df_patch.index = patches.timestamp
    
    df_patch["patch_number"] = np.arange(len(df_patch))
    if "odor_specification.index" in df_patch.columns:
        df_patch.rename(columns={"label": "patch_label", "odor_specification.index": "odor_label"}, inplace=True)
        df_patch = df_patch[["patch_label", "patch_number", "odor_label"]]
    else: 
        df_patch.rename(columns={"label": "odor_label"}, inplace=True)
        df_patch = df_patch[["patch_number", "odor_label"]]
        df_patch["patch_label"] = df_patch["odor_label"]
        
    all_epochs = pd.merge(active_site, df_patch, on="patch_number", how="left")
    all_epochs.index = active_site.index
    
    # ------------
    try:
        if 'calibration' in data["InputSchemas"]['Rig']["harp_olfactometer"]:
            if data["InputSchemas"]['Rig']["harp_olfactometer"]["calibration"] is not None:
                # Create a mapping dictionary from the nested structure
                mapping = {i: data["InputSchemas"]['Rig']["harp_olfactometer"]["calibration"]['input']['channel_config'][str(i)]['odorant'] for i in range(0, 3)}
    
                # Replace numbers in the dataframe column with the corresponding odorant values
                all_epochs['odor_label'] = all_epochs['odor_label'].replace(mapping)
                
            else:
                all_epochs["odor_label"] = all_epochs['patch_label']   
        else:
            all_epochs["odor_label"] = all_epochs['patch_label']
    except:
        all_epochs["odor_label"] = all_epochs['patch_label']
    # ----------------
    
    # Count 'OdorSite' occurrences within each group
    all_epochs["site_number"] = all_epochs[all_epochs["label"] == "OdorSite"].groupby(group).cumcount()
    all_epochs["stop_time"] = all_epochs.index.to_series().shift(-1)
    all_epochs.index.name = "start_time"
    
    # ## Add last timestamp
    # try:
    #     data["InputSchemas"]endsession.load_from_file()
    #     all_epochs.stop_time.iloc[-1] = data['InputSchemas']endsession.data['timestamp']    
    # except json.JSONDecodeError:
    #     print('Removing last epoch because of empty endsession file')
    #     all_epochs = all_epochs.loc[:-1]
    # except AttributeError:
    #     print('Removing last epoch because of empty endsession file')
    #     all_epochs = all_epochs.loc[:-1]
        
    # Recover tones
    choiceFeedback = ContinuousDataNWB(data, load_continuous=False).choice_feedback_loading()
    
    # Recover water delivery
    water = ContinuousDataNWB(data, load_continuous=False).water_valve_loading()[0]
    
    if "WaitRewardOutcome" in data["SoftwareEvents"]:
        succesfull_wait = pd.json_normalize(data["SoftwareEvents"]['WaitRewardOutcome']['data'].apply(lambda x: json.loads(x)).reset_index()['data'])
        succesfull_wait.index = data["SoftwareEvents"]['WaitRewardOutcome'].timestamp
        succesfull_wait = succesfull_wait[succesfull_wait["IsSuccessfulWait"] == True]
    else:
        succesfull_wait = pd.Series([])
    
    stop_cues = []
    reward_onsets = []
    successful_waits = []
    reward_sites = all_epochs[all_epochs["label"] == "OdorSite"]
    if reward_sites.empty:
        print("No reward sites found")
    
    # Loop over the reward_sites
    for current_idx, row in reward_sites.iterrows():
        # Define the current and next reward site index
        next_idx = row.stop_time
        
        # Find slices based on the current and next indices
        choice = choiceFeedback[(choiceFeedback.index >= current_idx) & (choiceFeedback.index < next_idx)]
        reward_in_site = water[(water.index >= current_idx) & (water.index < next_idx)]
        waits = succesfull_wait[(succesfull_wait.index >= current_idx) & (succesfull_wait.index < next_idx)]
        
        # Store the first relevant index or NaN
        stop_cues.append(choice.index[0] if len(choice) > 0 else np.nan)
        reward_onsets.append(reward_in_site.index[0] if len(reward_in_site) > 0 else np.nan)
    
        # If the stop is not rewarded, the delay is the wait
        successful_waits.append(waits.index[0] if len(waits) > 0 else np.nan)
    
    # Assign the results to the DataFrame
    reward_sites["choice_cue_time"] = stop_cues
    reward_sites["reward_onset_time"] = reward_onsets
    reward_sites["succesful_wait_time"] = successful_waits
    reward_sites["succesful_wait_time"] = reward_sites["reward_onset_time"].combine_first(reward_sites["succesful_wait_time"])
    
    # Add the new columns for choice and reward delivered
    reward_sites["is_choice"] = reward_sites["choice_cue_time"].notnull().astype(bool)
    reward_sites["is_reward"] = reward_sites["reward_onset_time"].notnull().astype(bool)
    
    try:
        reward_sites = RewardFunctionsNWB(data, reward_sites).calculate_reward_functions()
        all_epochs = pd.concat([all_epochs.loc[all_epochs.label != 'OdorSite'], reward_sites], axis=0).sort_index()
        
    except KeyError:
        print("Reward functions from software events")
        
        # Add the reward characteristics columns
        patch_stats = pd.DataFrame()
        patch_stats.index = data['SoftwareEvents']['PatchRewardProbability'].index
        patch_stats['reward_amount'] = data['SoftwareEvents']['PatchRewardAmount'].values
        patch_stats['reward_available'] = data['SoftwareEvents']['PatchRewardAvailable'].values
        patch_stats['reward_probability'] = data['SoftwareEvents']['PatchRewardProbability'].values
        patch_stats['reward_probability'] = patch_stats['reward_probability'].round(3)
    
        patch_stats['real_diff'] = patch_stats.index.to_series().diff().shift(-1).fillna(0.1)
        patch_stats = patch_stats[patch_stats.real_diff >= 0.03]
        
        # Make sure both DataFrames are sorted by index
        reward_sites = reward_sites.sort_index()
        patch_stats = patch_stats.sort_index()
    
        # Perform merge_asof on the index
        merged = pd.merge_asof(
            reward_sites,
            patch_stats.drop(columns=["real_diff"]),
            left_index=True,
            right_index=True,
            direction='backward'
        )
    
        assert len(merged) == len(reward_sites), "Length mismatch after merge"
    
        # Concatenate the results to all_epochs
        all_epochs = pd.concat([all_epochs.loc[all_epochs.label != 'OdorSite'], merged], axis=0).sort_index()
    
    ## ------------------------------------------------------------------------- ##
    return all_epochs

