# NWB Events Specification

**Date:** 2025-11-06

| Event Name | Data Type | Nullable | Unit | Description | Implementation Details |
|------------|-----------|----------|------|-------------|------------------------|
| start_time | number | No | second | Start time, in software, for this site. | Low-precision timestamp |
| stop_time | number | No | second | Stop time, in software, for this site. | Low-precision timestamp |
| start_position | number | No | centimeter | Start coordinate for this site in the VR environment |  |
| length | number | No | centimeter | The length of the site. |  |
| site_id | string | No |  | Label of the site | Uses https://allenneuraldynamics.github.io/Aind.Behavior.VrForaging/api.task_logic.html#aind_behavior_vr_foraging.task_logic.VirtualSiteLabels |
| friction | number | No | percentage | Assigned friction for the site. |  |
| patch_label | string | No |  | Patch type name | Uses (https://allenneuraldynamics.github.io/Aind.Behavior.VrForaging/api.task_logic.html#aind_behavior_vr_foraging.task_logic.Patch.label) |
| odor_label | string | No |  | Odor molecule assigned to patch | (#TODO Not clear where this should be coming from) Not even sure if this should be here at all.... |
| patch_index | integer | No |  | Patch number within the session |  |
| patch_in_block_index | integer | No |  | Patch number within the block |  |
| site_index | integer | No |  | Site number within the session |  |
| site_in_block_index | integer | No |  | Site number within the block |  |
| site_in_patch_index | integer | No |  | Site number within the patch |  |
| site_by_type_in_patch_index | integer | No |  | Same as site_in_patch_index but only counting sites of the same type (e.g. RewardSite) |  |
| odor_onset_time | number | Yes | second | Time of odor onset. Will be null if no odor was delivered. |  |
| odor_offset_time | number | Yes | second | Time of odor offset. Will be null if no odor was delivered. |  |
| reward_onset_time | number | Yes | second | Time when reward was delivered |  |
| reward_amount | number | Yes | milliliter | Amount of reward delivered |  |
| has_reward | boolean | Yes |  | Boolean whether reward was delivered, bool. |  |
| harvest_cue_time | number | Yes | second | Time when harvest cue was delivered. Also can be considered the stop cue. The harvest tone is delivered when a stop is successful. |  |
| has_harvest | boolean | Yes |  | Defines whether a harvest occurred in the site. |  |
| lick_in_site_times | array | No | second | Lick times in the site. Timestamps are still absolute. |  |
| has_lick | boolean | No |  | Defines whether a lick occurred in the site. |  |
| reward_delay_duration | number | Yes | second | reward_onset_time - harvest_cue_time |  |
| has_waited_reward_delay | boolean | Yes |  | Boolean whether the mouse successfully waited through the reward delay to get the reward. Will be null if has_harvest is false. |  |
| reward_probability | number | Yes | percentage | Reward probability at the time of the reward delivery. Will be null if the reward is not sampled (e.g. no has_harvest is False) | Should use the `PatchStateAtReward` SoftwareEvent. Otherwise (older datasets), it should use the closest `PatchState` |
| reward_available | number | Yes | milliliter | Reward left at the time of reward delivery. Will be null if the reward is not sampled (e.g. no has_harvest is False) | Should use the `PatchStateAtReward` SoftwareEvent. Otherwise (older datasets), it should use the closest `PatchState` |
| block_index | integer | No |  | Block number within the session |  |