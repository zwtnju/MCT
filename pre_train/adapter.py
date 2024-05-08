import logging

from transformers import HoulsbyInvConfig, PfeifferConfig, ParallelConfig, \
    PrefixTuningConfig, LoRAConfig, IA3Config, AdapterConfig

logger = logging.getLogger(__name__)





def getAdapter(adapter_type):
    if "houlsby" in adapter_type:
        adapter_config = HoulsbyInvConfig()
    elif "pfeiffer" in adapter_type:
        adapter_config = PfeifferConfig()
    elif "parallel" in adapter_type:
        adapter_config = ParallelConfig()
    elif "prefix_tuning" in adapter_type:
        adapter_config = PrefixTuningConfig()
    elif "lora" in adapter_type:
        adapter_config = LoRAConfig()
    elif "ia3" in adapter_type:
        adapter_config = IA3Config()
    else:
        adapter_config = AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor={'default': 16},
                                       non_linearity='gelu')
    return adapter_config


ADAPTER_TYPE = ["houlsby", "pfeiffer", "parallel", "prefix_tuning", "lora", "ia3"]
