# jkc edit for DPO
from DPO.stepdpo_trainer import StepDPOTrainer
from transformers import HfArgumentParser
# from alignment import (
#     DataArguments,
#     DPOConfig,
#     H4ArgumentParser,
#     ModelArguments,
#     get_checkpoint,
#     get_datasets,
#     get_kbit_device_map,
#     get_peft_config,
#     get_quantization_config,
#     get_tokenizer,
# )
from configs import (
    H4ArgumentParser,
    ModelArguments,
    DataArguments,
    RLArguments,
    DPOConfig
)

def main():
    # parser = H4ArgumentParser(RLArguments)
    print(f"\033[32mOK!\033[0m")
    # model_args, data_args, training_args = parser.parse()


if __name__ == "__main__":
    main()