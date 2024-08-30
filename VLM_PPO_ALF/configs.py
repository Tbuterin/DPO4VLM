import dataclasses
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, NewType, Optional, Tuple

import transformers
from transformers import MODEL_FOR_CAUSAL_LM_MAPPING, HfArgumentParser

import torch


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


DataClassType = NewType("DataClassType", Any)



class H4ArgumentParser(HfArgumentParser):
    def parse_yaml_and_args(self, yaml_arg: str, other_args: Optional[List[str]] = None) -> List[dataclass]:
        """
        Parse a YAML file and overwrite the default/loaded values with the values provided to the command line.

        Args:
            yaml_arg (`str`):
                The path to the config file used
            other_args (`List[str]`, *optional`):
                A list of strings to parse as command line arguments, e.g. ['--arg=val', '--arg2=val2'].

        Returns:
            [`List[dataclass]`]: a list of dataclasses with the values from the YAML file and the command line
        """
        arg_list = self.parse_yaml_file(os.path.abspath(yaml_arg))

        outputs = []
        # # strip other args list into dict of key-value pairs
        # other_args = {arg.split("=")[0].strip("-"): arg.split("=")[1] for arg in other_args}
        # used_args = {}


        # jkc：更新处理命令行参数的逻辑 comlogic
        combined_args = {}
        if other_args:
            it = iter(other_args)
            for arg in it:
                if "=" in arg:
                    key, value = arg.split("=", 1)
                else:
                    key = arg.strip("-")
                    value = next(it, None)
                combined_args[key] = value

        used_args = {}



        # overwrite the default/loaded value with the value provided to the command line
        # adapted from https://github.com/huggingface/transformers/blob/d0b5002378daabf62769159add3e7d66d3f83c3b/src/transformers/hf_argparser.py#L327
        for data_yaml, data_class in zip(arg_list, self.dataclass_types):
            keys = {f.name for f in dataclasses.fields(data_yaml) if f.init}
            inputs = {k: v for k, v in vars(data_yaml).items() if k in keys}
            # for arg, val in other_args.items():
            for arg, val in combined_args.items():
                # add only if in keys
                if arg in keys:
                    base_type = data_yaml.__dataclass_fields__[arg].type
                    inputs[arg] = val

                    # cast type for ints, floats (default to strings)
                    if base_type in [int, float]:
                        inputs[arg] = base_type(val)

                    if base_type == List[str]:
                        inputs[arg] = [str(v) for v in val.split(",")]

                    # bool of a non-empty string is True, so we manually check for bools
                    if base_type == bool:
                        if val in ["true", "True"]:
                            inputs[arg] = True
                        else:
                            inputs[arg] = False

                    # add to used-args so we can check if double add
                    if arg not in used_args:
                        used_args[arg] = val
                    else:
                        raise ValueError(f"Duplicate argument provided: {arg}, may cause unexpected behavior")

            obj = data_class(**inputs)
            outputs.append(obj)

        return outputs

    def parse(self) -> DataClassType | Tuple[DataClassType]:
        if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
            # If we pass only one argument to the script and it's the path to a YAML file,
            # let's parse it to get our arguments.
            output = self.parse_yaml_file(os.path.abspath(sys.argv[1]))
        # parse command line args and yaml file
        elif len(sys.argv) > 2 and sys.argv[1].endswith(".yaml"):
            output = self.parse_yaml_and_args(os.path.abspath(sys.argv[1]), sys.argv[2:])
        # parse command line args only
        else:
            output = self.parse_args_into_dataclasses()

        if len(output) == 1:
            output = output[0]
        return output



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """

    base_model_revision: Optional[str] = field(
        default=None,
        metadata={"help": ("The base model checkpoint for weights initialization with PEFT adatpers.")},
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    model_code_revision: str = field(default=None, metadata={"help": "The branch of the IFT model"})
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    trust_remote_code: bool = field(default=False, metadata={"help": "Trust remote code when loading a model."})
    use_flash_attention_2: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use flash attention 2. You must install this manually by running `pip install flash-attn --no-build-isolation`"
            )
        },
    )
    use_peft: bool = field(
        default=False,
        metadata={"help": ("Whether to use PEFT or not for training.")},
    )
    lora_r: Optional[int] = field(
        default=16,
        metadata={"help": ("LoRA R value.")},
    )
    lora_alpha: Optional[int] = field(
        default=32,
        metadata={"help": ("LoRA alpha.")},
    )
    lora_dropout: Optional[float] = field(
        default=0.05,
        metadata={"help": ("LoRA dropout.")},
    )
    lora_target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": ("LoRA target modules.")},
    )
    lora_modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={"help": ("Model layers to unfreeze & train")},
    )
    load_in_8bit: bool = field(default=False, metadata={"help": "use 8 bit precision"})
    load_in_4bit: bool = field(default=False, metadata={"help": "use 4 bit precision"})

    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4", metadata={"help": "precise the quantization type (fp4 or nf4)"}
    )
    use_bnb_nested_quant: bool = field(default=False, metadata={"help": "use nested quantization"})

    def __post_init__(self):
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("You can't use 8 bit and 4 bit precision at the same time")


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    dataset_mixer: Optional[Dict[str, float]] = field(
        default=None,
        metadata={"help": ("Datasets and their proportions to be used for training ift/rl.")},
    )
    dataset_splits: Optional[List[str]] = field(
        default_factory=lambda: ["train", "test"],
        metadata={"help": ("List of train test splits to use in the dataset")},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    truncation_side: Optional[str] = field(
        default=None, metadata={"help": "Truncation side to use for the tokenizer."}
    )


@dataclass
class SFTConfig(transformers.TrainingArguments):
    """
    Arguments related to the training process itself. For all parameters, see: https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/trainer#transformers.TrainingArguments
    """

    max_seq_length: Optional[int] = field(
        default=None,
        metadata={"help": ("Used by TRL for reward model training, which tries to read this parameter in init.")},
    )
    logging_first_step: bool = field(
        default=True,
        metadata={"help": ("Whether to log and evaluate the first global_step or not.")},
    )
    optim: Optional[str] = field(default="adamw_torch")


@dataclass
class DPOConfig():
    """
    Arguments related to the DPO training process itself. For all parameters, see: https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/trainer#transformers.TrainingArguments
    """

    beta: Optional[float] = field(
        default=0.1,
        metadata={"help": "The beta factor in DPO loss. Higher beta means less divergence from the initial policy."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": ("The Hub model branch to push the model to.")},
    )
    logging_first_step: bool = field(
        default=True,
        metadata={"help": ("Whether to log and evaluate the first global_step or not.")},
    )
    max_prompt_length: Optional[int] = field(
        default=None,
        metadata={"help": ("For DPO, the maximum length of the prompt to use for conditioning the model.")},
    )
    max_length: Optional[int] = field(
        default=None,
        metadata={"help": ("Used by TRL for reward model training, which tries to read this parameter in init.")},
    )
    optim: Optional[str] = field(default="rmsprop")
    remove_unused_columns: bool = field(default=False)



@dataclass
class RLArguments:
    init_lr: float = field(default=1e-6, metadata={"help": "initial learning rate"})
    end_lr: float = field(default=1e-8, metadata={"help": "final learning rate"})
    weight_decay: float = field(default=0, metadata={"help": "weight decay"})
    explore_portion: float = field(default=0.1, metadata={"help": "rate of exploration, updates, a number between 0-1"})
    eps: float = field(default=1e-7, metadata={"help": "RMSprop optimizer epsilon"})
    alpha: float = field(default=0.99, metadata={"help": "RMSprop optimizer alpha"})
    gamma: float = field(default=0.9, metadata={"help": "discount factor for rewards"})
    use_gae: bool = field(default=False, metadata={"help": "use generalized advantage estimation"})
    gae_lambda: float = field(default=0.95, metadata={"help": "gae lambda parameter"})
    entropy_coef: float = field(default=0.01, metadata={"help": "entropy term coefficient"})
    value_loss_coef: float = field(default=0.5, metadata={"help": "value loss coefficient"})
    max_grad_norm: float = field(default=0.01, metadata={"help": "max norm of gradients"})
    seed: int = field(default=1, metadata={"help": "random seed"})
    cuda_deterministic: bool = field(default=False, metadata={"help": "sets flags for determinism when using CUDA"})
    num_processes: int = field(default=1, metadata={"help": "how many training CPU processes to use"})
    num_steps: int = field(default=256, metadata={"help": "number of environment steps collected at each iteration"})
    ppo_epoch: int = field(default=4, metadata={"help": "number of ppo epochs"})
    grad_accum_steps: int = field(default=2, metadata={"help": "the number of gradient accumulation steps"})
    mini_batch_size: int = field(default=1, metadata={"help": "size of mini-batches for each update"})
    clip_param: float = field(default=0.1, metadata={"help": "ppo clip parameter"})
    log_interval: int = field(default=10, metadata={"help": "log interval, one log per n updates"})
    save_interval: int = field(default=100, metadata={"help": "save interval, one save per n updates"})
    eval_interval: Optional[int] = field(default=None, metadata={"help": "eval interval, one eval per n updates"})
    num_env_steps: int = field(default=int(10e6), metadata={"help": "number of environment steps to train"})
    env_name: str = field(default='gym_cards/Blackjack-v0', metadata={"help": "environment to train on"})
    save_dir: str = field(default='./trained_models/', metadata={"help": "directory to save agent logs"})
    no_cuda: bool = field(default=False, metadata={"help": "disables CUDA training"})
    use_proper_time_limits: bool = field(default=False, metadata={"help": "compute returns taking into account time limits"})
    recurrent_policy: bool = field(default=False, metadata={"help": "use a recurrent policy"})
    use_linear_lr_decay: bool = field(default=False, metadata={"help": "use a linear schedule on the learning rate"})
    eval_num_per_episode: int = field(default=100, metadata={"help": "number of episodes to evaluate the agent"})
    lr_max_steps: int = field(default=100, metadata={"help": "number of steps for lr scheduler"})
    
    # Arguments for llava interface
    model_path: Optional[str] = field(default=None)
    model_base: Optional[str] = field(default=None)
    pretrain_mm_adapter: Optional[str] = field(default=None)
    # prompt: str = field(default="What is the next action? Please format your response as 'The next action is {response}.")
    # data_path: str = field(default="../gym-cards/bc_data.json")
    image_folder: str = field(default="../gym-cards/images")
    question_file: str = field(default="tables/question.jsonl")
    answers_file: str = field(default="answer.jsonl")
    conv_mode: str = field(default="llava_v1")
    num_chunks: int = field(default=1)
    chunk_idx: int = field(default=0)
    max_new_tokens: int = field(default=128)
    temperature: float = field(default=0.2)
    top_p: Optional[float] = field(default=None)
    num_beams: int = field(default=1)
    cache_dir: Optional[str] = field(default=None)
    use_lora: bool = field(default=False)
    train_vision: str = field(default='all')
    thought_prob_coef: float = field(default=1.0, metadata={"help": "any number between 0-1, multiplier for the log thought probability"})
    action_only_prompt: bool = field(default=False)
    
    # Arguments for supporting alf config file
    alf_config: Optional[str] = field(default=None)
    
    # Arguments for wandb
    use_wandb: bool = field(default=False)
    wandb_project: str = field(default='test')
    wandb_run: str = field(default='test')
    q4: bool = field(default=False)
    q8: bool = field(default=False)

    def __post_init__(self):
        self.cuda = not self.no_cuda and torch.cuda.is_available()


@dataclass
class StepDPOConfig(DPOConfig):
    data_path: str = field(default="xinlai/math-step-dpo-10K")
    prompt: str = field(default="alpaca")

