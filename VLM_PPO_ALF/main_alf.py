from patch import replace_llama_attn_with_xformers_attn
replace_llama_attn_with_xformers_attn()

import time
from collections import deque

import gymnasium as gym
from gymnasium import spaces
import gym_cards
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision.transforms import ToPILImage  # jkc
import matplotlib.pyplot as plt  # jkc

from a2c_ppo_acktr import algo, utils, rl_utils
from a2c_ppo_acktr.rl_utils import get_prompt, text_projection, get_alfworld_prompt
from a2c_ppo_acktr.rl_utils import get_dpo_prompt  # jkc0904
# from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.model import VLMPolicy, VLMValue
from a2c_ppo_acktr.storage import RolloutStorage, TrajBuffer  # jkc
from a2c_ppo_acktr.llava_interface import llava_evaluate, llava_generate
from a2c_ppo_acktr.llava_interface import init_pretrained_model, find_all_linear_names, load_lora_model

# For alfworld
from alf_utils import load_config_file, get_obs_image, ALF_ACTION_LIST, process_action, compute_reward, AlfEnv


from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
## IMAGE_TOKEN_INDEX: 在处理图像或混合数据（例如文本和图像）时，用于指示图像数据在整个数据序列中的索引位置。
## DEFAULT_IMAGE_TOKEN: 表示一个默认的图像标记，用于在数据序列中插入图像的占位符。这通常是一个特殊的标记，用于与非图像数据（如文本）区分开来。
## DEFAULT_IM_START_TOKEN: 表示图像数据段的起始标记。这通常用于指示序列中图像数据的起始位置，方便模型在处理时正确地识别和处理图像数据。
## DEFAULT_IM_END_TOKEN: 表示图像数据段的结束标记。这与起始标记一起使用，用于定义图像数据在序列中的范围。

from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.model import LlavaLlamaForCausalLM

from functools import partial
from typing import List, Optional
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoImageProcessor
import transformers

from tqdm import tqdm

import accelerate
from accelerate.state import AcceleratorState

import warnings
warnings.filterwarnings("ignore")

import os
import copy

# jkc edit for DPO
from a2c_ppo_acktr.model import DPOPolicy
from transformers import HfArgumentParser
from configs import (
    H4ArgumentParser,
    ModelArguments,
    DataArguments,
    RLArguments,
    DPOConfig,
    StepDPOConfig
)
from dataclasses import dataclass, field


def torch_init(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        print(f"\033[32mCUDA Deterministic.\033[0m")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.set_num_threads(1)  # 限制 PyTorch （在CPU上）只使用一个线程，通常用于避免多线程竞争导致的性能下降。


def load_base_model(args):
    model_path = args.model_path
    cache_dir = args.cache_dir

    # 打印模型路径。如果路径中包含 lora，加载 LoRA 模型，并检查是否支持 8bit 或 4bit 量化。如果不包含 lora，则加载标准的 LLaVA 模型，可能使用 8bit 或 4bit 量化。
    print(f"Path of the model is {model_path}")
    if "lora" in model_path:
        base, tokenizer = load_lora_model(model_path, cache_dir=cache_dir)
        if args.q8 or args.q4:
            raise ValueError("Lora model does not support 8bit or 4bit quantization")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
        if args.q8:
            print("8bit quantization")
            base = LlavaLlamaForCausalLM.from_pretrained(model_path, load_in_8bit=True, cache_dir=cache_dir)
        elif args.q4:
            q4_config = transformers.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type='nf4'
                    )
            print("4bit quantization")
            base = LlavaLlamaForCausalLM.from_pretrained(model_path, load_in_4bit=True, quantization_config=q4_config, cache_dir=cache_dir)
        else:
            base = LlavaLlamaForCausalLM.from_pretrained(model_path, cache_dir=cache_dir)
    return base, tokenizer




def main():
    """
    注释了 get_args()
    添加了 args, model_args, data_args, training_args分别为RLArguments, ModelArguments, DataArguments, StepDPOConfig
    替换了 actor_critic结构为policy_model, 并添加ref_model = policy_model
    替换了 agent设置为alg0.DPO
    注释了 双端队列的初始化
    注释了 wandb中所有value相关的参数
    """

    ############################################################
    # 使用 H4ArgumentParser 来解析模型、数据和训练参数 KEY: addhfparser
    ############################################################
    parser = H4ArgumentParser((RLArguments, ModelArguments, DataArguments, StepDPOConfig))   # jkc0829
    args, model_args, data_args, training_args = parser.parse()   # jkc0829

    ###############
    # torch settings
    ###############
    torch_init(args)

    #########################
    # load model and tokenizer
    #########################
    base, tokenizer = load_base_model(args)  # base: 创建的Llava模型
    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.grad_accum_steps)  # 处理分布式训练和梯度累积  # TODO
    device = accelerator.device
    print(f"\033[33mUsing {device}.\033[0m")
    model_device = device
    base = base.to(model_device)  # jkc0904
    
    print(f"\033[32mModel created.\033[0m")

    base.config.max_length = 1024  # @TODO: 修改更大值，因为加入了历史数据
    print(f"\033[33mModel max context length:\033[0m{base.config.max_length}")
    base, tokenizer = init_pretrained_model(base, tokenizer, pretrain_mm_adapter = args.pretrain_mm_adapter)
    image_processor = base.get_vision_tower().image_processor

    # 配置 LoRA，设置其超参数 r, lora_alpha, target_modules, lora_dropout 等。如果启用了 LoRA，则使用该配置更新基础模型。
    base_lora_config = LoraConfig(
            r=128,
            lora_alpha=256,
            target_modules=find_all_linear_names(base,args.train_vision),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
    if args.use_lora:
        base = get_peft_model(base, base_lora_config)


    ## Inputing Prompt here
    ###############
    ## 实例化环境
    ###############
    assert args.alf_config is not None, "Alfworld environment requires a config file"
    print(f"\033[33mCreating Env: {args.alf_config}\033[0m")
    print(f"\033[33mPath: {os.getenv('ALFWORLD_DATA')}\033[0m")
    envs = AlfEnv(args.alf_config)
    obs, infos = envs.reset(seed=args.seed)
    admissible_commands = list(infos['admissible_commands'])[0]


    #################### Traj Storage ####################
    # jkc0904
    trajs = TrajBuffer(training_args.max_pairs, args.num_processes, training_args.max_history_tokens, args.max_new_tokens, (300, 300, 3), history_horizon=training_args.history_horizon)

    trajs.add_test_state(tokenizer)
    # print(f"\033[41m{type(infos['observation_text'])}: {infos['observation_text']}\033[0m")
    trajs.start_traj(infos['observation_text'][0])

    #################### Traj Storage End ####################


    # 生成提示词 @TODO:需要修改提示词, 加入历史数据🌟
    history = trajs.get_history_data()
    # qs = get_alfworld_prompt(envs, obs = infos['observation_text'], admissible_actions=admissible_commands, action_only = args.action_only_prompt)
    qs = get_dpo_prompt(envs, obs = infos['observation_text'], admissible_actions=admissible_commands, history=history, action_only = args.action_only_prompt)
    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv = conv_templates[args.conv_mode].copy()  # 使用对话模板构建对话并生成最终的提示文本。
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    print(f"\033[34m{prompt}\033[0m")

    # 使用 tokenizer_image_token 函数将提示文本转化为输入 ID，返回张量格式，并确保所有零值位置都被替换为特定的标记（259）。
    INPUT_IDS = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
    INPUT_IDS[INPUT_IDS == 0] = 259 # 869: . (period), 29871: SPIECE, 259: whitespace

    if "alfred" in args.env_name.lower():
        projection_f = partial(lambda x: x)

    policy_model = DPOPolicy(tokenizer=tokenizer,
                             image_processor=image_processor,
                             base=base,
                             projection_f=projection_f,
                             INPUT_IDS=INPUT_IDS,
                             args=args)
    ref_model = copy.deepcopy(policy_model) ##TODO
    for param in ref_model.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(policy_model.base.parameters(), lr=args.init_lr, eps=args.eps, weight_decay=args.weight_decay)  # 余弦退火学习率调度器，随着训练过程逐渐减少学习率。
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.lr_max_steps, eta_min=args.end_lr)
    AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = 1  # 设置 DeepSpeed 的训练微批大小为 1。

    policy_model, optimizer, lr_scheduler = accelerator.prepare(policy_model, optimizer, lr_scheduler) ##TODO

    #################################################################
    # 创建 DPO（Direct Preference Optimization）代理，用于强化学习的策略优化。
    #################################################################
    agent = algo.DPO(
            policy_model,
            ref_model,
            optimizer,
            accelerator,
            training_args.beta,
            args.ppo_epoch,
            args.mini_batch_size,
            args.max_grad_norm,
            training_args.label_smoothing,
            training_args.reference_free
            )


    ######################################## fzr TODO ########################################
    ## 创建一个 RolloutStorage 实例，用于存储回合数据，参数包括步数、进程数、观察空间、动作空间和最大新标记数量。
    # rollouts = RolloutStorage(args.num_steps, args.num_processes,
    #                           (300, 300, 3), spaces.Discrete(14), args.max_new_tokens)


    image_tensor = obs
    last_step_obs = copy.deepcopy(obs)  # 记得更新！！🌟

    ## 执行模型的 act 函数，基于输入图像张量和输入 ID 生成动作和相关的概率信息，并获取可行命令。
    output_ids, action, action_log_prob, action_tokens_log_prob = policy_model.act(image_tensor, INPUT_IDS = INPUT_IDS)
    admissible_commands = list(infos['admissible_commands'])[0]

    print(f"\033[34moutput_ids:\033[0m{output_ids}")
    print(f"\033[34mprompt:\033[0m{prompt}")
    print(f"\033[34maction:\033[0m{action}")
    print(f"\033[34maction_log_prob:\033[0m{action_log_prob}")
    print(f"\033[34maction_tokens_log_prob:\033[0m{action_tokens_log_prob}")


    # 初始化多个双端队列，用于存储每个回合的奖励、成功率、动作标记日志概率等信息，队列长度为每个回合最大评估次数。
    episode_rewards = deque(maxlen=args.eval_num_per_episode)
    episode_success_rate = deque(maxlen=args.eval_num_per_episode)
    episode_gc_success_rate = deque(maxlen=args.eval_num_per_episode)
    episode_succ_rate_pick_and_place = deque(maxlen=args.eval_num_per_episode)
    episode_succ_rate_pick_two_obj_and_place = deque(maxlen=args.eval_num_per_episode)
    episode_succ_rate_look_at_obj_in_light = deque(maxlen=args.eval_num_per_episode)
    episode_succ_rate_pick_heat_then_place_in_recep = deque(maxlen=args.eval_num_per_episode)
    episode_succ_rate_pick_cool_then_place_in_recep = deque(maxlen=args.eval_num_per_episode)
    episode_succ_rate_pick_clean_then_place_in_recep = deque(maxlen=args.eval_num_per_episode)
    episode_action_tokens_log_prob = deque(maxlen=args.eval_num_per_episode)



    ##########################################################################################
    ######################################## 开始训练 ########################################
    ##########################################################################################
    # 记录开始时间，计算训练中的更新次数。如果使用 wandb（Weights and Biases）进行实验追踪，初始化 wandb，并创建一个用于记录文本数据的表格。
    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    if args.use_wandb:
        import wandb
        run_name = args.wandb_run + "-" + args.env_name
        wandb.init(project=args.wandb_project, name=run_name, group=run_name, config=args)
        text_table = wandb.Table(columns=["epoch", "obs_text", "text_action"])
    # print(f"\033[44mprompt\033[34m:{prompt}\033[0m")
    running_episode_rewards = torch.zeros(args.num_processes).flatten()

    ### 主循环
    rs = []
    for j in tqdm(range(num_updates)):

        for step in tqdm(range(args.num_steps)):
            print(f"\033[31mstep {step} in {args.num_steps} total\033[0m")
            # Sample actions
            with torch.no_grad():
                INPUT_IDS = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
                output_id, action, action_log_prob, action_tokens_log_prob = policy_model.act(last_step_obs, INPUT_IDS = INPUT_IDS)  # TODO

                admissible_commands = list(infos['admissible_commands'])[0]
            text_action = tokenizer.decode(list(filter(lambda num: num != 0, output_id[0].tolist())))

            # Observation, reward and next obs
            # 执行动作，获取新观察、奖励、完成标志和信息。如果环境名称包含 alfred，则重新生成提示。
            obs, reward, done, infos = envs.step(action) # for alf this will already process action
            last_step_obs = copy.deepcopy(obs) # 更新last_obs🌟
            # print(f"\033[32mReward: {reward}\033[0m")
            if "alfred" in args.env_name.lower():
                admissible_commands = list(infos['admissible_commands'])[0]
                history = trajs.get_history_data()  # jkc0904
                # qs = get_alfworld_prompt(envs, obs = infos['observation_text'], admissible_actions=admissible_commands, action_only = args.action_only_prompt)
                qs = get_dpo_prompt(envs, obs = infos['observation_text'], admissible_actions=admissible_commands, history=history, action_only = args.action_only_prompt)
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
                conv = conv_templates[args.conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

            #################### Traj Storage ####################
            ######################################## fzr TODO ########################################
            # trajs.add_point(task_name, traj_name, {"prompt": prompt, "obs": infos['observation_text'], "act": action, "preference": copy.deepcopy(infos['goal_condition_success_rate'][0])})
            # if (args.num_steps * j + step) % 10 == 0:
            #     print(f"\033[44m{trajs}\033[0m")
            #     trajs.save_to_file(f"./trajs/{task_name}.pkl")


            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])  # 创建掩码张量，用于指示是否结束当前回合。
            
            # 更新累积奖励。如果回合结束，记录每个任务的成功率，并重置回合。
            running_episode_rewards += reward.flatten()
            for i, d, r in zip(range(args.num_processes), done, reward):
                if d:
                    episode_rewards.append(running_episode_rewards[i].item())
                    # record success rate of different types of tasks
                    if "pick_and_place" in infos["extra.gamefile"][0]:
                        episode_succ_rate_pick_and_place.append(float(infos['won'][0]))
                    elif "pick_two_obj_and_place" in infos["extra.gamefile"][0]:
                        episode_succ_rate_pick_two_obj_and_place.append(float(infos['won'][0]))
                    elif "look_at_obj_in_light" in infos["extra.gamefile"][0]:
                        episode_succ_rate_look_at_obj_in_light.append(float(infos['won'][0]))
                    elif "pick_heat_then_place_in_recep" in infos["extra.gamefile"][0]:
                        episode_succ_rate_pick_heat_then_place_in_recep.append(float(infos['won'][0]))
                    elif "pick_cool_then_place_in_recep" in infos["extra.gamefile"][0]:
                        episode_succ_rate_pick_cool_then_place_in_recep.append(float(infos['won'][0]))
                    elif "pick_clean_then_place_in_recep" in infos["extra.gamefile"][0]:
                        episode_succ_rate_pick_clean_then_place_in_recep.append(float(infos['won'][0]))
                    # record the final success rate
                    episode_success_rate.append(float(infos['won'][0]))
                    episode_gc_success_rate.append(float(infos['goal_condition_success_rate'][0]))
                    print(len(episode_success_rate))
                    episode_action_tokens_log_prob.append(action_tokens_log_prob[i].item())
                    running_episode_rewards[i] = 0
                    obs, infos = envs.reset(seed=args.seed)
                    # print(f"\033[34mreset! {infos}\033[0m")
                    # return 0

                    ######################################## fzr TODO ########################################
                    #################### Traj Storage ####################
                    trajs.start_traj(infos['observation_text'][0])
                    # 重置轨迹存储🌟


                    # 重置环境后，重新生成提示。
                    admissible_commands = list(infos['admissible_commands'])[0]
                    history = trajs.get_history_data()  # jkc0904
                    # qs = get_alfworld_prompt(envs, obs = infos['observation_text'], admissible_actions=admissible_commands, action_only = args.action_only_prompt)
                    qs = get_dpo_prompt(envs, obs = infos['observation_text'], admissible_actions=admissible_commands, history=history, action_only = args.action_only_prompt)
                    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
                    conv = conv_templates[args.conv_mode].copy()
                    conv.append_message(conv.roles[0], qs)
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()
                    
            
            # 创建 bad_masks 张量，并确定动作 ID（在当前代码中未使用）。
            # bad_masks is a legact implementation in the storage
            bad_masks = torch.zeros(args.num_processes, 1)
            # action_id is also a legacy implementation in the storage, it is never used in the PPO update
            action_id = None
            for i in range(len(admissible_commands)):
                if admissible_commands[i] == action:
                    action_id = i
                    break
            if not action_id:
                action_id = 0
            action_id = torch.tensor(action_id)

            ######################################## fzr TODO ########################################
            trajs.add_new_state(obs, infos['observation_text'][0], text_action, float(infos['goal_condition_success_rate'][0]), prompt=None)
            # rollouts.insert(obs, output_id, action_id,
            #                     action_log_prob, value, reward, masks, bad_masks)  # 将当前观察、输出 ID、动作 ID、日志概率、价值、奖励、掩码和 bad_masks 插入到回合存储中。

        print(f"\033[43mUpdates:{j}\033[0m")
        print(f"\033[33mprompt:\033[0m{prompt}")
        print(f"\033[33maction_log_prob:\033[0m{action_log_prob}")
        print(f"\033[33mtext_action:\033[0m{text_action}")
        print(f"\033[33maction:\033[0m{action}")
        print(f"\033[33mground truth:\033[0m{infos}")
        print(f"\033[33msuccess_rate:\033[0m{np.mean(episode_success_rate)}")


        ##### 使用 DPO 算法更新策略，计算价值和动作损失以及策略的熵。并更新学习率调度器。#####
        # rollouts.compute_returns(next_value, args.use_gae, args.gamma,
        #                          args.gae_lambda, args.use_proper_time_limits)
        
        if trajs.valid_pairs >= training_args.start_training_pair_nums:
            action_loss = agent.update(trajs)
            lr_scheduler.step()


            # 更新后的回合存储。打印更新状态，包括奖励、成功率和其他统计信息。如果使用 wandb，则记录当前迭代的相关数据。
            ######################################## fzr TODO ########################################
            # rollouts.after_update() # TODO
            if len(episode_rewards) > 1:

                try:
                    rs.append(episode_success_rate)
                    np.save("./rewarddddddddddddddd.npy", np.array(rs))
                except Exception as e:
                    for _ in range(5):
                        print(f"\033[31m###############################\033[0m")
                    print(f"\033[43m{e}\033[0m")
                    for _ in range(5):
                        print(f"\033[31m###############################\033[0m")


                total_num_steps = (j + 1) * args.num_processes * args.num_steps
                end = time.time()
                print(
                    "\033[32mUpdates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.2f}/{:.2f}, min/max reward {:.2f}/{:.2f}, success_rate {:.2f}, loss: {:.2f}\n\033[0m"
                    .format(j, total_num_steps,
                            int(total_num_steps / (end - start)),
                            len(episode_rewards), np.mean(episode_rewards),
                            np.median(episode_rewards), np.min(episode_rewards),
                            np.max(episode_rewards), np.mean(episode_success_rate),
                            action_loss))
                if args.use_wandb:
                    wandb_images = [wandb.Image(image.cpu().numpy()) for image in obs]
                    text_table.add_data(j, infos['observation_text'][0], text_action)
                    wandb.log({"iteration": j,
                            "num_timesteps": total_num_steps,
                            "FPS": int(total_num_steps / (end - start)),
                            "episode_reward.mean": np.mean(episode_rewards),
                            "episode_reward.median": np.median(episode_rewards),
                            "episode_reward.min": np.min(episode_rewards),
                            "episode_reward.max": np.max(episode_rewards),
                            "episode_success_rate.mean": np.mean(episode_success_rate),
                            "episode_action_tokens_log_prob.mean": np.mean(episode_action_tokens_log_prob),
                            "episode_(goal_condition)_success_rate.mean": np.mean(episode_gc_success_rate),
                            "episode_succ_rate_pick_and_place.mean": np.mean(episode_succ_rate_pick_and_place),
                            "episode_succ_rate_pick_two_obj_and_place.mean": np.mean(episode_succ_rate_pick_two_obj_and_place),
                            "episode_succ_rate_look_at_obj_in_light.mean": np.mean(episode_succ_rate_look_at_obj_in_light),
                            "episode_succ_rate_pick_heat_then_place_in_recep.mean": np.mean(episode_succ_rate_pick_heat_then_place_in_recep),
                            "episode_succ_rate_pick_cool_then_place_in_recep.mean": np.mean(episode_succ_rate_pick_cool_then_place_in_recep),
                            "episode_succ_rate_pick_clean_then_place_in_recep.mean": np.mean(episode_succ_rate_pick_clean_then_place_in_recep),
                            "episode_num": len(episode_success_rate),
                            # "distribution_entropy": dist_entropy,
                            "text": text_table,
                            "image": wandb_images,
                            # "value.loss": value_loss,
                            "action.loss": action_loss,
                            "action_log_prob": action_log_prob.to('cpu').float().numpy()[0],
                            # "reward.max": rollouts.rewards.max().item(),
                            # "reward.min": rollouts.rewards.min().item(),
                            # "reward.mean": rollouts.rewards.mean().item(),
                            # "reward.std": rollouts.rewards.std().item(),
                            # "reward.median": rollouts.rewards.median().item(),
                            # "return.max": rollouts.returns.max().item(),
                            # "return.min": rollouts.returns.min().item(),
                            # "return.mean": rollouts.returns.mean().item(),
                            # "return.std": rollouts.returns.std().item(),
                            # "value.max": rollouts.value_preds.max().item(),
                            # "value.min": rollouts.value_preds.min().item(),
                            # "value.mean": rollouts.value_preds.mean().item(),
                            # "value.std": rollouts.value_preds.std().item(),
                            })
        else:
            print(f"\033[43m!!!Not Enough Pairs!!!\033[0m")



if __name__ == "__main__":
    main()
