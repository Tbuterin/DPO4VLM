import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tqdm import tqdm


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space, max_new_tokens):
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        #hard-code to cases of max_new_tokens being smaller than 32
        self.output_ids = torch.zeros(
            num_steps, num_processes, 2*max_new_tokens).long()
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.output_ids = self.output_ids.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)

    def insert(self, obs, output_ids, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks):
        self.obs[self.step + 1].copy_(obs)
        self.output_ids[self.step].copy_(output_ids)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(self,
                        next_value,
                        use_gae,
                        gamma,
                        gae_lambda,
                        use_proper_time_limits=True):
        if use_proper_time_limits:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[
                        step + 1] * self.masks[step +
                                               1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step +
                                                                  1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = (self.returns[step + 1] * \
                        gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                        + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[
                        step + 1] * self.masks[step +
                                               1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step +
                                                                  1] * gae
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = self.returns[step + 1] * \
                        gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self,
                               advantages,
                               mini_batch_size=None):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            actions_batch = self.actions.view(-1,
                                              self.actions.size(-1))[indices]
            output_ids_batch = self.output_ids.view(-1,
                                              self.output_ids.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1,
                                                                    1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, output_ids_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ




# 240825tra # 20240830dic
def find_first_diff(list1, list2):
    # @TODO: 考虑最后一个输入的动作
    # @TODO: 考虑相同动作不同状态转移🌟    
    # 确保输入是列表
    if not isinstance(list1, list) or not isinstance(list2, list):
        raise TypeError("输入必须是列表")
    
    # 找到最短的列表长度
    min_length = min(len(list1), len(list2))

    # 遍历至最短列表的长度，比较每个位置的第二个元素 (text_obs)
    for i in range(min_length):
        if list1[i][1] != list2[i][1]:
            return i
    
    # 如果没有发现不同返回 -1
    return -1

def compare_2_trajs(ta, tb):
    """
    ta & tb: List(
        [obs, text_obs, text_action, success_rate]
    )
    """
    # 定位两个列表出现第一个text_obs元素不同的坐标
    step_idx = find_first_diff(ta, tb)
    
    if step_idx == -1 or step_idx == 0:  # 如果轨迹相同或者第一个就不同，都没有意义 @TODO: 考虑最后一个point动作不同🌟
        return ["same", -1]
    
    reward_a, reward_b = float(ta[-1][3]), float(tb[-1][3])
    if reward_a > reward_b:
        return ["better", step_idx]
    elif reward_a < reward_b:
        return ["worse", step_idx]
    else:
        return ["same", step_idx]


def get_preference_data(preference, diff_idx, traA, traB, history_horizon=3):
    """
    这个函数用于将轨迹对转换成prompt + chosen/rejected action的方式返回
    Input:
        preference: str "better","worse"
        diff_idx: int 第一个不同元素的索引
        traA & traB: List([obs, text_obs, text_action, success_rate, prompt], ...)
    
    Output:
        pre_prompt
        pre_better
        pre_worse
        obs
    """
    # @TODO: 这里只考虑了text_obs相同，历史动作就一定相同的情况🌟
    # 真正动作不同的应该是第diff_idx - 1个动作
    if diff_idx > 1:
        text_obs_action_pairs = [arr[1] + "\n" + arr[2] for arr in traA[:diff_idx - 2]]
    else:
        text_obs_action_pairs = []
    text_obs_action_pairs = text_obs_action_pairs[-history_horizon:]
    text_obs_action_pairs.append(traA[diff_idx - 1][4]) # 该步的prompt要添加
    pre_prompt_text = '\n'.join(text_obs_action_pairs)

    if preference == "better":
        pre_better_text = traA[diff_idx - 1][2]
        pre_worse_text = traB[diff_idx - 1][2]
    else:
        pre_better_text = traB[diff_idx - 1][2]
        pre_worse_text = traA[diff_idx - 1][2]
    obs = traA[diff_idx - 1][0]
    # print(pre_prompt_text, pre_better_text, pre_worse_text)
    return pre_prompt_text, pre_better_text, pre_worse_text, obs
    
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
class TrajBuffer(object):
    def __init__(self, max_pairs, num_processes, max_history_tokens, max_new_tokens, obs_shape, history_horizon=3):
        """
        better_sample = better_obs_batch, better_output_ids_batch
        """
        self.max_pairs = max_pairs
        self.max_history_tokens = max_history_tokens
        self.max_new_tokens = max_new_tokens
        self.buffer = {}
        self.current_init_state = None
        self.current_traj_index = 0
        self.pre_prompt = torch.zeros(max_pairs, num_processes, 2*max_history_tokens).long()
        self.pre_better = torch.zeros(max_pairs, num_processes, 2*max_new_tokens).long()
        self.pre_better_obs = torch.zeros(max_pairs, num_processes, *obs_shape)
        self.pre_worse = torch.zeros(max_pairs, num_processes, 2*max_new_tokens).long()
        self.pre_worse_obs = torch.zeros(max_pairs, num_processes, *obs_shape)  # @TODO: 看看obs是否一致

        self.history_horizon = history_horizon
        self.valid_pairs = 0  # 这个变量用于存储既有数据的数量
        self.saving_index = 0  # 这个变量用于循环更新buffer的存储变量
    
    def start_traj(self, init_text_obs=None):
        """
        这个函数的作用是启动一个新的轨迹。
        在main_alf.py运行时如果接收到done=True, 或开始新的update循环, 则调用此函数。
        根据轨迹的init_observation_text, 如果存在则添加新轨迹, 不存在则创建相应的KEY。
        更新当前轨迹的KEY和Index (Index用于定位trajs的List(), 指的是在trajs中的第几个traj)。
        Input: init_observation_text (Str)
        Output: -
        """
        self.current_init_state = init_text_obs
        if init_text_obs in self.buffer:
            self.buffer[init_text_obs].append([])
            self.current_traj_index = len(self.buffer[init_text_obs]) - 1
        else:
            self.buffer[init_text_obs] = [[],]
            self.current_traj_index = 0

    def get_history_data(self):
        """
        这个函数的作用是用于prompt生成的时候引入历史轨迹信息
        Input: -
        Output: Str 由历史text_obs和text_action组成文本段落
        """
        try:
            if len(self.buffer[self.current_init_state][self.current_traj_index]) > 0:  # jkc0904
                text_obs_action_pairs = ["text_observation: " + arr[1] + "\naction: " + arr[2] for arr in self.buffer[self.current_init_state][self.current_traj_index][-self.history_horizon:]]
            else:
                return ""
            text_history = '\n'.join(text_obs_action_pairs)
        except Exception as e:
            print(f"\033[31m{e}, please start a trajectory first.\033[0m")
            exit(1)
        return text_history

    def add_new_state(self, obs, text_obs, text_action, success_rate, prompt=None):
        """
        这个函数的作用是加入一组数据到当前轨迹
        Input:
            obs: tensor([1, 300, 300, 3])
            text_obs: str
            text_action: str
            success_rate: float(1)
            prompt: str 这个是每一步所使用的prompt
        Output: -
        """
        self.buffer[self.current_init_state][self.current_traj_index].append([obs, text_obs, text_action, success_rate, prompt])
    
    def add_test_state(self, tokenizer):
        self.start_traj("test")
        self.add_new_state(torch.rand(300, 300, 3), "i see the environment", "look", 0.03, "")
        self.add_new_state(torch.rand(300, 300, 3), "i see table 2", "go to table 2", 0.08, "")
        self.start_traj("test")
        self.add_new_state(torch.rand(300, 300, 3), "i see the environment is big", "asd", 0.03, "")
        self.add_new_state(torch.rand(300, 300, 3), "nothing happens", "go to table 2", 0.0, "")
        self.get_pairs_data(tokenizer)
    
    def get_pairs_data(self, tokenizer):
        """
        构造样本对数据
        从self.buffer遍历读取相同init_state的轨迹, 构造样本对;
        存储到self.pre_prompt、self.pre_better、self.pre_better_obs和self.pre_worse、self.pre_worse_obs中。
        Input: tokenizer
        Output: _
        """
        # 遍历字典buffer中所有初始状态相同的轨迹对
        for init_stat, trajs in tqdm(self.buffer.items()):
            for i in range(len(trajs) - 1, 0, -1):
                for j in range(i - 1, -1, -1):
                    preference, diff_idx = compare_2_trajs(trajs[i], trajs[j])
                    if preference == "same": 
                        print("same")
                        continue
                    print("valid")
                    # print("valid")
                    pre_prompt_text, pre_better_text, pre_worse_text, obs = get_preference_data(preference, diff_idx, trajs[i], trajs[j], history_horizon=self.history_horizon)
                    pre_prompt = tokenizer_image_token(pre_prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
                    pre_prompt[pre_prompt == 0] = 259 # 869: . (period), 29871: SPIECE, 259: whitespace
                    # print(pre_prompt.size())

                    pre_better = tokenizer(pre_better_text).input_ids
                    pre_worse = tokenizer(pre_worse_text).input_ids

                    # 广播到max_new_tokens的长度
                    if len(pre_better) < 2 * self.max_new_tokens:
                        pre_better += [0] * (2 * self.max_new_tokens - len(pre_better))
                    if len(pre_worse) < 2 * self.max_new_tokens:
                        pre_worse += [0] * (2 * self.max_new_tokens - len(pre_worse))

                    if pre_prompt.size()[-1] < 2 * self.max_history_tokens:
                        pre_prompt = torch.cat((pre_prompt, torch.zeros(1, 2 * self.max_history_tokens - pre_prompt.size()[-1])), dim=1)

                    pre_better = pre_better[:2 * self.max_new_tokens]
                    pre_worse = pre_worse[:2 * self.max_new_tokens]
                    pre_prompt = pre_prompt[:, -2 * self.max_history_tokens:]  # jkc0904: 获取prompt中靠后的历史🌟  # @TODO: need to check

                    self.pre_prompt[self.saving_index % self.max_pairs].copy_(pre_prompt)

                    self.pre_better[self.saving_index % self.max_pairs].copy_(torch.tensor(pre_better))
                    self.pre_worse[self.saving_index % self.max_pairs].copy_(torch.tensor(pre_worse))
                    self.pre_better_obs[self.saving_index % self.max_pairs].copy_(obs)
                    self.pre_worse_obs[self.saving_index % self.max_pairs].copy_(obs)

                    if self.valid_pairs < self.max_pairs:
                        self.valid_pairs += 1  # 更新存储数据量
                    self.saving_index += 1  # 更新循环存储变量
                    print(self.valid_pairs)

        
    def feed_forward_generator(self, mini_batch_size=None):
        """
        由self.pre_prompt、self.pre_better和self.pre_worse读取并yield数据
        Output: prompt, better_sample和worse_sample的样本生成器--generator()
        """
        num_samples = self.valid_pairs


        sampler = BatchSampler(
            SubsetRandomSampler(range(num_samples)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:
            pre_obs_batch = self.pre_better_obs[:-1].view(-1, *self.pre_better_obs.size()[2:])[indices]
            pre_prompt_batch = self.pre_prompt[:-1].view(-1, self.pre_prompt.size()[-1])[indices]
            pre_better_batch = self.pre_better[:-1].view(-1, self.pre_better.size()[-1])[indices]
            pre_worse_batch = self.pre_worse[:-1].view(-1, self.pre_worse.size()[-1])[indices]
            
            
            yield pre_obs_batch, pre_prompt_batch, pre_better_batch, pre_worse_batch




if __name__ == "__main__":
    import random
    import string
    import torch
    def generate_random_string(length):
        letters = string.ascii_letters  # 包含所有大小写字母
        return ''.join(random.choice(letters) for i in range(length))


    buffer = TrajBuffer(20, 1, 100, 50, (300, 300, 3))
    buffer.start_traj("haha")
    print(buffer.get_history_data(), "<<!!!!!!!!!!!")
    # max_pairs, num_processes, max_history_tokens, max_new_tokens, obs_shape
    # for i in range(5):
    #     init_stat = generate_random_string(6)
    #     for x in range(6):
    #         buffer.start_traj(init_stat)
    #         obs = torch.rand(5, 5, 3)
    #         buffer.add_new_state(obs, init_stat, generate_random_string(3), float(random.random()), generate_random_string(random.randint(2,7)))
    #         for j in range(7):
    #             obs = torch.rand(5, 5, 3)
    #             text_obs = generate_random_string(14)
    #             success_rate = float(random.random())
    #             text_action = generate_random_string(3)
    #             prompt = generate_random_string(random.randint(2,7))
    #             buffer.add_new_state(obs, text_obs, text_action, success_rate, prompt)
    

    # print(buffer.buffer.keys())
    # print(buffer.buffer)
    # print(buffer.get_history_data())

    from transformers import AutoTokenizer

    # 使用预训练模型名称来加载一个基础的tokenizer
    # 这里我们使用BERT的预训练模型
    tokenizer = AutoTokenizer.from_pretrained("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/jiaokechen/Qwen2-7B-Instruct")
    # print(tokenizer("sadasd 000 0.213123123").input_ids)
    # print(tokenizer("100!0 0").input_ids)
    # print(tokenizer("!").input_ids)

    # buffer.get_pairs_data(tokenizer)
    buffer.add_test_state(tokenizer)
    # print(f"\033[32m{buffer.pre_prompt}\033[0m")

    rollout = buffer.feed_forward_generator(1)
    for obs, pt, pre, rej in rollout:
        print(f"\033[34m{obs.size()}, \033[0m{pt.size()}, \033[32m{pre}, \033[33m{rej.size()}\033[0m\n\n")

