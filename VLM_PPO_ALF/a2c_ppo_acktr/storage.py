import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


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
import copy
import pickle
from dataclasses import dataclass, field
from datasets import Dataset, DatasetDict
import pandas as pd
class TrajStorage:
    def __init__(self):
        self.tasks = {}  # 存储所有任务的字典，任务名是键，对应轨迹的字典是值

    def start_task(self, task_id):
        """开始一个新的任务"""
        if task_id in self.tasks:
            print(f"任务 {task_id} 已经存在。")
        else:
            self.tasks[task_id] = {}

    def start_trajectory(self, task_id, trajectory_id):
        """在指定任务下开始一条新的轨迹"""
        if task_id not in self.tasks:
            print(f"任务 {task_id} 不存在。")
        elif trajectory_id in self.tasks[task_id]:
            print(f"轨迹 {trajectory_id} 已经存在于任务 {task_id} 中。")
        else:
            self.tasks[task_id][trajectory_id] = []

    def add_point(self, task_id, trajectory_id, point):
        """向指定任务下的轨迹添加数据点"""
        if task_id not in self.tasks:
            print(f"任务 {task_id} 不存在。")
        elif trajectory_id not in self.tasks[task_id]:
            print(f"轨迹 {trajectory_id} 不存在于任务 {task_id} 中。")
        else:
            self.tasks[task_id][trajectory_id].append(copy.deepcopy(point))

    def get_trajectory(self, task_id, trajectory_id):
        """获取指定任务下的轨迹的全部数据"""
        if task_id in self.tasks and trajectory_id in self.tasks[task_id]:
            return self.tasks[task_id][trajectory_id]
        else:
            return []

    def get_all_tasks(self):
        """获取所有任务及其轨迹"""
        return self.tasks

    def delete_trajectory(self, task_id, trajectory_id):
        """删除指定任务下的轨迹"""
        if task_id in self.tasks and trajectory_id in self.tasks[task_id]:
            del self.tasks[task_id][trajectory_id]
        else:
            print(f"任务 {task_id} 或轨迹 {trajectory_id} 不存在。")

    def delete_task(self, task_id):
        """删除指定的任务及其所有轨迹"""
        if task_id in self.tasks:
            del self.tasks[task_id]
        else:
            print(f"任务 {task_id} 不存在。")
    
    def to(self, device):
        """将所有轨迹数据转移到指定设备"""
        for task_id, trajectories in self.tasks.items():
            for trajectory_id, points in trajectories.items():
                self.tasks[task_id][trajectory_id] = [point.to(device) for point in points]
    
    def save_to_file(self, file_path):
        """将所有任务及其轨迹保存到本地文件"""
        with open(file_path, 'wb') as f:
            pickle.dump(self.tasks, f)
        print(f"\033[32m数据已保存到 {file_path}\033[0m")

    def load_from_file(self, file_path):
        """从本地文件加载任务及其轨迹"""
        with open(file_path, 'rb') as f:
            self.tasks = pickle.load(f)
        print(f"\033[32m数据已从 {file_path} 加载")

    def to_dataset_dict(self):
        """将所有任务及其轨迹转换为DatasetDict格式"""
        dataset_dict = {}
        for task_id, trajectories in self.tasks.items():
            data = []
            for trajectory_id, points in trajectories.items():
                for point in points:
                    data.append({
                        "task_id": task_id,
                        "trajectory_id": trajectory_id,
                        "point": point
                    })
            dataset = Dataset.from_pandas(pd.DataFrame(data))
            # 这里的键名可以根据需要调整，例如使用任务ID
            dataset_dict[task_id] = dataset
        return DatasetDict(dataset_dict)



if __name__ == "__main__":
    traj_storage = TrajStorage()

    # 开始新任务
    traj_storage.start_task("task1")

    # 在任务下开始新的轨迹
    traj_storage.start_trajectory("task1", "traj1")

    # 添加数据点
    traj_storage.add_point("task1", "traj1", {"step": 1, "obs": "you are in a bedroom"})
    traj_storage.add_point("task1", "traj1", {"step": 2, "obs": "you are in a livingroom"})

    # 获取指定任务下的轨迹的全部数据
    print("任务 task1 下的轨迹 traj1 的全部数据:")
    trajectory = traj_storage.get_trajectory("task1", "traj1")
    for point in trajectory:
        print(point)
    data = traj_storage.to_dataset_dict()
    print(f"\033[33m{data}\033[0m")
