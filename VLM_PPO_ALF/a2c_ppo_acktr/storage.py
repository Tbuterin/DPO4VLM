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




# 240825tra
class TrajStorage:
    def __init__(self):
        self.trajectories = {}  # 存储所有轨迹的字典

    def start_trajectory(self, trajectory_id):
        """开始一条新的轨迹"""
        if trajectory_id in self.trajectories:
            print(f"轨迹 {trajectory_id} 已经存在。")
        else:
            self.trajectories[trajectory_id] = []

    def add_point(self, trajectory_id, point):
        """向指定轨迹添加数据点"""
        if trajectory_id not in self.trajectories:
            print(f"轨迹 {trajectory_id} 不存在。")
        else:
            self.trajectories[trajectory_id].append(point)

    def get_trajectory(self, trajectory_id):
        """获取指定轨迹的全部数据"""
        return self.trajectories.get(trajectory_id, [])

    def get_all_trajectories(self):
        """获取所有轨迹"""
        return self.trajectories

    def delete_trajectory(self, trajectory_id):
        """删除指定的轨迹"""
        if trajectory_id in self.trajectories:
            del self.trajectories[trajectory_id]
        else:
            print(f"轨迹 {trajectory_id} 不存在。")

# 使用示例
if __name__ == "__main__":
    traj_storage = TrajStorage()

    # 开始新的轨迹
    traj_storage.start_trajectory("traj1")

    # 添加数据点
    traj_storage.add_point("traj1", {"x": 1, "y": 2})
    traj_storage.add_point("traj1", {"x": 2, "y": 3})
    traj_storage.add_point("traj1", {"x": 3, "y": 4})

    # 获取指定轨迹的全部数据
    print("轨迹 traj1 的全部数据:")
    trajectory = traj_storage.get_trajectory("traj1")
    for point in trajectory:
        print(point)

    # 获取所有轨迹
    print("\n所有轨迹的数据:")
    all_trajectories = traj_storage.get_all_trajectories()
    for traj_id, points in all_trajectories.items():
        print(f"轨迹 {traj_id} 的数据点数量: {len(points)}")

    # 删除轨迹
    traj_storage.delete_trajectory("traj1")
    print("\n删除轨迹 traj1 后，所有轨迹的数据:")
    all_trajectories = traj_storage.get_all_trajectories()
    for traj_id, points in all_trajectories.items():
        print(f"轨迹 {traj_id} 的数据点数量: {len(points)}")
