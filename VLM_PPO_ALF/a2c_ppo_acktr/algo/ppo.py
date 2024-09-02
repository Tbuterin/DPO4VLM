import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import accelerate


class PPO():
    def __init__(self,
                 actor_critic,
                 optimizer,
                 accelerator,
                 clip_param,
                 ppo_epoch,
                 mini_batch_size,
                 value_loss_coef,
                 entropy_coef,
                 max_grad_norm=None,
                 use_clipped_value_loss=True):

        self.actor_critic = actor_critic

        self.mini_batch_size = mini_batch_size

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.clip_param = clip_param

        self.ppo_epoch = ppo_epoch

        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optimizer
        self.accelerator = accelerator

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        grad_step = 0
        self.actor_critic.train()
        for e in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(
                    advantages, self.mini_batch_size)
            for sample in data_generator:
                with self.accelerator.accumulate(self.actor_critic):
                    grad_step += 1
                    obs_batch, output_ids_batch, actions_batch, \
                    value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                            adv_targ = sample
                    # Reshape to do in a single forward pass for all steps
                    values, action_log_probs = self.actor_critic.evaluate_actions(
                        obs_batch, output_ids_batch)
                    #values and action_log_probs on two different devices!! because they come from two llava
                    if torch.isnan(action_log_probs).any():
                        continue
                    old_action_log_probs_batch = old_action_log_probs_batch.to(action_log_probs.device).view(-1)
                    adv_targ = adv_targ.to(action_log_probs.device)
                    value_preds_batch = value_preds_batch.to(values.device)
                    return_batch = return_batch.to(values.device)


                    ratio = torch.exp(action_log_probs -
                                    old_action_log_probs_batch)

                    surr1 = ratio * adv_targ
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                        1.0 + self.clip_param) * adv_targ
                    ## adding a ratio clip, inspired by https://github.com/huggingface/trl/blob/5a233546ee48532eaeb24b89b8d0042147574688/trl/trainer/ppo_trainer.py#L1199
                    if torch.any(ratio > 10):
                        action_loss = -surr2.mean()
                    else:
                        action_loss = -torch.min(surr1, surr2).mean()
                    if self.use_clipped_value_loss:
                        value_pred_clipped = value_preds_batch + \
                            (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                        value_losses = (values - return_batch).pow(2)
                        value_losses_clipped = (
                            value_pred_clipped - return_batch).pow(2)
                        value_loss = 0.5 * torch.max(value_losses,
                                                    value_losses_clipped).mean()
                    else:
                        value_loss = 0.5 * (return_batch - values).pow(2).mean()

                    try:
                        assert not torch.isnan(value_loss), "value_loss is nan"
                        assert not torch.isnan(action_loss), "action_loss is nan"
                    except:
                        print("value/action loss is nan")
                        exit(1)
                    loss = value_loss * self.value_loss_coef+action_loss
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.actor_critic.parameters(),
                            self.max_grad_norm
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    value_loss_epoch += value_loss.item()
                    action_loss_epoch += action_loss.item()

        value_loss_epoch /= grad_step
        action_loss_epoch /= grad_step
        dist_entropy_epoch /= grad_step

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch



# jkc240830: DPO
class DPO():
    def __init__(self,
                 policy_model,  # 修改: 原来的actor_critic改为policy_model，表示DPO中的策略模型  # @TODO: 这边直接采用VLM_Policy，后续需要优化掉Value model部分
                 reference_model,  # 添加: 参考模型，用于计算DPO损失
                 optimizer,
                 accelerator,
                 beta,  # 修改: PPO的clip_param替换为DPO的beta参数，用于控制DPO损失中的温度系数
                 dpo_epoch,  # 修改: ppo_epoch替换为dpo_epoch，表示DPO的训练轮数
                 mini_batch_size,
                 max_grad_norm=None,
                 label_smoothing=0.0,  # 添加: 标签平滑参数，用于DPO损失计算
                 reference_free=False):  # 添加: 是否使用参考模型的标志

        self.policy_model = policy_model  # 将actor_critic改为policy_model
        self.reference_model = reference_model  # 保存参考模型（添加reference_model属性）
        self.mini_batch_size = mini_batch_size
        self.beta = beta  # 将clip_param替换为beta
        self.dpo_epoch = dpo_epoch  # 将ppo_epoch替换为dpo_epoch
        self.label_smoothing = label_smoothing  # 添加label_smoothing属性
        self.reference_free = reference_free  # 添加reference_free属性

        self.optimizer = optimizer
        self.accelerator = accelerator
        self.max_grad_norm = max_grad_norm

    def update(self, rollouts):
        """
        进行DPO的参数更新。
        rollouts: 由RolloutStorage管理的收集的样本，包括成对的较好和较差的样本。
        """
        action_loss_epoch = 0  # DPO中不需要区分value和action loss，因此移除action_loss_epoch和dist_entropy_epoch
        # action_loss_epoch = 0
        grad_step = 0
        self.policy_model.train()  # 将actor_critic改为policy_model
        for e in range(self.dpo_epoch):
            data_generator = rollouts.feed_forward_generator(self.mini_batch_size)  # @TODO
            for better_sample, worse_sample in data_generator:
                with self.accelerator.accumulate(self.policy_model):  # 将actor_critic替换为policy_model
                    grad_step += 1
                    better_obs_batch, better_output_ids_batch = better_sample
                    worse_obs_batch, worse_output_ids_batch = worse_sample

                    # 评估策略模型在较好和较差样本上的对数概率
                    better_log_probs = self.policy_model.evaluate_actions(better_obs_batch, better_output_ids_batch) # obs是图片，ids是文本
                    worse_log_probs = self.policy_model.evaluate_actions(worse_obs_batch, worse_output_ids_batch)  # @TODO: 这两个是要加input_IDS的


                    # Forward pass for reference model (or use precomputed reference log probs)
                    if self.reference_free:
                        reference_chosen_logps = torch.zeros_like(better_log_probs)  # reference_free模式下，参考模型的log probs为0
                        reference_reject_logps = torch.zeros_like(worse_log_probs)
                    else:
                        reference_chosen_logps = self.reference_model.evaluate_actions(better_obs_batch, better_output_ids_batch)
                        reference_reject_logps = self.reference_model.evaluate_actions(worse_obs_batch, worse_output_ids_batch)

                    # 计算DPO损失
                    losses, chosen_rewards, rejected_rewards = self.dpo_loss(
                        better_log_probs,
                        worse_log_probs,
                        reference_better_log_probs,
                        reference_worse_log_probs,
                        self.beta,
                        self.label_smoothing,
                        self.ipo,
                        self.reference_free
                    )

                    # 检查数据合法性
                    try:
                        assert not torch.isnan(loss).any(), "loss contains nan"
                    except:
                        print("\033[31mloss contains nan\033[0m")
                        exit(1)


                    loss = losses.mean()

                    # 反向传播和梯度更新
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.policy_model.parameters(),
                            self.max_grad_norm
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    action_loss_epoch += loss.item()

        action_loss_epoch /= grad_step

        return action_loss_epoch

    def dpo_loss(self, policy_chosen_logps, policy_rejected_logps,
                        reference_chosen_logps, reference_rejected_logps,
                        beta, label_smoothing=0.0, ipo=False, reference_free=False):
        """
        计算DPO损失函数
        policy_chosen_logps: 策略模型对选择样本的对数概率
        policy_rejected_logps: 策略模型对未选择样本的对数概率
        reference_chosen_logps: 参考模型对选择样本的对数概率
        reference_rejected_logps: 参考模型对未选择样本的对数概率
        beta: DPO损失的温度参数
        label_smoothing: DPO损失的标签平滑参数
        ipo: 是否使用IPO损失
        reference_free: 是否忽略参考模型
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        if self.reference_free:
            ref_logratios = 0  # 如果不使用参考模型

        logits = pi_logratios - ref_logratios

        # 计算DPO损失
        losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing

        chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards

