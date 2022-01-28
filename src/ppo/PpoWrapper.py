import random

import torch
from tqdm import trange

from logging_callbacks.wandbLogger import preprocess_logs
from src.common import mas_dict2tensor, Params
from .PPO import PPO
from .RolloutStorage import RolloutStorage
from ..model.ModelFree import ModelFree


class PpoWrapper:
    def __init__(self, env, config: Params):

        self.env = env
        self.obs_shape = env.obs_shape
        self.action_space = env.action_space
        self.num_agents = config.agents

        self.gamma = config.gamma
        self.device = config.device

        self.num_steps = config.horizon
        self.num_minibatch = config.minibatch

        self.guided_learning_prob = config.guided_learning_prob

        policy_configs = config.get_model_free_configs()
        self.base_hidden_size = config.base_hidden_size

        self.actor_critic_dict = {
            agent_id: ModelFree(**policy_configs).to(self.device) for agent_id in self.env.agents
        }
        # epochs = config.epochs,
        self.ppo_agent = PPO(
            actor_critic_dict=self.actor_critic_dict,
            ppo_epochs=config.batch_epochs,
            clip_param=config.ppo_clip_param,
            num_minibatch=self.num_minibatch,
            value_loss_coef=config.value_loss_coef,
            entropy_coef=config.entropy_coef,
            lr=config.lr,
            eps=config.eps,
            max_grad_norm=config.max_grad_norm,
            use_clipped_value_loss=config.clip_value_loss
        )

        self.use_wandb = config.use_wandb
        if self.use_wandb:

            cams = []

            from logging_callbacks import PPOWandb

            self.logger = PPOWandb(
                train_log_step=5,
                val_log_step=5,
                project="model_free",
                opts={},
                models=self.actor_critic_dict["agent_0"].get_modules(),
                horizon=config.horizon,
                action_meaning=self.env.env.action_meaning_dict,
                cams=cams,
            )

    def get_learning_rate(self):
        lrs = []
        for k, optim in self.ppo_agent.optimizers.items():
            param_group = optim.param_groups[0]
            lrs.append(param_group['lr'])

        return sum(lrs) / len(lrs)

    def set_guided_learning_prob(self, value):
        self.guided_learning_prob = value

    def set_entropy_coeff(self, value):
        self.ppo_agent.entropy_coef = value

    def learn(self, epochs):
        rollout = RolloutStorage(
            num_steps=self.num_steps,
            obs_shape=self.obs_shape,
            num_actions=self.action_space,
            num_agents=self.num_agents,
            # recurrent_hs_size=self.actor_critic_dict["agent_0"].recurrent_hidden_state_size
        )
        rollout.to(self.device)

        schedulers = self.init_schedulers(
            epochs,
            use_curriculum=False,
            use_guided_learning=False,
            use_learning_rate=False,
            use_entropy_reg=False
        )

        # init dicts
        action_dict = {agent_id: False for agent_id in self.env.agents}
        values_dict = {agent_id: False for agent_id in self.env.agents}
        action_log_dict = {agent_id: False for agent_id in self.env.agents}
        #recurrent_hs_dict = {agent_id: False for agent_id in self.env.agents}

        for epoch in trange(epochs, desc="Training model free"):
            self.ppo_agent.eval()

            for s in schedulers:
                s.update_step(epoch)

            logs = {ag: dict(
                ratio=[],
                surr1=[],
                surr2=[],
                returns=[],
                adv_targ=[],
                perc_surr1=[],
                perc_surr2=[],
                curr_log_probs=[],
                old_log_probs=[]
            ) for ag in self.actor_critic_dict.keys()}

            observation = self.env.reset()
            rollout.states[0] = observation.unsqueeze(dim=0)

            for step in range(self.num_steps):
                obs = observation.to(self.device).unsqueeze(dim=0)
                guided_learning = {
                    agent_id: False for agent_id in self.env.agents}

                for agent_id in self.env.agents:
                    # perform guided learning with scheduler
                    # todo: remove optimal end generalize with policy
                    if self.guided_learning_prob > random.uniform(0, 1):
                        action, action_log_prob = self.env.optimal_action(
                            agent_id)
                        guided_learning[agent_id] = True
                        value = -1
                    else:
                        # FIXED: NORMALIZED THE STATE
                        with torch.no_grad():
                            value, action, action_log_prob = self.actor_critic_dict[agent_id].act(
                                obs, rollout.masks[step]
                            )

                    # get action with softmax and multimodal (stochastic)
                    action_dict[agent_id] = action
                    values_dict[agent_id] = value
                    action_log_dict[agent_id] = action_log_prob
                    #recurrent_hs_dict[agent_id] = recurrent_hs[0]

                # Obser reward and next obs
                # fixme: questo con multi agent non funziona, bisogna capire come impostarlo
                observation, rewards, done, infos = self.env.step(action_dict)

                # if guided then use actual reward as predicted value
                for agent_id, b in guided_learning.items():
                    if b:
                        values_dict[agent_id] = rewards[agent_id]

                # FIXME mask dovrebbe avere un valore per ogni agente
                masks = (~torch.tensor(done["__all__"])).float().unsqueeze(0)
                rewards = mas_dict2tensor(rewards, float)
                actions = mas_dict2tensor(action_dict, int)
                values = mas_dict2tensor(values_dict, float)
                #recurrent_hs = mas_dict2tensor(recurrent_hs_dict, list)
                action_log_probs_list = [elem.unsqueeze(dim=0) for _, elem in action_log_dict.items()]
                action_log_probs = torch.cat(action_log_probs_list, 0)

                rollout.insert(
                    state=observation,
                    # recurrent_hs=recurrent_hs,
                    action=actions,
                    action_log_probs=action_log_probs,
                    value_preds=values,
                    reward=rewards,
                    mask=masks
                )

                if done["__all__"]:
                    observation = self.env.reset()

            # fixme: qui bisogna come farlo per multi agent
            with torch.no_grad():
                next_value = self.actor_critic_dict["agent_0"].get_value(
                    rollout.states[-1].unsqueeze(dim=0), rollout.masks[-1]
                ).detach()

            rollout.compute_returns(next_value, True, self.gamma, 0.95)

            self.ppo_agent.train()
            with torch.enable_grad():
                value_loss, action_loss, entropy = self.ppo_agent.update(
                    rollout, logs)

            if self.use_wandb:
                logs = preprocess_logs(
                    [value_loss, action_loss, entropy, logs], self)
                self.logger.on_batch_end(
                    logs=logs, batch_id=epoch, rollout=rollout)

            rollout.after_update()

        return

    def set_env(self, env):
        self.env = env

    def init_schedulers(self, epochs, use_curriculum: bool = True, use_guided_learning: bool = True,
                        use_learning_rate: bool = True, use_entropy_reg: bool = True):
        schedulers = []

        if use_curriculum:
            from common.schedulers import CurriculumScheduler

            curriculum = {
                400: dict(reward=0, landmark=1),
                600: dict(reward=1, landmark=0),
                800: dict(reward=1, landmark=1),
                900: dict(reward=0, landmark=2),
                1100: dict(reward=1, landmark=2),
                1300: dict(reward=2, landmark=2),
            }

            cs = CurriculumScheduler(
                epochs=epochs,
                values_list=list(curriculum.values()),
                set_fn=self.env.set_strategy,
                step_list=list(curriculum.keys()),
                get_curriculum_fn=self.env.get_curriculum
            )
            schedulers.append(cs)

        if use_guided_learning:
            from common.schedulers import linear_decay, StepScheduler

            guided_learning = {
                100: 0.8,
                200: 0.7,
                400: 0.6,
                600: 0.4,
                800: 0.2,
                900: 0.0,
                1200: 0.4,
                1400: 0.2,
                1600: 0.1,
                1700: 0.0,
            }

            ep = int(epochs * 0.8)
            guided_learning = linear_decay(start_val=1, epochs=ep)

            gls = StepScheduler(
                epochs=ep,
                values_list=guided_learning,
                set_fn=self.set_guided_learning_prob
            )
            schedulers.append(gls)

        if use_learning_rate:
            from torch.optim.lr_scheduler import ExponentialLR
            from common.schedulers import LearningRateScheduler

            kwargs = dict(gamma=0.997)
            lrs = LearningRateScheduler(
                base_scheduler=ExponentialLR,
                optimizer_dict=self.ppo_agent.optimizers,
                scheduler_kwargs=kwargs
            )
            schedulers.append(lrs)

        if use_entropy_reg:
            from common.schedulers import exponential_decay, StepScheduler

            values = exponential_decay(
                Params().entropy_coef, epochs, gamma=0.999)
            es = StepScheduler(
                epochs=epochs,
                values_list=values,
                set_fn=self.set_entropy_coeff
            )
            schedulers.append(es)

        return schedulers
