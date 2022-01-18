from typing import Tuple

import numpy as np

from PettingZoo.pettingzoo.mpe._mpe_utils.core import Agent, Entity, World
from PettingZoo.pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from src.common import min_max_norm, is_collision, get_distance
from .TimerLandmark import TimerLandmark


class Border(Entity):

    def __init__(self, start: Tuple[int, int], end: Tuple[int, int], name=None, color=(1, 0, 0), linewidth=1):
        super(Border, self).__init__()
        self.start = np.array(start)
        self.end = np.array(end)
        self.color = np.array(color)
        self.linewidth = linewidth
        self.name = name

        p_pos = (
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2,
        )

        self.state.p_pos = np.array(p_pos)


class BoundedWorld(World):
    """
    A bounded word for the env. Uses unmovable landmarks on the border of the image in order to bound the agents
    """

    def __init__(self, max_size: int):
        super(BoundedWorld, self).__init__()

        self.max_size = max_size
        max_size -= 0.1

        b1 = Border((-max_size, -max_size), (max_size, -max_size), "border_0")
        b2 = Border((-max_size, -max_size), (-max_size, max_size), "border_1")
        b3 = Border((max_size, -max_size), (max_size, max_size), "border_2")
        b4 = Border((max_size, max_size), (-max_size, max_size), "border_3")

        self.borders = [b1, b2, b3, b4]
        self.contact_margin = 0.1

    @property
    def entities(self):
        return self.landmarks + self.borders + self.agents

    # @property
    # def contact_margin(self):
    #     if len(self.entities) == 0:
    #         return self._contact_margin
    #     else:
    #         min_size = [ent.size for ent in self.entities]
    #         min_size = min(min_size)
    #         return min_size*100
    #
    # @contact_margin.setter
    # def contact_margin(self, value):
    #
    #     self._contact_margin = value

class CollectLandmarkScenario(BaseScenario):
    def __init__(
            self,
            num_agents: int,
            num_landmarks: int,
            max_size: int,
            landmark_reward: int = 1,
            max_landmark_counter: int = 4,
            landmark_penalty: int = 2,
            task: str = "simple",
            agent_size: int = 0.1,
            landmark_size: int = 0.3,
            step_reward: int =0
    ):
        """

        Args:
            num_agents: the number of agents in the system
            num_landmarks: the number of landmarks
            max_size: maximum size of the word, used in word initialization
            landmark_reward: the reward value (positive) for visiting a landmark
            max_landmark_counter: At each timestep a landmarks increases its penalty until step>=max_landmark_counter,
                    then it stays the same until visited
            landmark_penalty: the penalty value (negative) for not visited landmarks
            task: task to follow
            agent_size:
            landmark_size:
        """
        self.registered_collisions = {}
        self.landmarks = {}
        self.max_landmark_counter = max_landmark_counter
        self.landmark_reward = landmark_reward
        self.landmark_penalty = landmark_penalty
        self.num_agents = num_agents
        self.num_landmarks = num_landmarks
        self.max_size = max_size
        self.task = task
        self.agent_size = agent_size
        self.landmark_size = landmark_size

        self.step_reward = step_reward

        (
            self.reward_curriculum,
            self.landmark_curriculum,
        ) = self.init_curriculum_learning()

    @staticmethod
    def init_curriculum_learning():

        reward_modalities = {
            0: "Reward is the (world.maxsize - distance between agent and closest landmark), +landmark_reward when agent on landmark",
            1: "Reward is 0 at every time step and +landmark_reward when agent on landmark",
            2: "Reward is -step_reward at every time step and +landmark_reward when agent on landmark",
            "current": 0,
        }

        landmark_modalities = {
            0: "Landmark have static dimension and positions",
            1: "Landmark have static dimension and random positions",
            2: "Landmark have random dimension and position",
            "current": 0,
        }

        return reward_modalities, landmark_modalities

    def set_curriculum(self, reward: int = None, landmark: int = None):
        if reward is not None:
            assert (
                    reward in self.reward_curriculum.keys()
            ), f"Reward curriculum modality '{reward}' is not in range"
            self.reward_curriculum["current"] = reward

        if landmark is not None:
            assert (
                    landmark in self.landmark_curriculum.keys()
            ), f"Landmark curriculum modality '{landmark}' is not in range"
            self.landmark_curriculum["current"] = landmark

    def get_curriculum(self) -> Tuple[Tuple[int, str], Tuple[int, str]]:
        r = self.reward_curriculum["current"]
        l = self.landmark_curriculum["current"]

        return (r, self.reward_curriculum[r]), (l, self.landmark_curriculum[l])

    def make_world(self) -> World:
        """
        Init world and populate it with agents and landmarks
        Returns:

        """
        world = BoundedWorld(self.max_size)

        # set any world properties first
        world.dim_c = 2

        # add agents
        world.agents = [Agent() for _ in range(self.num_agents)]

        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.collide = True
            agent.silent = True
            agent.size = self.agent_size
            agent.accel = 4.0
            agent.max_speed = 1.3
            agent.color = np.array([0, 0, 1])

            # add agents collisions
            self.registered_collisions = {agent.name: [] for agent in world.agents}

        # add landmarks
        world.landmarks = [
            TimerLandmark(self.np_random) for _ in range(self.num_landmarks)
        ]
        landmark_pos = {}
        for i, landmark in enumerate(world.landmarks):
            landmark.name = f"landmark_{i}"
            landmark.collide = False
            landmark.movable = False
            landmark.boundary = False
            pos = landmark.get_random_pos(world)
            landmark_pos[landmark.name] = pos

        self.landmarks = {landmark.name: landmark for landmark in world.landmarks}

        return world

    def reset_world(self, world, random):
        self.num_landmarks = len(self.landmarks)
        self.registered_collisions = {agent.name: [] for agent in world.agents}

        # set landmarks randomly in the world
        for land_id, landmark in self.landmarks.items():
            if landmark not in world.landmarks:
                world.landmarks.append(landmark)

            if self.landmark_curriculum["current"] == 0:
                landmark.reset(world, position=self.landmark_pos[land_id], size=1)
            elif self.landmark_curriculum["current"] == 1:
                landmark.reset(world, size=1)
            elif self.landmark_curriculum["current"] == 2:
                landmark.reset(world)
            else:
                raise ValueError(
                    f"Value '{self.landmark_curriculum['current']}' has not been implemented for landmark reset"
                )

        collide = True
        eta = 0.2
        # set random initial states
        for agent in world.agents:
            agent.color = np.array([0, 0, 1])
            while collide:
                agent.state.p_pos = self.np_random.uniform(
                    -world.max_size + eta,
                    world.max_size - eta,
                    world.dim_p)

                collide = any([is_collision(agent, land) for land in self.landmarks.values()])

            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

    # return all agents that are not adversaries
    @staticmethod
    def get_agents(world):
        return [agent for agent in world.agents]

    def reward_francesco(self, agent, world):
        """reward method.

        Reward function wich return a reward based on the current task.

        Parameters
        ----------
        agent : TYPE
            DESCRIPTION.
        world : TYPE
            DESCRIPTION.
        task : str, optional
            The default is "simple". Possible tasks:
                "simple": return 1 when the agent enters a landmark, 0
                    otherwise
                "time_penalty": If the agent does not enter a landmark, it
                    receives a negative reward that increases with each step.
                    When the agent enters a landmark it receives a positive
                    reward and resets the previously accumulated negative
                    reward. (Note: the agent receives a negative reward even
                    if it remains in a landmark. To receive a positive reward
                    it must exit and re-enter)
                "change_landmark": The agent receives a positive reward when
                    entering a new landmark. This means that the agent cannot
                    receive a positive reward by exiting and re-entering the
                    same landmark. (Note: at least two landmarks are required
                    in this mode). If self.landmark_penalty is not 0, the agent
                    receive a penalty at each time step (Note: penalty should
                    be much smaller than the reward)
                "change_landmark_avoid_borders": same as before but the agent
                    negative reward when hit the boarders of the map
                "all_landmarks": The agent receives a positive reward only when
                    it enters a new landmark. So to maximize the reward of the
                    episode it must reach all points of reference in the
                    environment

        Returns
        -------
        rew : float

        """
        rew = 0

        if self.task == "simple":
            # check for every landmark
            for landmark in world.landmarks:

                already_collided = landmark.name in self.registered_collisions[agent.name]
                had_collided = is_collision(agent, landmark)

                if not already_collided and had_collided:
                    # positive reward, and add collision
                    rew += self.landmark_reward
                    self.registered_collisions[agent.name] += [landmark.name]
                elif already_collided and not had_collided:
                    # if not on landmark and remove registered collision
                    self.registered_collisions[agent.name].remove(
                        landmark.name)
                    rew += 0

        elif self.task == "time_penalty":
            for landmark in world.landmarks:
                already_collided = landmark.name in self.registered_collisions[agent.name]
                had_collided = is_collision(agent, landmark)

                if not already_collided and had_collided:
                    # positive reward, and add collision
                    rew += self.landmark_reward
                    self.registered_collisions[agent.name] += [landmark.name]
                elif already_collided and not had_collided:
                    # if not on landmark and remove registered collision
                    self.registered_collisions[agent.name].remove(
                        landmark.name)

            if rew == 0:
                rew += self.landmark_penalty * min(
                    landmark.counter, self.max_landmark_counter
                )

        elif self.task == "change_landmark":
            assert self.num_landmarks > 1, "At least 2 landmarks are " \
                                           + f"needed for the task '{self.task}'"
            for landmark in world.landmarks:
                already_collided = landmark.name in self.registered_collisions[agent.name]
                had_collided = is_collision(agent, landmark)

                if not already_collided and had_collided:
                    # positive reward
                    rew += self.landmark_reward
                    # reset all the other landmark collisions
                    self.registered_collisions[agent.name] = [landmark.name]
            # if an agent doesn't collide with any landmark, it receive
            # negative a reward
            if rew == 0:
                rew += self.landmark_penalty
        elif self.task == "change_landmark_avoid_borders":
            for entity in world.entities:
                already_collided = entity.name in self.registered_collisions[agent.name]
                had_collided = is_collision(agent, entity)

                if not already_collided and had_collided and entity.name.split("_")[0] == "landmark":
                    # positive reward
                    rew += self.landmark_reward
                    # reset all the other landmark collisions
                    self.registered_collisions[agent.name] = [entity.name]
                elif had_collided and entity.name.split("_")[0] == "border":
                    rew += self.landmark_penalty

        # fix : we should add the possibility to stop the episode when the
        # agent reaches all the landmarks
        elif self.task == "all_landmarks":
            assert self.num_landmarks > 1, "At least 2 landmarks are " \
                                           + f"needed for the task '{self.task}'"
            for landmark in world.landmarks:
                already_collided = landmark.name in self.registered_collisions[agent.name]
                had_collided = is_collision(agent, landmark)

                if not already_collided and had_collided:
                    # positive reward
                    rew += self.landmark_reward
                    # reset all the other landmark collisions
                    self.registered_collisions[agent.name] = [landmark.name]

        return rew

    def reward(self, agent, world):
        def dist_reward():
            min_dist = 99999
            for landmark in world.landmarks:
                dist = get_distance(agent, landmark)
                min_dist = min(min_dist, dist)

            return world.max_size * 2 - min_dist

        lower_bound = 0
        upper_bound = self.landmark_reward

        if self.reward_curriculum["current"] == 0:
            rew = dist_reward()
        elif self.reward_curriculum["current"] == 2:
            rew = self.step_reward
            lower_bound = self.step_reward
        else:
            raise ValueError(
                f"Value '{self.reward_curriculum['current']}' has not been implemented for reward mode"
            )

        for landmark in world.landmarks:
            if is_collision(agent, landmark):
                # positive reward, and add collision
                rew = self.landmark_reward
                self.visited_landmarks.append(landmark.name)
                break

        if self.normalize_rewards:
            rew = min_max_norm(rew, lower_bound, upper_bound)

        return rew

    @staticmethod
    def observation(agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        # communication of all other agents
        comm = []
        other_agents_pos = []
        other_agents_vel = []
        for other in world.agents:
            comm.append(other.state.c)
            other_agents_pos.append(other.state.p_pos - agent.state.p_pos)
            other_agents_vel.append(other.state.p_vel)
        return np.concatenate(
            [agent.state.p_vel] +
            [agent.state.p_pos] +
            entity_pos +
            other_agents_pos +
            other_agents_vel
        )
