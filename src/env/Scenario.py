from typing import Tuple, List

import numpy as np
from numpy.random import RandomState

from PettingZoo.pettingzoo.mpe._mpe_utils.core import Agent, Entity, World
from PettingZoo.pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from src.common import is_collision, get_distance
from .TimerLandmark import TimerLandmark
from ..common.utils import is_collision_border


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
            np_random: RandomState,
            landmark_reward: int = 1,
            max_landmark_counter: int = 4,
            landmark_penalty: int = -2,
            border_penalty: int = -3,
            agent_size: int = 0.1,
            landmark_size: int = 0.3,
            step_reward: int = 0,
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
        self.border_penalty = border_penalty
        self.num_agents = num_agents
        self.num_landmarks = num_landmarks
        self.max_size = max_size
        self.agent_size = agent_size
        self.landmark_size = landmark_size
        self.np_random = np_random
        self.step_reward = step_reward

        self.reward_step_strategy = "simple"
        self.reward_collision_strategy = "simple"
        self.landmark_reset_strategy = "simple"
        self.landmark_collision_strategy = "stay"

    def set_strategy(self,
                     reward_step_strategy: str = None,
                     reward_collision_strategy: str = None,
                     landmark_reset_strategy: str = None,
                     landmark_collision_strategy: str = None,
                     ):
        """

        Set internal strategy for given options.


        --landmark_reset_strategy : str,
            Dictates the strategy for initializing landmarks
            The default is "simple". Possible tasks:
              "simple": Landmark have static dimension and positions
              "random_pos": Landmark have static dimension and random positions
              "random_size": Landmark have random dimension and static positions
              "fully_random":Landmark have random dimension and position
        --landmark_collision_strategy : str,
            Dictates the strategy for when landmark experience collision with agent

            The default is "stay". Possible tasks:
              "stay": Does nothing
              "remove": Landmark is removed


        --reward_step_strategy/reward_collision_strategy : str, optional
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
                "positive_distance" : The agent receives a positive reward which
                    increases when getting closer to a landmark.
                 "negative_distance" : the agent receives a negative reward which
                    tends to zero when getting closer to a landmark

        """

        reward_step_keys, reward_collision_keys, landmark_reset_keys, landmark_collision_keys = self._get_valid_strategies()

        if reward_step_strategy is not None:
            assert (
                reward_step_strategy in reward_step_keys
            ), f"Reward step strategy '{reward_step_strategy}' is not valid." \
               f"\nValid options are {reward_step_keys}"
            self.reward_step_strategy = reward_step_strategy

        if reward_collision_strategy is not None:
            assert (
                reward_collision_strategy in reward_collision_keys
            ), f"Reward collision strategy '{reward_collision_strategy}' is not valid." \
               f"\nValid options are {reward_collision_keys}"
            self.reward_collision_strategy = reward_collision_strategy

        if landmark_reset_strategy is not None:
            assert (
                landmark_reset_strategy in landmark_reset_keys
            ), f"Landmark reset strategy '{landmark_reset_strategy}' is not valid." \
               f"\nValid options are {landmark_reset_keys}"
            self.landmark_reset_strategy = landmark_reset_strategy

        if landmark_collision_strategy is not None:
            assert (
                landmark_collision_strategy in landmark_collision_keys
            ), f"Landmark collision strategy '{landmark_collision_strategy}' is not valid." \
               f"\nValid options are {landmark_collision_keys}"
            self.landmark_collision_strategy = landmark_collision_strategy

    def _get_valid_strategies(self):

        strat_docs = self.get_descriptive_strategy()

        landmark_reset_keys = list(
            strat_docs["landmark_reset_strategy"].keys())
        landmark_collision_keys = list(
            strat_docs["landmark_collision_strategy"].keys())
        reward_step_keys = list(strat_docs["reward_step_strategy"].keys())
        reward_collision_keys = list(
            strat_docs["reward_collision_strategy"].keys())

        return reward_step_keys, reward_collision_keys, landmark_reset_keys, landmark_collision_keys

    def get_current_strategy(self) -> Tuple[str, str, str, str]:
        """get_current_strategy method.

        returns a list of strings describing the strategies adopted for
        defining certain events in the environment, i.e. reward_step_strategy
        (which reward is given during a normal step in the environment),
        reward_collision_strategy (which reward is given during a collision),
        landmark_reset_strategy (how landmarks are handled),
        landmark_collision_strategy (what happens when there is a
        collision with a landmark)
        Returns
        -------
        Tuple[str, str, str, str]

        """
        return self.reward_step_strategy, self.reward_collision_strategy, self.landmark_reset_strategy, self.landmark_collision_strategy

    def get_descriptive_strategy(self):
        """get_descriptive_strategy method.

        returns a dictionary containing customizable elements within the
        environment. Each of the elements is represented with a dictionary
        having as keys the possible selectable options and as values ​​some
        descriptions on their behavior
        Returns
        -------
        doc : dict


        """
        doc = dict(
            landmark_reset_strategy=dict(
                simple=" Landmark have static dimension and positions",
                random_pos=" Landmark have static dimension and random positions",
                random_size="Landmark have random dimension and static positions",
                fully_random=" Landmark have random dimension and positions",

            ),
            landmark_collision_strategy=dict(
                stay="Does nothing",
                remove="Landmark is removed on collision"
            ),
            reward_step_strategy=dict(
                simple="Reward at each timestep is zero",
                time_penalty=""" If the agent does not enter a landmark, it receives a negative reward that increases with each step.
                    When the agent enters a landmark it receives a positive reward and resets the previously accumulated negative reward. 
                    (Note: the agent receives a negative reward even if it remains in a landmark. To receive a positive reward it must exit and re-enter)""",
                change_landmark=f"The step reward is equal to landmark_penalty {self.landmark_penalty}",
                positive_distance="""The agent receives a positive reward which increases when getting closer to a landmark. """,
                negative_distance="""The agent receives a negative reward which increases when getting closer to a landmark. """

            ),
            reward_collision_strategy=dict(
                simple=f"The agent get a reward equal to landmark_reward ({self.landmark_reward}) when colliding with a landmark",
                time_penalty=f"""The agent get a reward equal to landmark_reward ({self.landmark_reward}) when colliding with a landmark.
                The collision also resets the previously accumulated negative reward in the landmarks.
                (Note: the agent receives a negative reward even if it remains in a landmark. To receive a positive reward it must exit and re-enter)""",
                change_landmark=f"""The agent receives a positive reward when entering a new landmark ({self.landmark_reward}). 
                This means that the agent cannot receive a positive reward by exiting and re-entering the same landmark. 
                (Note: at least two landmarks are required in this mode). 
                If self.landmark_penalty is not 0, the agent receive a penalty at each time step (Note: penalty should be much smaller than the reward)""",
                change_landmark_avoid_borders=f"Same as change landmark but the agent also experience a penalty of border_penalty ({self.border_penalty}) when hitting a border.",
                all_landmarks="""The agent receives a positive reward only when it enters a new landmark. 
                So to maximize the reward of the episode it must reach all points of reference in the environment"""
            )

        )

        return doc

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
            self.registered_collisions = {agent.name: []
                                          for agent in world.agents}

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

        self.landmarks = {
            landmark.name: landmark for landmark in world.landmarks}
        self.landmark_pos = landmark_pos

        return world

    def remove_collided_landmarks(self, world) -> bool:
        """
        If the strategy is correct check for collision with landmarks and remove them from the render geoms
        """
        if self.landmark_collision_strategy == "remove":
            # get visited landmarks ids
            visited_landmarks = list(self.registered_collisions.values())
            visited_landmarks = [
                item for sublist in visited_landmarks for item in sublist]

            # remove duplicates
            visited_landmarks = list(set(visited_landmarks))
            self.num_landmarks -= len(visited_landmarks)
            for lndmrk_id in visited_landmarks:
                landmark = self.landmarks[lndmrk_id]
                world.entities.remove(landmark)
                world.landmarks.remove(landmark)
            if len(visited_landmarks) > 0:
                return True

        return False

    def reset_world(self, world, random, landmarks_positions=None, agents_positions=None):
        """reset_world method.

        reset agents and landmarks. landmarks positions are reset based on the
        landmark_reset_strategy, while agents are reset randomly. 
        If landmarks_positions or agents_position are not None, landmarks and
        agents are positioned accordingly. 
        Parameters
        ----------
        world :
        random :
        landmarks_positions : list, optional
            positions of landmarks. The default is None.
        agents_positions : list, optional
            positions of agents. The default is None.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.num_landmarks = len(self.landmarks)
        self.registered_collisions = {agent.name: [] for agent in world.agents}

        if landmarks_positions is not None:
            assert len(landmarks_positions) == len(world.landmarks),\
                f"{len(landmarks_positions)} positions have been identified but there are {len(world.landmarks)} landmarks"
            for landmark, landmark_position in zip(world.landmarks, landmarks_positions):
                landmark.reset(world, position=landmark_position,
                               size=self.landmark_size)
        else:
            # set landmarks randomly in the world
            for land_id, landmark in self.landmarks.items():
                if landmark not in world.landmarks:
                    world.landmarks.append(landmark)

                if self.landmark_reset_strategy == "simple":
                    landmark.reset(
                        world, position=self.landmark_pos[land_id], size=self.landmark_size)
                elif self.landmark_reset_strategy == "random_pos":
                    landmark.reset(world, size=self.landmark_size)
                elif self.landmark_reset_strategy == "random_size":
                    landmark.reset(world, position=self.landmark_pos[land_id])
                elif self.landmark_reset_strategy == "fully_random":
                    landmark.reset(world)
                else:
                    raise ValueError(
                        f"Value '{self.landmark_reset_strategy}' has not been implemented for landmark reset"
                    )

        collide = True
        eta = 0.2
        # set random initial states
        if agents_positions is not None:
            assert len(agents_positions) == len(world.agents),\
                f"{len(agents_positions)} positions have been identified but there are {len(world.agents)} agents"
            for agent, agent_position in zip(world.agents, agents_positions):
                agent.color = np.array([0, 0, 1])
                agent.state.p_pos = agent_position.copy()
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)
        else:
            for agent in world.agents:
                agent.color = np.array([0, 0, 1])
                while collide:
                    # agent.state.p_pos = self.np_random.uniform(
                    #     -world.max_size + eta,
                    #     world.max_size - eta,
                    #     world.dim_p)
                    length = np.sqrt(np.random.uniform(0, 1))
                    angle = np.pi * np.random.uniform(0, 2)
                    x = length * np.cos(angle)
                    y = length * np.sin(angle)
                    agent.state.p_pos = np.array([x, y])

                    collide = any([is_collision(agent, land)
                                  for land in self.landmarks.values()])

                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)

    # return all agents that are not adversaries
    @staticmethod
    def get_agents(world):
        return [agent for agent in world.agents]

    def set_landmarks_pos(self, world, landmarks_positions: List):
        assert len(landmarks_positions) == len(world.landmarks),\
            f"{len(landmarks_positions)} positions have been identified but there are {len(world.landmarks)} landmarks"
        for landmark, landmark_position in zip(world.landmarks, landmarks_positions):
            landmark.set_pos(world, landmark_position)

    def set_agents_pos(self, world, agents_positions: List):
        assert len(agents_positions) == len(world.agents),\
            f"{len(agents_positions)} positions have been identified but there are {len(world.agents)} agents"
        for agent, agent_position in zip(world.agents, agents_positions):
            agent.color = np.array([0, 0, 1])
            agent.state.p_pos = agent_position
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

    def reward(self, agent, world):
        """reward method.

        Reward function wich return a reward based on the current task.

        Parameters
        ----------
        agent : TYPE
            DESCRIPTION.
        world : TYPE
            DESCRIPTION.

        -------
        rew : float

        """

        rew = 0

        if self.reward_collision_strategy in ["change_landmark", "all_landmarks"]:
            assert self.num_landmarks > 1, "At least 2 landmarks are " \
                                           + f"needed for the task '{self.reward_collision_strategy}'"

        # check for every landmark
        for landmark in world.landmarks:
            already_collided = landmark.name in self.registered_collisions[agent.name]
            had_collided = is_collision(agent, landmark)

            ##############################################
            # LANDMARK COLLISION REWARD
            ##############################################
            # decide the reward strategy when agent collides with a landmark

            # first registered collision between agent and landmark
            if not already_collided and had_collided:
                # positive reward
                rew += self.landmark_reward

                if self.reward_collision_strategy in ["simple", "time_penalty"]:
                    # if not on landmark and remove registered collision
                    self.registered_collisions[agent.name] += [landmark.name]
                elif self.reward_collision_strategy in ["change_landmark", "change_landmark_avoid_borders",
                                                        "all_landmarks"]:
                    # reset all the other landmark collisions
                    self.registered_collisions[agent.name] = [landmark.name]

            if self.reward_collision_strategy in ["simple", "time_penalty"]:

                # if not on landmark and remove registered collision
                if already_collided and not had_collided:
                    self.registered_collisions[agent.name].remove(
                        landmark.name)

        if self.reward_collision_strategy == "change_landmark_avoid_borders":
            for border in world.borders:
                had_collided = is_collision_border(border, agent)

                if had_collided:
                    rew += self.border_penalty

        ##############################################
        # STEP REWARD
        ##############################################
        # decide the reward strategy for every step

        def dist_reward(is_positive=True):
            """
            estiamate the min distance between current agent and closest landmark
            """
            min_dist = 99999
            for landmark in world.landmarks:
                dist = get_distance(agent, landmark)
                min_dist = min(min_dist, dist)

            if is_positive:
                rew = world.max_size * 2 - min_dist
            else:
                rew = - min_dist

            return rew

        if rew >= 0:

            if self.reward_step_strategy == "time_penalty":

                counters = [land.counter for land in world.landmarks]
                counters = max(counters)

                rew += self.landmark_penalty * min(
                    counters, self.max_landmark_counter
                )

            elif self.reward_step_strategy == "change_landmark":

                # if an agent doesn't collide with any landmark, it receive
                # negative a reward
                rew += self.landmark_penalty
            elif self.reward_step_strategy == "positive_distance":
                rew += dist_reward(is_positive=True)
            elif self.reward_step_strategy == "negative_distance":
                rew += dist_reward(is_positive=False)

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
