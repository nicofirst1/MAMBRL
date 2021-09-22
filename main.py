import time

from NavigationEnv import get_env

par_env = get_env()

episodes = 30
MAX_RESETS = 2

for n_resets in range(MAX_RESETS):
    obs = par_env.reset()
    par_env.render()

    done = {agent: False for agent in par_env.agents}
    live_agents = par_env.agents[:]
    has_finished = set()
    for i in range(episodes):
        actions = {
            agent: space.sample()
            for agent, space in par_env.action_spaces.items()
            if agent in done and not done[agent]
        }
        obs, rew, done, info = par_env.step(actions)
        par_env.render()
        time.sleep(0.1)

        for agent, d in done.items():
            if d:
                live_agents.remove(agent)
        has_finished |= {agent for agent, d in done.items() if d}
        if not par_env.agents:
            assert has_finished == set(
                par_env.possible_agents
            ), "not all agents finished, some were skipped over"
            break
