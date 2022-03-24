import keyboard
from tqdm import trange

from src.common import Params
from src.env import get_env
from src.env.EnvWrapper import EnvWrapper, get_env_wrapper
from src.model.EnvModel import NextFramePredictor
from src.trainer.EnvModelTrainer import EnvModelTrainer
from src.trainer.Policies import OptimalAction

params = Params()

if not params.visible:
    import pyglet

    pyglet.options['shadow_window'] = False


def user_game(env, config):
    wrapper_configs = config.get_env_wrapper_configs()
    real_env = env(
        env=get_env(config.get_env_configs()),
        **wrapper_configs,
    )
    moves = {"w": 4, "a": 1, "s": 3, "d": 2}

    game_reward = 0
    finish_game = False

    real_env.reset()
    while finish_game is False:
        while True:
            real_env.cur_env.render()
            if keyboard.is_pressed("w"):
                user_move = 4
                break
            elif keyboard.is_pressed("a"):
                user_move = 1
                break
            elif keyboard.is_pressed("s"):
                user_move = 3
                break
            elif keyboard.is_pressed("d"):
                user_move = 2
                break

        _, reward, done, _ = real_env.step({"agent_0": user_move})
        game_reward += reward["agent_0"]

        if done["__all__"]:
            while True:
                print("Finee! Total reward: ", game_reward)
                exit_input = input(
                    "Gioco terminato! Iniziare un'altra partita? (y/n)"
                )
                if exit_input == "n":
                    finish_game = True
                    real_env.cur_env.close()
                    break
                elif exit_input == "y":
                    game_reward = 0
                    real_env.reset()
                    break


if __name__ == "__main__":
    params = Params()
    # uncomment the following 2 lines to train the model free
    env = get_env_wrapper(params)
    # trainer = ModelFreeTrainer(ModelFree, PPO_Agent, env, params)
    # trainer.train()
    trainer = EnvModelTrainer(NextFramePredictor, env, params)
    epochs = 3000
    for step in trange(epochs, desc="Training env model"):
        trainer.collect_trajectories(OptimalAction)
        trainer.train(step, trainer.cur_env)

    trainer.train()
