import gym
from env.SnakeEnv import SnakeEnv
from stable_baselines3 import PPO, DQN
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv
import time


numberOfGames = 6
timestamps = 1200000 * numberOfGames

def main():
    envs = [lambda: SnakeEnv(20) for _ in range(numberOfGames)]
    vec_env = SubprocVecEnv(envs)

    model = PPO('MultiInputPolicy', vec_env, verbose=1, device='cpu')
    

    start = time.time()
    model.learn(timestamps)
    end = time.time()
    print(f"Training time: {end - start:.2f} seconds")
    model.save('ppo_snake_30000000_timestamps')
    plot_rewards(vec_env)

    # Run on CPU
    # model = PPO.load('ppo_snake_10000000_timestamps', env).learn(10000)10

    # env = SnakeEnv(50)
    # timestamp = 1000000
    # model = DQN('MlpPolicy', env, verbose=1)
    # for i in range (10):
    #     print(f"on generation {i * timestamp:,}")
    #     model.learn(timestamp)
    #     model.save('dqn_generation_' + str(i * timestamp))


def plot_rewards(vec_env):
    rewardPerGenerations = vec_env.get_attr('rewardPerGeneration')
    for env in range(vec_env.num_envs):    
        generations = np.arange(len(rewardPerGenerations[env]))  # X-axis: generations
        plt.plot(generations, rewardPerGenerations[env], label='game #' + str(env))
        # plt.show()
    plt.xlabel('Generations')
    plt.ylabel('Episode Reward')
    plt.title('Reward per Generation - PPO Model #' + str(env))
    plt.legend(loc="upper left")
    plt.savefig('ppo_' + str(timestamps) + 'timestamps.png')

if __name__ == "__main__":
    main()