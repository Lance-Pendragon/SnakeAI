import gym
from env.SnakeEnv import SnakeEnv
from stable_baselines3 import PPO, DQN
import matplotlib.pyplot as plt
import numpy as np


def main():
    print("Hello, SnakeAI world!")
    env = SnakeEnv(50)
    timestamp = 3000000
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(timestamp)
    model.save('ppo_snake_3_million_timestamps')
    plot_rewards(env)

    # env = SnakeEnv(50)
    # timestamp = 1000000
    # model = DQN('MlpPolicy', env, verbose=1)
    # for i in range (10):
    #     print(f"on generation {i * timestamp:,}")
    #     model.learn(timestamp)
    #     model.save('dqn_generation_' + str(i * timestamp))

    
def plot_rewards(env):
    rewardPerGeneration = env.rewardPerGeneration
    generations = np.arange(len(rewardPerGeneration))  # X-axis: generations
    plt.plot(generations, rewardPerGeneration)
    plt.xlabel('Generations')
    plt.ylabel('Episode Reward')
    plt.title('Reward per Generation - PPO Model')
    plt.savefig('ppo_snake_10_million_generations.png')
    plt.show()


if __name__ == "__main__":
    main()