import gym
from env.SnakeEnv import SnakeEnv
from stable_baselines3 import PPO, DQN


def main():
    print("Hello, SnakeAI world!")
    env = SnakeEnv(50)
    timestamp = 300000
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(timestamp)
    # model.save('ppo_snake')

    env = SnakeEnv(50)
    timestamp = 1000000
    model = DQN('MlpPolicy', env, verbose=1)
    for i in range (10):
        print(f"on generation {i * timestamp:,}")
        model.learn(timestamp)
        model.save('dqn_generation_' + str(i * timestamp))

    

if __name__ == "__main__":
    main()