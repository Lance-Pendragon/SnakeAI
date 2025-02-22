import gym
from env.SnakeEnv import SnakeEnv
from stable_baselines3 import PPO


def main():
    print("Hello, SnakeAI world!")
    env = SnakeEnv(50)
    timestamp = 1000
    model = PPO('MlpPolicy', env, verbose=1)
    for i in range (10000):
        print(f"on generation {i * timestamp:,}")
        model.learn(timestamp)


if __name__ == "__main__":
    main()