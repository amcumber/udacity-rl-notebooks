# import argparse
import gym
import numpy as np

from agent import Agent, DoubleQAgent, QLearningAgent
from monitor import interact

# p = argparse.ArgumentParser(description='Run an agent in Taxi-v3')
# p.add_argument('--agent', )

# r_alpha = np.arange(0.01, 0.11, 0.01)
# Agent = DoubleQAgent
# best_alpha = -np.inf
# best_reward = -np.inf
# best_gamma = -np.inf
# best_agent = None

# for Agent in (DoubleQAgent, ):#QLearningAgent):
#     for a in r_alpha:
#         try:
#             env = gym.make('Taxi-v3')
#             agent = Agent(alpha=a, eps=1.0, eps_func=lambda x,y:x*0.999)
#             avg_rewards, best_avg_reward = interact(env, agent)
#             print(f'Agent: {agent}(alpha={agent.alpha}), gamma={agent.gamma}')
#             if best_reward < best_avg_reward:
#                 best_reward = best_avg_reward
#                 best_alpha = a
#                 best_agent = Agent
#                 print(f'Best Agent (alpha): {agent}')
#                 print(f'Best_avg_reward: {best_avg_reward}\n\n')
#         except KeyboardInterrupt:
#             break

#     r_gamma = r_alpha
#     for g in r_gamma:
#         try:
#             env = gym.make('Taxi-v3')
# #             agent = Agent(gamma=g, alpha=best_alpha)
#             agent = Agent(alpha=a,gamma=g, eps=1.0, eps_func=lambda x,y:x*0.999)
#             avg_rewards, best_avg_reward = interact(env, agent)
#             print(f'Agent: {agent}(alpha={agent.alpha}), gamma={agent.gamma}')
#             print(f'Best_avg_reward: {best_avg_reward}\n\n')
#             if best_reward < best_avg_reward:
#                 best_reward = best_avg_reward
#                 best_gamma = g
#                 best_agent = Agent
#                 print(f'Best Agent (alpha): {agent}')
#                 print(f'Best_avg_reward: {best_avg_reward}\n\n')
#         except KeyboardInterrupt:
#             break
            
# print(f'Best Agent (All): {best_agent}')
# print(f'Best_avg_reward (All): {best_avg_reward}\n\n')


env = gym.make('Taxi-v3')
agent = Agent()
avg_rewards, best_avg_reward = interact(env, agent)