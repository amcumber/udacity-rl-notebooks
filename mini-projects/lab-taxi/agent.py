from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from functools import reduce
import numpy as np
import random
from typing import Union


@dataclass
class DoubleQAgent:
    """ Agent using Double Q-learning and greedy-epsilon function"""
    nA: int = 6
    alpha: 'float (0,1]' = 0.2
    gamma: 'float [0,1]' = 1
    eps: 'float (0,np.inf)' = 1.0
    eps_func: 'callable f(x)' = None
    actions: np.array = None
    
    def __post_init__(self):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.Q1 = defaultdict(lambda: np.zeros(self.nA))
        self.Q2 = defaultdict(lambda: np.zeros(self.nA))
        
        if self.eps_func is None:
            self.eps_func = self._default_eps_update
        self._episodes = 1
        if self.actions is None:
            self.actions = np.arange(self.nA)
        
    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        action = self.eps_greedy_double_q(
            (self.Q1, self.Q2),
            state,
            self.eps,
            self.actions,
        )
        return action
        
    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.Q1, self.Q2 = self.update_Q_double_Q_max(
            (self.Q1, self.Q2),
            self.alpha,
            self.gamma,
            state,
            action,
            reward,
            next_state,
        )
        if done:
            self.update_epsilon()
        return None
        
    def update_epsilon(self):
        """ Update epsilon with number of learning episodes"""
        self._episodes += 1
        self.eps = self.eps_func(self.eps, self._episodes)
        return None
    
    @staticmethod
    def _default_eps_update(eps, episodes):
        return 1 / episodes

    @staticmethod
    def eps_greedy_double_q(
        Qs: List[defaultdict[int: np.array]], 
        state: int,
        eps: float,
        actions: np.array = None
    ) -> float:
        """ Select the $\epsilon$-greedy policy, `$\pi$`

        Parameters
        ----------
        Qs : List[defaultdict[int: np.array]]
            The list of Q-tables
        eps : float [0, 1]
            `$\epsilon$`, percent that a random choice is selected over the greedy 
            policy
        """
        if actions is None:
            actions = np.arange(len(Qs[0][state]))
        if random.random() > eps:
            # select Q
            Qs_max = [np.max(Q[state]) for Q in Qs]
            i_q = np.argmax(Qs_max)
            Q_state = Qs[i_q][state]
            return np.argmax(Q_state)
        else:
            return random.choice(actions)

    @staticmethod
    def update_Q_double_Q_max(
        Qs: List[defaultdict[int: np.array]], 
        alpha: float,
        gamma: float,
        state: int,
        action: int,
        reward : int,
        next_state: int = None,
    ):
        """ Update Q-table for Q learning

        Parameters
        ----------
        Q1 : defaultdict[int: np.array]
            List of Q-tables
        alpha : float 
            scaling parameter to weight previous policy values
        gamma : float
            past reward scaling
        state : int
            current state
        action : int
            planned action from state to next_state
        reward : float
            reward
        next_state : int
            next state
        """
        if np.random.random() >= 0.5:
            i = 0
        else:
            i = 1
        j = (i+1)%2  # flip: iff i=1 -> j=0

        Qsa = Qs[i][state][action]
        # use other table e.g. if i = 0, use table for i = 1
        if next_state is not None:
            Qsa_next = np.max(Qs[j][next_state])
        else:
            Qsa_next = 0
        target = gamma * Qsa_next
        Qs[i][state][action] = ((1 - alpha) * Qsa) + (alpha * (reward + target))
        return Qs

    
@dataclass
class QLearningAgent:
    """ Agent using Q-learning and greedy-epsilon function"""
    nA: int = 6
    alpha: 'float (0,1]' = 0.2
    gamma: 'float [0,1]' = 1
    eps : 'float (0,np.inf)' = 1.0
    eps_func: 'callable f(x)' = None
    actions: np.array = None
    
    def __post_init__(self):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self._episodes = 1
        
        if self.eps_func is None:
            self.eps_func = self._default_eps_update
        if self.actions is None:
            self.actions = np.arange(self.nA)
        
    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        action = self.eps_greedy(
            self.Q[state],
            self.eps,
            self.actions,
        )
        return action
        
    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.Q = self.update_Q_sarsamax(
            self.Q,
            self.alpha,
            self.gamma,
            state,
            action,
            reward,
            next_state,
        )
        if done:
            self.update_epsilon()
        return None
    
    def update_epsilon(self):
        """ Update epsilon with number of learning episodes"""
        self._episodes += 1
        self.eps = self.eps_func(self.eps, self._episodes)
        return None
    
    @staticmethod
    def _default_eps_update(eps, episodes):
        return 1 / episodes

    @staticmethod
    def eps_greedy(
        Q_state: np.array,                
        eps: float,
        actions: np.array,
    ) -> float:
        """ Select the $\epsilon$-greedy policy, `$\pi$`

        Parameters
        ----------
        Q_state : np.array
            The Q-table of policies, `$\pi$` for a specific state (Q[state])
        eps : float [0, 1]
            `$\epsilon$`, percent that a random choice is selected over the greedy 
            policy
        actions : np.array
            indexable collection of actions
        """
        if random.random() > eps:
            return np.argmax(Q_state)
        else:
            return random.choice(actions)

    @staticmethod
    def update_Q_sarsamax(
        Q: defaultdict[int: np.array], 
        alpha: float,
        gamma: float,
        state: int,
        action: int,
        reward : int,
        next_state: int = None,
    ) -> defaultdict[int: np.array]:
        """ Update Q-table for Q learning

        Parameters
        ----------
        Q : defaultdict[int: np.array]
            Q-table with Q[state][action] organization
        alpha : float 
            scaling parameter to weight previous policy values
        gamma : float
            past reward scaling
        state : int
            current state
        action : int
            planned action from state to next_state
        reward : float
            reward
        next_state : int
            next state
        """
        Qsa = Q[state][action]
        Qsa_next = np.max(Q[next_state]) if (next_state is not None) else 0
        target = reward + (gamma * Qsa_next)
        Qsa = Qsa + alpha * (target - Qsa)
        Q[state][action] = Qsa
        return Q
    
@dataclass
class ExpectedSarsaAgent:
    """ Agent using Expected-Sarsa learning and greedy-epsilon function"""
    nA: int = 6
    alpha: 'float (0,1]' = 0.2
    gamma: 'float [0,1]' = 1
    eps : 'float (0,np.inf)' = 1.0
    eps_func: 'callable f(x)' = None
    actions: np.array = None
    
    def __post_init__(self):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self._episodes = 1
        if self.eps_func is None:
            self.eps_func = self._default_eps_update
        if self.actions is None:
            self.actions = np.arange(self.nA)
        
    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        action = self.eps_greedy(
            self.Q[state],
            self.eps,
            self.actions,
        )
        return action
        
    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.Q = self.update_Q_sarsaexpected(
            self.Q,
            self.alpha,
            self.gamma,
            state,
            action,
            reward,
            next_state,
        )
        if done:
            self.update_epsilon()
        return None
    
    def update_epsilon(self):
        """ Update epsilon with number of learning episodes"""
        self._episodes += 1
        self.eps = self.eps_func(self.eps, self._episodes)
        return None
    
    @staticmethod
    def _default_eps_update(eps, episodes):
        return 1 / episodes


    @staticmethod
    def eps_greedy(
        Q_state: np.array,                
        eps: float,
        actions: np.array,
    ) -> float:
        """ Select the $\epsilon$-greedy policy, `$\pi$`

        Parameters
        ----------
        Q_state : np.array
            The Q-table of policies, `$\pi$` for a specific state (Q[state])
        eps : float [0, 1]
            `$\epsilon$`, percent that a random choice is selected over the greedy 
            policy
        actions : np.array
            indexable collection of actions
        """
        if random.random() > eps:
            return np.argmax(Q_state)
        else:
            return random.choice(actions)

    @staticmethod
    def update_Q_sarsaexpected(
        Q: defaultdict[int: np.array],
        alpha: float,
        gamma: float,
        eps: float,
        state: int,
        action: int,
        reward: float,
        next_state: int = None,
    ) -> defaultdict[int: np.array]:
        """ Update Q-table using expected sarsa

        Parameters
        ----------
        Q : defaultdict[int: np.array]
            Q-table with Q[state][action] organization
        alpha : float 
            scaling parameter to weight previous policy values
        gamma : float
            past reward scaling
        eps : float
            epsilon - used for evaluating expected value of the next state using
            the epsilon greedy policy ($<|\pi(a|S_{s+1})|>$, where 
            $\pi = \epsilon-greedy(Q, S, A)$)
        state : int
            current state
        action : int
            planned action from state to next_state
        reward : 
        next_state : int
            next state
        """
        Qsa = Q[state][action]
        nA = len(Q[state])

        policy_s = np.ones(nA) * eps / nA
        action_greed = np.argmax(Q[next_state])
        policy_s[action_greed] += (1 - eps)
        Qsa_next = np.dot(Q[next_state], policy_s)
        target = reward + (gamma * Qsa_next)
        Qsa = Qsa + alpha * (target - Qsa)
        Q[state][action] = Qsa
        return Q

class Agent(DoubleQAgent):
    def __init__(self):
        super().__init__(alpha=0.1, gamma=1.0,
                         eps=5e-2, eps_func=lambda x,y: x*0.999)
# class Agent(QLearningAgent):
#     def __init__(self):
#         super().__init__(alpha=0.25, gamma=0.77,
#                          eps=0.01, eps_func=lambda x,y: x)
# Agent = DoubleQAgent
Agent = QLearningAgent
# Agent = ExpectedSarsaAgent
