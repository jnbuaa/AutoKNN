# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 19:27:05 2019

@author: Nan Ji
"""
import os
import gym
import random
import numpy as np

from collections import deque

from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K


class DQN:
    def __init__(self):
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

        if os.path.exists('dqn.h5'):
            self.model.load_weights('dqn.h5')

        # 经验池
        self.memory_buffer = deque(maxlen=2000)
        # Q_value的discount rate，以便计算未来reward的折扣回报
        self.gamma = 0.95
        # 贪婪选择法的随机选择行为的程度
        self.epsilon = 1.0
        # 上述参数的衰减率
        self.epsilon_decay = 0.995
        # 最小随机探索的概率
        self.epsilon_min = 0.01

        self.env = gym.make('CartPole-v0')

    def build_model(self):
        """基本网络结构.
        """
        inputs = Input(shape=(4,))
        x = Dense(16, activation='relu')(inputs)
        x = Dense(16, activation='relu')(x)
        x = Dense(2, activation='linear')(x)

        model = Model(inputs=inputs, outputs=x)

        return model

    def update_target_model(self):
        """更新target_model
        """
        self.target_model.set_weights(self.model.get_weights())

    def egreedy_action(self, state):
        """ε-greedy选择action

        Arguments:
            state: 状态

        Returns:
            action: 动作
        """
        if np.random.rand() <= self.epsilon:
            #print('random action:_')
            return random.randint(0, 1)
        else:
            q_values = self.model.predict(state)[0]
            #print('q_values action:', np.argmax(q_values),'q_values:', q_values)
            return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done):
        """向经验池添加数据

        Arguments:
            state: 状态
            action: 动作
            reward: 回报
            next_state: 下一个状态
            done: 游戏结束标志
        """
        item = (state, action, reward, next_state, done)
        self.memory_buffer.append(item)

    def update_epsilon(self):
        """更新epsilon
        """
        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def process_batch(self, batch_size):
        """batch数据处理

        Arguments:
            batch_size: batch size

        Returns:
            X: states
            y: [Q_value1, Q_value2]
        """
         # 从经验池中随机采样一个batch
        data = random.sample(self.memory_buffer, batch_size)
        # 生成Q_target。
        states = np.array([d[0] for d in data])
        next_states = np.array([d[3] for d in data])

        y = self.model.predict(states)
        q = self.target_model.predict(next_states)

        for i, (_, action, reward, _, done) in enumerate(data):
            target = reward
            if not done:
                target += self.gamma * np.amax(q[i])
            y[i][action] = target

        return states, y


    def train(self, episode, batch_size):
        """训练
        Arguments:
            episode: 游戏次数
            batch_size： batch size

        Returns:
            history: 训练记录
        """
        self.model.compile(loss='mse', optimizer=Adam(1e-3))

        history = {'episode': [], 'episode_reward': [], 'loss': []}

        count = 0
        for i in range(episode):
            observation = self.env.reset() # The Obervation here is (State, reward, done, ).
            reward_sum = 0
            loss = np.infty
            done = False

            while not done:
                # 通过贪婪选择法ε-greedy选择action。
                x = observation.reshape(-1, 4)
                action = self.egreedy_action(x)
                observation, reward, done, _ = self.env.step(action) # The Obervation here is next State.
                # 将数据加入到经验池。
                reward_sum += reward
                self.remember(x[0], action, reward, observation, done)

                if len(self.memory_buffer) > batch_size:
                    # 训练
                    X, y = self.process_batch(batch_size)
                    loss = self.model.train_on_batch(X, y)

                    count += 1
                    # 减小egreedy的epsilon参数。
                    self.update_epsilon()

                    # 固定次数更新target_model
                    if count != 0 and count % 20 == 0:
                        self.update_target_model()

            if i % 5 == 0:
                history['episode'].append(i)
                history['episode_reward'].append(reward_sum)
                history['loss'].append(loss)
    
                print('Episode: {} | Episode reward: {} | loss: {:.3f} | epislon:{:.2f}'.format(i, reward_sum, loss, self.epsilon))

        self.model.save_weights('dqn.h5')

        return history

    def play(self):
        """使用训练好的模型测试游戏.
        """
        observation = self.env.reset()

        count = 0
        reward_sum = 0
        random_episodes = 0

        while random_episodes < 10:
            self.env.render()

            x = observation.reshape(-1, 4)
            q_values = self.model.predict(x)[0]
            action = np.argmax(q_values)
            observation, reward, done, _ = self.env.step(action)

            count += 1
            reward_sum += reward

            if done:
                print("Reward for this episode was: {}, turns was: {}".format(reward_sum, count))
                random_episodes += 1
                reward_sum = 0
                count = 0
                observation = self.env.reset()

        self.env.close()


if __name__ == '__main__':
    model = DQN()
    history = model.train(600, 32)
    model.play()    
    
    
    
    
    
