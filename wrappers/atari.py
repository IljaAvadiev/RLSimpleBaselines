import gym
from gym import Wrapper, ObservationWrapper, RewardWrapper
from collections import deque
import numpy as np
import cv2


class RewardClipping(RewardWrapper):
    pass


class MaxFrame(ObservationWrapper):
    def __init__(self, env):
        super(MaxFrame, self).__init__(env)
        self.frames = deque(maxlen=2)

    def reset(self):
        observation = self.env.reset()
        if self.env.unwrapped.get_action_meanings()[1] == 'FIRE':
            obs, _, _, _ = self.env.step(1)
        for _ in range(2):
            self.frames.append(np.zeros(observation.shape))
        return observation

    def observation(self, observation):
        self.frames.append(observation)
        max_frames_values = np.maximum(self.frames[0], self.frames[1])
        return max_frames_values


# repeat action
# frame skipping
class RepeatAction(Wrapper):
    def __init__(self, env, repeat=4):
        super(RepeatAction, self).__init__(env)
        self.repeat = repeat

    def step(self, action):
        sum_reward = 0
        for _ in range(self.repeat):
            observation, reward, done, info = self.env.step(action)
            sum_reward += reward
            if done:
                break
        return observation, sum_reward, done, info


# remove y channel
# rescale to size between 0 and 1
# and rescale to 84x84
class PreprocessImage(ObservationWrapper):
    def __init__(self, env, shape):
        super(PreprocessImage, self).__init__(env)

        self.observation_space = gym.spaces.Box(
            0.0, 1.0, shape=shape, dtype=np.float32)

    def observation(self, observation):
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        observation = cv2.resize(
            observation, self.observation_space.shape[1:], interpolation=cv2.INTER_AREA)
        observation = observation.reshape(
            self.observation_space.shape).astype('float32')
        observation = observation / self.observation_space.high
        return observation


# stack n frames (4 was used)
class StackFrames(ObservationWrapper):
    def __init__(self, env, maxlen=4):
        super(StackFrames, self).__init__(env)
        self.maxlen = maxlen
        self.frames = deque(maxlen=maxlen)
        low = self.env.observation_space.low.repeat(maxlen, axis=0)
        high = self.env.observation_space.high.repeat(maxlen, axis=0)
        dtype = self.env.observation_space.dtype
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=dtype)

    def reset(self):
        observation = self.env.reset()
        for _ in range(self.maxlen):
            self.frames.append(observation)

        observation = np.vstack(self.frames)
        return observation

    def observation(self, observation):
        self.frames.append(observation)
        observation = np.vstack(self.frames)
        return observation


def apply_wrappers(env):
    env = MaxFrame(env)
    env = RepeatAction(env)
    env = PreprocessImage(env, shape=(1, 88, 88))
    env = StackFrames(env)
    return env
