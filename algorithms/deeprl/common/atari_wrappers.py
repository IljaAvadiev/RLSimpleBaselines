import gym
from gym import Wrapper, ObservationWrapper, RewardWrapper
from collections import deque
import numpy as np
import cv2

# Most of the wrappers were taken from OpenAI Baselines

class RewardClipping(RewardWrapper):
    pass


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        print(env.unwrapped.get_action_meanings())
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs




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

        # # to trigger done when you lose a life
        # self.ale = env.unwrapped.ale
        # self.lives = 0

    def step(self, action):
        sum_reward = 0
        for _ in range(self.repeat):
            observation, reward, done, info = self.env.step(action)
            sum_reward += reward

            # # if you lose a life trigger done
            # new_lives = self.ale.lives()
            # done = done or new_lives < self.lives
            # self.lives = new_lives

            if done:
                break
        return observation, sum_reward, done, info

    # def reset(self):
    #     observation = self.env.reset()
    #     self.lives = self.ale.lives()
    #     return observation


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
    env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    env = MaxFrame(env)
    env = RepeatAction(env)
    env = PreprocessImage(env, shape=(1, 88, 88))
    env = StackFrames(env)
    return env
