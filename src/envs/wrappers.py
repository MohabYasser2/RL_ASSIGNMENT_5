"""
Credits to https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
"""

from typing import Tuple
import gym
import numpy as np
from PIL import Image


def _unwrap_obs(obs):
    return obs[0] if isinstance(obs, tuple) else obs


def _unwrap_step(out):
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = terminated or truncated
    else:
        obs, reward, done, info = out
    return obs, reward, done, info


def make_atari(id, size=64, max_episode_steps=None, noop_max=30, frame_skip=4,
               done_on_life_loss=False, clip_reward=False):
    env = gym.make(id)
    assert 'NoFrameskip' in env.spec.id or 'Frameskip' not in env.spec
    env = ResizeObsWrapper(env, (size, size))
    if clip_reward:
        env = RewardClippingWrapper(env)
    if max_episode_steps is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    if noop_max is not None:
        env = NoopResetEnv(env, noop_max=noop_max)
    env = MaxAndSkipEnv(env, skip=frame_skip)
    if done_on_life_loss:
        env = EpisodicLifeEnv(env)
    return env


class ResizeObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, size: Tuple[int, int]) -> None:
        super().__init__(env)
        self.size = tuple(size)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(size[0], size[1], 3), dtype=np.uint8
        )
        self.unwrapped.original_obs = None

    def resize(self, obs: np.ndarray):
        img = Image.fromarray(obs)
        img = img.resize(self.size, Image.BILINEAR)
        return np.array(img)

    def observation(self, observation):
        observation = _unwrap_obs(observation)
        self.unwrapped.original_obs = observation
        return self.resize(observation)


class RewardClippingWrapper(gym.RewardWrapper):
    def reward(self, reward):
        return np.sign(reward)


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        obs = _unwrap_obs(out)

        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)

        for _ in range(noops):
            out = self.env.step(self.noop_action)
            obs, _, done, _ = _unwrap_step(out)
            if done:
                obs = _unwrap_obs(self.env.reset(**kwargs))

        return obs

    def step(self, action):
        return _unwrap_step(self.env.step(action))


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = _unwrap_step(self.env.step(action))
        self.was_real_done = done

        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            done = True

        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs = _unwrap_obs(self.env.reset(**kwargs))
        else:
            obs, _, _, _ = _unwrap_step(self.env.step(0))
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)

    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}

        for i in range(self._skip):
            obs, reward, done, info = _unwrap_step(self.env.step(action))
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break

        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return _unwrap_obs(self.env.reset(**kwargs))
