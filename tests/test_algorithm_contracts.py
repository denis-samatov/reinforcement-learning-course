"""Tiny deterministic mathematical checks; no training campaign or model downloads."""
import importlib.util
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
import torch

CODE = Path(__file__).resolve().parents[1] / "code"


def load(folder, name):
    path = CODE / folder / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_dqn_seed_fixes_initial_weights_and_td_terminal_mask():
    dqn = load("10_deep_q_network", "dqn_algorithm")
    envs = [gym.make("CartPole-v1") for _ in range(2)]
    try:
        agents = [dqn.DQNAgent(env, dqn.DQNConfig(seed=17, device="cpu", hidden_dims=(4,)))
                  for env in envs]
        for a, b in zip(agents[0].policy_net.parameters(), agents[1].policy_net.parameters()):
            assert torch.equal(a, b)
        agent = agents[0]
        agent.config.gamma = .5
        for parameter in agent.target_net.parameters():
            parameter.data.zero_()
        agent.target_net.head.bias.data.copy_(torch.tensor([3., 4.]))
        batch = dqn.Transition(torch.zeros(2, 4), torch.zeros(2, dtype=torch.long),
                               torch.tensor([1., 2.]), torch.zeros(2, 4), torch.tensor([0., 1.]))
        assert agent.compute_td_target(batch).tolist() == [3., 2.]
        before = agent.policy_net.head.bias.detach().clone()
        assert np.isfinite(agent.update(batch))
        assert not torch.equal(agent.policy_net.head.bias, before)
    finally:
        for env in envs:
            env.close()


def test_a2c_seed_fixes_initial_weights():
    a2c = load("12_actor_critic", "pendulum_a2c")
    agents = [a2c.A2CAgent(a2c.A2CConfig(seed=17, hidden_sizes=(4, 4), render_mode=None))
              for _ in range(2)]
    try:
        for a, b in zip(agents[0].model.parameters(), agents[1].model.parameters()):
            assert torch.equal(a, b)
    finally:
        for agent in agents:
            agent.env.close()


def test_bellman_value_iteration_known_one_step_mdp():
    grid = load("13_dynamic_programming", "gridworld_env")
    dp = load("13_dynamic_programming", "dynamic_programming")
    env = grid.GridWorldEnv(height=1, width=2, start=(0, 0), goal=(0, 1))
    _, values, _ = dp.value_iteration(env, gamma=.9)
    np.testing.assert_allclose(values, [10., 0.], rtol=0, atol=1e-6)


def test_time_limit_truncation_keeps_dqn_bootstrap():
    dqn = load("10_deep_q_network", "dqn_algorithm")
    env = gym.wrappers.TimeLimit(gym.make("CartPole-v1").unwrapped, max_episode_steps=1)
    try:
        agent = dqn.DQNAgent(env, dqn.DQNConfig(seed=17, device="cpu", hidden_dims=(4,),
                                               total_steps=1, warmup_steps=100))
        agent.train()
        assert len(agent.buffer.buffer) == 1
        assert agent.buffer.buffer[0].terminated == 0.0
        assert agent.episode_lengths == [1]
    finally:
        env.close()


def test_toy_categorical_kl_matches_three_action_definition(monkeypatch):
    monkeypatch.syspath_prepend(str(CODE / "15_rlhf_basics"))
    toy = load("15_rlhf_basics", "ppo_rlhf")
    p = np.array([.2, .3, .5])
    q = np.array([.4, .4, .2])
    assert np.isclose(toy.kl_divergence(p, q), np.sum(p * np.log(p / q)))
    assert toy.kl_divergence(p, p) == 0.0
