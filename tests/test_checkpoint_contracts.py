"""Checkpoint tensors/configuration survive safe deserialization on CPU."""
import importlib.util
from pathlib import Path
import sys
import pickle
from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch

class UnexpectedPayload:
    pass


CODE = Path(__file__).resolve().parents[1] / "code"


def module(folder, name):
    spec = importlib.util.spec_from_file_location(name, CODE / folder / f"{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


@pytest.mark.parametrize("lesson", ["dqn", "a2c", "reinforce", "ppo"])
def test_save_load_uses_only_tensors_and_primitive_config(tmp_path, monkeypatch, lesson):
    if lesson == "dqn":
        source = module("10_deep_q_network", "dqn_algorithm")
        agent = source.DQNAgent(gym.make("CartPole-v1"), source.DQNConfig(device="cpu", hidden_dims=(4,)))
        network, environment = agent.policy_net, agent.env
    elif lesson == "a2c":
        source = module("12_actor_critic", "pendulum_a2c")
        agent = source.A2CAgent(source.A2CConfig(hidden_sizes=(4, 4), render_mode=None))
        network, environment = agent.model, agent.env
    elif lesson == "reinforce":
        source = module("11_policy_gradient", "lunarlander_reinforce")
        # Substitute the environment boundary, not the checkpoint implementation.
        environment = gym.make("CartPole-v1")
        monkeypatch.setattr(source, "gym", SimpleNamespace(make=lambda *a, **kw: environment))
        agent = source.REINFORCEAgent(source.REINFORCEConfig(hidden_sizes=(4, 4), render_mode=None))
        network = agent.policy
    else:
        source = module("14_ppo_trpo", "ppo_agent")
        agent = source.PPOAgent(source.PPOConfig(env_id="Pendulum-v1", n_envs=1, n_steps=2,
                                               hidden_dim=4, device="cpu"))
        network, environment = agent.network, agent.envs
    try:
        expected = {key: value.clone() for key, value in network.state_dict().items()}
        path = tmp_path / "model.pt"
        agent.save(str(path))
        checkpoint = torch.load(path, weights_only=True, map_location="cpu")
        assert isinstance(checkpoint["config"], dict)
        for parameter in network.parameters():
            parameter.data.zero_()
        agent.load(str(path))
        for key, value in network.state_dict().items():
            assert torch.equal(value, expected[key])
        torch.save({"config": UnexpectedPayload()}, path)
        with pytest.raises(pickle.UnpicklingError):
            agent.load(str(path))
    finally:
        environment.close()


@pytest.mark.parametrize("lesson", ["reinforce", "ppo"])
def test_initial_weights_follow_config_seed(lesson, monkeypatch):
    if lesson == "reinforce":
        source = module("11_policy_gradient", "lunarlander_reinforce")
        monkeypatch.setattr(source, "gym", SimpleNamespace(make=lambda *a, **kw: gym.make("CartPole-v1")))
        def factory():
            return source.REINFORCEAgent(source.REINFORCEConfig(seed=17, hidden_sizes=(4, 4), render_mode=None))
    else:
        source = module("14_ppo_trpo", "ppo_agent")
        def factory():
            return source.PPOAgent(source.PPOConfig(env_id="Pendulum-v1", seed=17, n_envs=1,
                                                   n_steps=2, hidden_dim=4, device="cpu"))
    agents = []
    try:
        agents.append(factory())
        torch.manual_seed(999)
        agents.append(factory())
        networks = [a.policy if lesson == "reinforce" else a.network for a in agents]
        assert all(torch.equal(a, b) for a, b in zip(networks[0].parameters(), networks[1].parameters()))
    finally:
        for agent in agents:
            (agent.env if lesson == "reinforce" else agent.envs).close()
