# Session 14 — PPO for continuous control

**Status:** One PPO implementation is present in [`ppo_agent.py`](ppo_agent.py).
This directory has no TRPO implementation, evaluation script, comparison script,
captured training result, or benchmark. The [theory note](../../notes/md/note_14_ppo_trpo.md)
covers the algorithms and their assumptions.

## What the code contains

`ppo_agent.py` defines a `PPOConfig`, an actor-critic network, a rollout buffer,
and a PPO training loop. It uses Gymnasium's vector environment API, PyTorch,
NumPy, and tqdm. The file's `__main__` block trains on `BipedalWalker-v3`
with four environments and one million configured timesteps, then saves
`ppo_bipedalwalker.pt` in the current working directory.

## Running it

Install the dependencies used by this file, including Box2D support for
`BipedalWalker-v3`:

```bash
python -m pip install swig
python -m pip install "gymnasium[box2d]" numpy torch tqdm
python code/14_ppo_trpo/ppo_agent.py
```

Run the command from the repository root. It starts a potentially long training
run and writes a checkpoint; use an isolated environment and output directory.
The full run, checkpoint quality, runtime, and compatibility across platforms
have **not** been validated in this audit. There is no CLI for changing the
configuration: edit the `PPOConfig` construction in the file or import its
classes from your own script.

In a Python 3.13 macOS environment with PyTorch and Gymnasium already available,
installing `swig` before the Box2D extra succeeded, and a single eight-step
PPO update completed. This smoke check does not measure training quality or
validate the million-step default run.

## Reproducibility and limits

The source contains a seed setting and training configuration, but this
repository does not include a captured run, environment lock, evaluation
protocol, or comparison against another implementation. Do not treat the
default hyperparameters or an expected reward curve as measured results.
TRPO is discussed in the theory note; it is not implemented here.
