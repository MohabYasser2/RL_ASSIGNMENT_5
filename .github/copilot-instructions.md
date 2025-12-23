# Δ-IRIS: Efficient World Models with Context-Aware Tokenization

## Architecture Overview

Δ-IRIS implements a world model-based RL agent with three core components:
- **Tokenizer**: Compresses 64x64 RGB observations into discrete latent tokens using a VQ-VAE
- **WorldModel**: Transformer that predicts sequences of latents, rewards, and episode termination
- **ActorCritic**: Policy network for action selection trained in the world model's imagination

The agent alternates between collecting real environment data and training components in curriculum order: tokenizer → world model → actor-critic.

## Key Files & Structure

- `src/agent.py`: Main Agent class composing the three components
- `src/trainer.py`: Training orchestration with Hydra config management
- `src/main.py`: Entry point using Hydra's `@hydra.main`
- `config/trainer.yaml`: Main config with defaults for env and params
- `config/env/`: Environment-specific configs (atari.yaml, crafter.yaml)
- `params/`: Training hyperparameters (atari.yaml, crafter.yaml)
- `src/models/`: Core neural network implementations
- `src/envs/`: Environment wrappers and multiprocessing setup

## Configuration System

Uses Hydra with hierarchical configs:
```yaml
# Override pattern examples
python src/main.py env=atari params=atari env.train.id=BreakoutNoFrameskip-v4
```

- `env`: Selects environment config (atari/crafter)
- `params`: Selects training parameters
- `env.train.id`: Specific environment ID
- Configs compose via `_target_` for instantiation

## Training Workflow

1. **Data Collection**: Parallel environments collect episodes → stored in dataset
2. **Component Training**: Train tokenizer, world model, actor-critic in sequence
3. **Evaluation**: Test agent performance periodically
4. **Checkpointing**: Save model weights and dataset state

Run folders created in `outputs/YYYY-MM-DD/hh-mm-ss/` contain:
- `checkpoints/`: Model weights and optimizer state
- `media/episodes/`: Saved episodes for visualization
- `media/reconstructions/`: Autoencoder reconstruction samples

## Critical Commands

```bash
# Train on Crafter (default)
python src/main.py

# Train on Atari
python src/main.py env=atari params=atari env.train.id=BreakoutNoFrameskip-v4

# Resume training (from run folder)
./scripts/resume.sh

# Visualize agent
./scripts/play.sh              # Agent in real environment
./scripts/play.sh -w           # Play in world model
./scripts/play.sh -a           # Agent in world model
./scripts/play.sh -e           # Replay saved episodes
```

## Data Flow Patterns

- **Batch Processing**: Episodes → segments → collated batches with `collate_segments_to_batch`
- **Tokenization**: RGB frames → discrete latents via VQ-VAE
- **Sequence Modeling**: Transformer processes action + latent sequences
- **Imagination**: Actor-critic trained on world model rollouts

## Common Patterns

- **Hydra Instantiation**: Use `instantiate(cfg)` for config-driven object creation
- **Device Handling**: Components have `.device` property, prefer `component.to(device)`
- **Mixed Precision**: Training uses `torch.cuda.amp` for tokenizer/world model
- **W&B Logging**: Metrics logged per epoch with `wandb.log({'epoch': epoch, **metrics})`
- **Seed Management**: Set via `set_seed(cfg.params.common.seed)` in trainer init

## Environment Setup

- Atari: Uses `gymnasium[atari]` with custom wrappers for 64x64 resizing
- Crafter: Custom environment with procedurally generated worlds
- Multi-processing: `MultiProcessEnv` for parallel data collection
- World model env: `WorldModelEnv` for imagination-based training

## Debugging Tips

- Check `outputs/` folders for run-specific logs and media
- Use `./scripts/play.sh -e` to visualize collected episodes
- World model memory flushes after ~10 seconds in interactive mode
- Failed runs often due to missing Atari ROMs or CUDA memory issues