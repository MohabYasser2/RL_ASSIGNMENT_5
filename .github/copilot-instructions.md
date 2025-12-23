# Δ-IRIS World Model RL Agent - AI Coding Guidelines

## Architecture Overview

Δ-IRIS implements a world model-based RL agent with three sequential components:

- **Tokenizer**: Compresses 64x64 RGB observations into discrete tokens (VQ-VAE style)
- **World Model**: Transformer that predicts token sequences, rewards, and episode termination
- **Actor-Critic**: Policy network acting in world model's latent space for imagination-based training

## Key Design Patterns

### Model Components

All models (`src/models/`) inherit from `nn.Module` and implement `compute_loss(batch, **kwargs) -> LossWithIntermediateLosses, metrics_dict`:

```python
def compute_loss(self, batch, **kwargs):
    # Forward pass
    outputs = self.forward(batch)
    # Compute losses
    losses = LossWithIntermediateLosses(
        loss_total=total_loss,
        intermediate_losses={'reconstruction': rec_loss, 'commitment': comm_loss}
    )
    return losses, {'accuracy': acc, 'perplexity': perplexity}
```

### Data Flow

- Episodes stored as `Episode(observations, actions, rewards, ends)` dataclasses
- Training uses priority sampling based on component-specific episode counts
- Batches created via `collate_segments_to_batch()` with sequence packing
- **Reward logging**: Mean, std, min, max episode returns logged per collection phase

### Configuration Management

- Hydra-based configs in `config/` with hierarchical overrides
- Common pattern: `python src/main.py env=atari params=atari env.train.id=BreakoutNoFrameskip-v4`
- Parameters reference each other using `${params.other.value}` syntax

## Training Workflow

### Sequential Component Training

Components train in stages with `start_after_epochs` delays:

1. Tokenizer (epoch 1): Learns to compress observations
2. World Model (epoch 2): Predicts dynamics in token space
3. Actor-Critic (epoch 3): Learns policy via imagined rollouts

### Key Training Features

- Mixed precision (bfloat16) with `torch.autocast()`
- Gradient accumulation (`grad_acc_steps`) for larger effective batch sizes
- Priority sampling with `EpisodeCountManager` for balanced training
- Separate optimizers for each component with different weight decay

## Development Commands

### Training

```bash
# Crafter (default)
python src/main.py

# Atari games
python src/main.py env=atari params=atari env.train.id=BreakoutNoFrameskip-v4

# Resume training
./scripts/resume.sh
```

### Evaluation & Visualization

```bash
# Watch trained agent in environment
./scripts/play.sh

# Watch agent in world model (faster, limited memory)
./scripts/play.sh -a

# Play manually in world model
./scripts/play.sh -w

# Replay saved episodes
./scripts/play.sh -e
```

## Code Organization

### Directory Structure

- `src/models/`: Neural network components (world_model.py, tokenizer.py, actor_critic/)
- `src/data/`: Dataset management and batching logic
- `src/envs/`: Environment wrappers (SingleProcessEnv, MultiProcessEnv, WorldModelEnv)
- `config/`: Hydra configuration files
- `outputs/YYYY-MM-DD/HH-MM-SS/`: Run-specific outputs with checkpoints, media, scripts

## Kaggle Notes

- If running on Kaggle, the trainer prefers persistent storage under `/kaggle/working/<repo-name>/outputs/YYYY-MM-DD/HH-MM-SS/checkpoints/`.
- Use the `-s` (save-mode) flag for `./scripts/play.sh` to record gameplay; recordings and checkpoints are written under the run's `media/` and `checkpoints/` directories respectively.
- If files disappear in the Kaggle UI, refresh the `/kaggle/working/<repo-name>/outputs/...` path — the code creates a run-specific folder there to improve visibility and persistence.

### Key Classes

- `Trainer`: Orchestrates training loop and component scheduling
- `Agent`: Container for tokenizer + world_model + actor_critic
- `Collector`: Manages environment interaction and data collection
- `BatchSampler`: Handles priority-based episode sampling

## Common Patterns

### Tensor Operations

Extensive use of `einops` for dimension manipulation:

```python
# Rearrange batch dimensions
tokens = rearrange(batch.observations, 'b t h w c -> b t (h w c)')
# Add sequence dimension for transformer
tokens = repeat(tokens, 'b t d -> b t 1 d')
```

### Loss Computation

Structured losses with intermediate components:

```python
@dataclass
class LossWithIntermediateLosses:
    loss_total: torch.Tensor
    intermediate_losses: Dict[str, torch.Tensor]
```

### Environment Interaction

All environments implement gym-like interface with `num_actions` property and `step()`/`reset()` methods.

## Debugging Tips

### Checkpoint Loading

Checkpoints save component state dicts prefixed by component name. Load with:

```python
agent.load('checkpoints/last.pt', device, load_tokenizer=True, load_world_model=True, load_actor_critic=True)
```

Epoch-specific checkpoints are also saved as `epoch_{epoch}.pt` to prevent overwriting previous checkpoints.

### Memory Issues

- World model evaluation flushes memory after ~10 seconds for real-time interaction
- Use `pin_memory=True` in DataLoaders for GPU transfer efficiency
- Monitor gradient norms with `max_grad_norm` clipping

### Configuration Debugging

- Use `OmegaConf.resolve(cfg)` to expand all references
- Check resolved config with `print(OmegaConf.to_yaml(cfg))`
