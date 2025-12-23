import hydra
from omegaconf import DictConfig, OmegaConf

from trainer import Trainer

OmegaConf.register_new_resolver("eval", eval)


@hydra.main(config_path="../config", config_name="trainer", version_base="1.1")
def main(cfg: DictConfig):
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
