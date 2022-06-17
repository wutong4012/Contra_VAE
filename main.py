import os
import hydra
import torch
import pytorch_lightning as pl

from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from contra_vae_system import ContraVAESystem


torch.backends.cudnn.benchmark = True

@hydra.main(config_path='./', config_name='contra_vae_config')
def run(config):
    seed_everything(config.exp_params.seed, workers=True)
    
    ckpt_callback = ModelCheckpoint(
        save_top_k=3,
        monitor=config.optim_params.monitor,
        every_n_train_steps=config.exp_params.save_n_steps,
        filename=config.model_params.checkpoint_name +'-{step}' +'-{train_all_loss:.2f}'
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    trainer = pl.Trainer(
        default_root_dir=config.model_params.checkpoint_path,
        accelerator='gpu',
        # strategy='ddp',
        gpus=config.exp_params.num_gpus,
        callbacks=[ckpt_callback, lr_monitor],
        max_steps=int(config.exp_params.train_steps),
        min_steps=int(config.exp_params.train_steps),
        precision=16,
        log_every_n_steps=10,
        # auto_scale_batch_size='binsearch',
    )
    
    system = ContraVAESystem(config)
    # find_best_lr(trainer.tuner, system)

    trainer.fit(system)
    trainer.test(system)


def find_best_lr(tuner, system):
    lr_finder = tuner.lr_find(system)
    print(lr_finder.results)
    fig = lr_finder.plot(suggest=True)
    fig.savefig('lr.jpg')
    new_lr = lr_finder.suggestion()
    print('new_lr:', new_lr)


if __name__ == '__main__':
    run()
