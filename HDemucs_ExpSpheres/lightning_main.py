import torch
import lightning as pl
import yaml
import pathlib
from MyDataset import AutoSourceDataset
from LightningModel import LightningModel
from lightning.pytorch.loggers import TensorBoardLogger
import bitsandbytes as bnb

from demucs.hdemucs import HDemucs


def main():
    # Set matmul precision for Tensor Core optimization
    torch.set_float32_matmul_precision('medium')
    
    # read the configuration file
    with open('./conf.yaml','r') as f:
        conf = yaml.safe_load(f)
    
    # initialize the train dataset 
    train_dataset = AutoSourceDataset(
        root_dir = pathlib.Path(conf['dataset']['root_dir']),
        chunk_duration = conf['dataset']['chunk_duration'],
        stride_duration = conf['dataset']['stride_duration']
    )
    print(f'\nTrain set length: {len(train_dataset)}')
    
    # initialize the test dataset(s) 
    val_datasets = [
        AutoSourceDataset(
            root_dir = pathlib.Path(conf['dataset']['val_root_dir']),
            chunk_duration = conf['dataset']['chunk_duration'],
            stride_duration = conf['dataset']['stride_duration'],
            subset = 'validation'  
        )
    ]
    print(f'Number of validation sets: {len(val_datasets)}')
    for val_dataset_idx in range(len(val_datasets)):
        print(f' - Set {val_dataset_idx} length: {len(val_datasets[val_dataset_idx])}')

    # Define dataloaders
    train_loader = torch.utils.data.DataLoader(
                            train_dataset,
                            batch_size = conf['dataset']['batch_size'],
                            shuffle = True,
                            pin_memory = True,
                            num_workers= conf['dataset']['num_workers'],
                        )

    val_loaders = [ torch.utils.data.DataLoader(
                            val_dataset,
                            batch_size = conf['dataset']['batch_size'],
                            shuffle = False,
                            pin_memory = True,
                            num_workers = conf['dataset']['num_workers'],
                    )
                    for val_dataset in val_datasets]

    # number of sources
    mix, sources, instrument_names = next(iter(val_loaders[0]))

    print("\n--- Ejemplo de batch del val_loader[0] ---")
    print(f"Mezcla shape: {mix.shape}")        # [B, C, T]
    print(f"Fuentes shape: {sources.shape}")   # [B, C, T]
    print(f"Instrumentos: {instrument_names}") # Lista de strings por muestra
    print("------------------------------------------\n")
    
    S = mix.shape[1]
    print(f'The model is being trained for {S} sources')

    confdemucs = {
        'sources': [str(s) for s in range(S)],
        'audio_channels': S,
        'cac': conf['hdemucs_conf']['cac'],
        'samplerate': conf['hdemucs_conf']['samplerate'],
        'channels': conf['hdemucs_conf']['channels'],
        'segment': conf['dataset']['chunk_duration']
    }
    
    model = HDemucs(**confdemucs)

    # experiment name
    conf_loss = conf['train']['loss_function']
    conf_chan = confdemucs['channels']
    conf_perm = conf['train']['permute_sources']
    conf_cac  = conf['hdemucs_conf']['cac']
    exp_name  = conf['train']['exp_name']
    exp_name  = f"{exp_name}_loss={conf_loss}_channels={conf_chan}_perm={conf_perm}_cac={conf_cac}"

    print(f"\n----------------------------------------------------------------")
    print(f"Running experiment {exp_name}")
    print(f"----------------------------------------------------------------\n")

    # define the optimizer parameters
    optimizer = bnb.optim.Adam8bit(model.parameters(), **conf['optimizer_conf'])

    # instantiate the lightning module
    lightning_model = LightningModel(
                                    model = model,
                                    optimizer = optimizer,
                                    conf = conf,
                                    )
    
    # Initialize the TensorBoard logger
    tb_logger = TensorBoardLogger('lightning_logs/', name=exp_name)

    # save model with best val_loss metric
    best_checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(
                                    monitor = 'same_acoustics_same_layout_val_sdr',
                                    save_top_k = 1,
                                    mode = 'max',
                                    filename = 'best_{epoch:02d}'
                                    )

    every_5_epochs_checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(
                                    every_n_epochs = 10,
                                    save_top_k = -1,
                                    filename = 'checkpoint_{epoch:02d}'
                                    )

    # launch training
    trainer = pl.Trainer(
                        max_epochs = conf['train']['num_epochs'],
                        enable_checkpointing = True,
                        log_every_n_steps = 1,
                        # precision='16-mixed',
                        accelerator = "auto",
                        logger = tb_logger,
                        callbacks = [
                                best_checkpoint_callback,
                                every_5_epochs_checkpoint_callback],
                        # num_sanity_val_steps=0,
                        # detect_anomaly=True,
                        gradient_clip_val = 5.0,
                        )
    trainer.fit(lightning_model, train_loader, val_loaders)

if __name__ == '__main__':
    main()
