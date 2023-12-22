import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import RichProgressBar
from csdp_pipeline.pipeline_elements.pipeline_dataset import PipelineDataset
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.neptune import NeptuneLogger
from csdp_pipeline.pipeline_elements.full_data_samplers import Full_Eval_Dataset_Sampler, Full_Train_Dataset_Sampler
import os
import neptune
from csdp_training.lightning_models.usleep import USleep_Lightning

class EESM_Channel_Combiner():
    def process(self, x):
        signal_data, labels, meta = x

        signal_data: torch.tensor = signal_data

        eeg = signal_data
        eog = signal_data

        return eeg, eog, labels, f"eesm/{meta}/{meta}"

environment = "LOCAL"

logging_enabled = False

api_key = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5YzViZjJlYy00NDNhLTRhN2EtOGZmYy00NDEzODBmNTgxYzMifQ=="
project = "NTLAB/ear-eeg"
name = "ear-eeg"

num_workers = 0

finetune_max_epochs = 1
window_size = 35
batch_size = 64
lr = 0.000001
early_stopping_patience = 200
persistent_workers = False

if environment == "LOCAL":
    base_repo_path = "C:/Users/au588953/Git Repos/CSDP demonstration"
    checkpoint_path = "C:/Users/au588953/eesm_test/epoch=571-step=253396.ckpt"
    eesm_base_split_path = "C:/Users/au588953/Git Repos/CSDP demonstration/base_eesm_test.json"
    accelerator = "cpu"
    dataset_path = "C:/Users/au588953/eesm_test/eesm.hdf5"
    logging_folder = "C:/Users/au588953"
    all_test_split_path = "C:/Users/au588953/Git Repos/CSDP demonstration/cross_val"

elif environment == "PRIME":
    base_repo_path = "/home/js/repos/usleep-eareeg-filter"
    checkpoint_path = "/home/com/ecent/NOBACKUP/.neptune/ear-eeg/EAR-31/checkpoints/epoch=156-step=69551.ckpt"
    eesm_base_split_path = "/home/js/repos/usleep-eareeg-filter/splits/cross_val"
    accelerator = "gpu"
    dataset_path = "/com/ecent/NOBACKUP/EESM_from_ERDA/new/eesm.hdf5"
    logging_folder = "/com/ecent/NOBACKUP"
    all_test_split_path = "TODO"

elif environment == "LUMI":
    base_repo_path = "/home/js/repos/usleep-eareeg-filter"
    checkpoint_path = "/home/com/ecent/NOBACKUP/.neptune/ear-eeg/EAR-31/checkpoints/epoch=156-step=69551.ckpt"
    eesm_base_split_path = "/home/js/repos/usleep-eareeg-filter/splits/cross_val"
    accelerator = "gpu"
    dataset_path = "/com/ecent/NOBACKUP/EESM_from_ERDA/new/eesm.hdf5"
    logging_folder = "/com/ecent/NOBACKUP"
    all_test_split_path = "TODO"

def main():
    org = os.getcwd()
    os.chdir(logging_folder)

    torch.set_float32_matmul_precision('high')

    net = USleep_Lightning.load_from_checkpoint(checkpoint_path,
                                                map_location= torch.device(accelerator))

    trainer = init_trainer("", None, 0)

    test(net, trainer, split=eesm_base_split_path)

    run_cross_validation()

    os.chdir(org)

def run_cross_validation():

    splits = os.listdir(all_test_split_path)

    run = init_logging_run()

    for split in splits:
        split_file_name = f"{all_test_split_path}/{split}"

        new_net = USleep_Lightning.load_from_checkpoint(checkpoint_path,
                                                        map_location= torch.device(accelerator))

        trainer = init_trainer(split, run, max_epochs=finetune_max_epochs)

        trainer, net = finetune(new_net, trainer, split_file_name)

        test(net, trainer, split_file_name)

def init_logging_run():
    if logging_enabled == True:
        try:
            run = neptune.init_run(project=project,
                                   api_token=api_key,
                                   name=name,
                                   mode="sync")
        except:
            print("Error: No valid neptune logging credentials configured.")
            exit()
    else:
        run = None

    return run

def init_trainer(log_prefix, run, max_epochs):
    richbar = RichProgressBar()

    early_stopping = pl.callbacks.EarlyStopping(
        monitor="valKap",
        min_delta=0.00,
        patience=early_stopping_patience,
        verbose=True,
        mode="max"
    )

    checkpoint_callback = ModelCheckpoint(filename=f"best-{log_prefix}", monitor="valKap", mode="max")

    callbacks = [richbar,
                 early_stopping,
                 checkpoint_callback]

    if logging_enabled:
        logger = NeptuneLogger(run=run,
                               prefix=log_prefix)
    else:
        logger = True

    trainer = pl.Trainer(logger=logger,
                         max_epochs=max_epochs,
                         callbacks=callbacks,
                         accelerator=accelerator,
                         devices=1,
                         num_nodes=1)
    
    return trainer

def finetune(net, trainer, split, train_records = None, val_records = None):
    train_sampler = Full_Train_Dataset_Sampler(dataset_path,
                                               window_size,
                                               train_records,
                                               ["EEG_EarEEGRef.COMB_AVG-EarEEGRef.REF"],
                                               split,
                                               dataset_name="eesm")
    
    val_sampler = Full_Eval_Dataset_Sampler(dataset_path,
                                            val_records,
                                            ["EEG_EarEEGRef.COMB_AVG-EarEEGRef.REF"],
                                            split,
                                            "val",
                                            "eesm")

    train_pipes = [train_sampler, EESM_Channel_Combiner()]
    val_pipes = [val_sampler, EESM_Channel_Combiner()]

    train_set = PipelineDataset(train_pipes,
                                iterations=train_pipes[0].num_windows)

    val_set = PipelineDataset(val_pipes,
                              iterations=len(val_pipes[0].record_hyps))

    train_loader = DataLoader(train_set,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=num_workers,
                             drop_last=False,
                             pin_memory=True,
                             persistent_workers=persistent_workers)
    
    val_loader = DataLoader(val_set,
                            batch_size=1,
                            num_workers=num_workers,
                            persistent_workers=persistent_workers)

    trainer.fit(net, train_loader, val_loader)

    return trainer, net

def test(net, trainer: pl.Trainer, split, test_records = None):
    test_sampler = Full_Eval_Dataset_Sampler(dataset_path,
                                             test_records,
                                             ["EEG_EarEEGRef.COMB_AVG-EarEEGRef.REF"], 
                                             split,
                                             "test",
                                             "eesm")

    test_pipes = [test_sampler, EESM_Channel_Combiner()]

    test_set = PipelineDataset(test_pipes,
                               len(test_pipes[0].record_hyps))

    test_loader = DataLoader(test_set,
                                batch_size=1,
                                shuffle=False,
                                num_workers=num_workers)

    _ = trainer.test(net, test_loader,)

if __name__ == '__main__':
    main()
    
