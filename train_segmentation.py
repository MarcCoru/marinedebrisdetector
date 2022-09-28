
from datetime import datetime
import os
import argparse

from data.plastic_litter_project import PLPDataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from callbacks import PLPCallback
from data.marinedebrisdatamodule import MarineDebrisDataModule
from model.segmentation_model import SegmentationModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default="/data/marinedebris")
    parser.add_argument('--model', type=str, default="unet")
    parser.add_argument('--resume-from', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--weight-decay', type=float, default=1e-12)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--augmentation-intensity', type=int, default=1, help="number indicating intensity 0, 1 (noise), 2 (channel shuffle)")
    parser.add_argument('--cache-to-numpy', action="store_true", help="performance optimization: caches images to npz files in a npy folder within data-path.")
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--device', type=str, choices=["cpu", "cuda"], default="cuda")
    parser.add_argument('--no-checkpoint', action="store_true")
    parser.add_argument('--max-epochs', type=int, default=100)

    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--pos-weight', type=float, default=1, help="positional weight for the floating object class, large values counteract")

    args = parser.parse_args()

    return args

def main(args):

    model = SegmentationModel(args.model,
                              learning_rate=args.learning_rate,
                              weight_decay=args.weight_decay)

    marinedebris_datamodule = MarineDebrisDataModule(data_root=args.data_path,
                                   image_size=args.image_size,
                                   workers=args.workers,
                                   batch_size=args.batch_size)

    ts = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    run_name = f"{args.model}_{ts}"
    logger = WandbLogger(project="flobs-segm", name=run_name, log_model=True, save_code=True)
    #logger.watch(model)

    checkpointer = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(os.getcwd(), "checkpoints", run_name),
        filename="{epoch}-{val_loss:.2f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )

    plp_dataset = PLPDataset(root="/data/marinedebris/PLP", year=2022, output_size=32)
    plp_callback = PLPCallback(logger, plp_dataset)

    trainer = pl.Trainer(accelerator="gpu", logger=logger, devices=1,
                         callbacks=[checkpointer,
                                    plp_callback],
                         max_epochs=args.max_epochs,
                         fast_dev_run=False)

    if args.no_checkpoint:
        ckpt_path = None
    else:
        if args.resume_from is not None:
            ckpt_path = args.resume_from
        else:
            ckpt_path = f"checkpoints/{run_name}/last.ckpt"
            if not os.path.exists(ckpt_path):
                ckpt_path = None
        print(f"checkpointing/resuming from {ckpt_path}")

    trainer.fit(model, marinedebris_datamodule, ckpt_path=ckpt_path)
    #trainer.test(model, marinedebris_datamodule, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main(parse_args())
