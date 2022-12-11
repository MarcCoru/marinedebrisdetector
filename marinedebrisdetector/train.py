
from datetime import datetime
import os
import argparse

from data.plastic_litter_project import PLPDataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from callbacks import PLPCallback, RefinedRegionsQualitativeCallback
from data.marinedebrisdatamodule import MarineDebrisDataModule
from model.segmentation_model import SegmentationModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default="/data/marinedebris")
    parser.add_argument('--project', type=str, default="flobs-segm")
    parser.add_argument('--run-name', type=str, default=None)
    parser.add_argument('--model', type=str, default="unet")
    parser.add_argument('--resume-from', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--weight-decay', type=float, default=1e-6)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--augmentation-intensity', type=int, default=1, help="number indicating intensity 0, 1 (noise), 2 (channel shuffle)")
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--device', type=str, choices=["cpu", "cuda"], default="cuda")
    parser.add_argument('--hr-only', action="store_true")
    parser.add_argument('--no-checkpoint', action="store_true")
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--download', action="store_true")

    # label-refinement
    parser.add_argument('--no-label-refinement', action="store_true")
    parser.add_argument('--no-s2ships', action="store_true")
    parser.add_argument('--no-marida', action="store_true")

    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--pos-weight', type=float, default=1, help="positional weight for the floating object class, large values counteract")

    args = parser.parse_args()

    return args

def main(args):
    pl.seed_everything(args.seed)

    model = SegmentationModel(args)

    marinedebris_datamodule = MarineDebrisDataModule(data_root=args.data_path,
                                        image_size=args.image_size,
                                        workers=args.workers,
                                        batch_size=args.batch_size,
                                        no_label_refinement=args.no_label_refinement,
                                        no_s2ships=args.no_s2ships,
                                        no_marida=args.no_marida,
                                        download=args.download)

    marinedebris_datamodule.prepare_data()

    if args.run_name is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        run_name = f"{args.model}_{ts}"
    else:
        run_name = args.run_name

    logger = WandbLogger(project=args.project, name=run_name, log_model=True, save_code=True)
    #logger.watch(model)

    checkpointer = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(os.getcwd(), "checkpoints", args.project, run_name),
        filename="{epoch}-{val_loss:.2f}-{auroc:.3f}",
        save_top_k=3,
        monitor="auroc",
        mode="max",
        save_last=False,
    )

    plp_callback_2021 = PLPCallback(logger, marinedebris_datamodule.get_plp_dataset(2021))
    plp_callback_2022 = PLPCallback(logger, marinedebris_datamodule.get_plp_dataset(2022))

    qual_image_callback = RefinedRegionsQualitativeCallback(logger,
                                                            marinedebris_datamodule.get_qualitative_validation_dataset())

    trainer = pl.Trainer(accelerator="gpu", logger=logger, devices=1,
                         callbacks=[checkpointer,
                                    plp_callback_2021,
                                    plp_callback_2022,
                                    qual_image_callback],
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
