import argparse
import os
import torch
from functools import partial
from lightning import Trainer, seed_everything
from lightning.pytorch import loggers, callbacks
from torch.optim import Adam
from de.dataio.proteins import ProteinsDataModule
from de.predictors.attention.module import ESM2_Attention, ESM2DecoderModule, AMix_Attention, AMixDecoderModule


def parse_args():
    parser = argparse.ArgumentParser(description="Train decoder.")
    parser.add_argument("--data_file",
                        type=str,
                        help="Path to data directory.")
    parser.add_argument("--dataset_name",
                        type=str,
                        help="Name of trained dataset.")
    parser.add_argument("--pretrained_encoder",
                        type=str,
                        default="facebook/esm2_t12_35M_UR50D",
                        help="Path to pretrained encoder.")
    parser.add_argument("--dec_hidden_dim",
                        type=int,
                        default=1280,
                        help="Hidden dim of decoder.")
    parser.add_argument("--batch_size",
                        type=int,
                        default=128,
                        help="Batch size.")
    parser.add_argument("--ckpt_path",
                        type=str,
                        help="Checkpoint of model.")
    parser.add_argument("--devices",
                        type=str,
                        default="-1",
                        help="Training devices separated by comma.")
    parser.add_argument("--output_dir",
                        type=str,
                        default="./exps",
                        help="Path to output directory.")
    parser.add_argument("--grad_accum_steps",
                        type=int,
                        default=1,
                        help="No. updates steps to accumulate the gradient.")
    parser.add_argument("--lr",
                        type=float,
                        default=1e-4,
                        help="Learning rate.")
    parser.add_argument("--num_epochs",
                        type=int,
                        default=30,
                        help="Number of epochs.")
    parser.add_argument("--wandb_project",
                        type=str,
                        default="directed_evolution",
                        help="WandB project's name.")
    parser.add_argument("--seed",
                        type=int,
                        default=0,
                        help="Random seed for reproducibility.")
    parser.add_argument("--set_seed_only",
                        action="store_true",
                        help="Whether to not set deterministic flag.")
    parser.add_argument("--num_workers",
                        type=int,
                        default=64,
                        help="No. workers.")
    parser.add_argument("--num_ckpts",
                        type=int,
                        default=5,
                        help="Maximum no. checkpoints can be saved.")
    parser.add_argument("--log_interval",
                        type=int,
                        default=100,
                        help="How often to log within steps.")
    parser.add_argument("--precision",
                        type=str,
                        choices=["highest", "high", "medium"],
                        default="highest",
                        help="Internal precision of float32 matrix multiplications.")
    parser.add_argument("--encoder_type",
                        type=str,
                        choices=["esm2", "amix"],
                        default="esm2",
                        help="Encoder type to use (esm2 or amix).")
    parser.add_argument("--amix_ckpt_path",
                        type=str,
                        help="Path to AMix checkpoint file (required when encoder_type is amix).")
    args = parser.parse_args()
    
    # Validate encoder arguments
    if args.encoder_type == "amix" and args.amix_ckpt_path is None:
        parser.error("--amix_ckpt_path is required when --encoder_type is amix")
    
    return args


def init_model(encoder_type, pretrained_encoder_or_ckpt, hidden_dim):
    if encoder_type == "amix":
        assert pretrained_encoder_or_ckpt is not None, "Encoder path must be provided when encoder_type is amix"
        model = AMix_Attention(pretrained_encoder_or_ckpt, hidden_dim)
    else:
        model = ESM2_Attention(pretrained_encoder_or_ckpt, hidden_dim)
    tokenizer = model.tokenizer
    model.freeze_encoder()
    return model, tokenizer


def train(args):
    seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision(args.precision)
    accelerator = "cpu" if args.devices == "-1" else "gpu"

    # Load model
    encoder_path = args.amix_ckpt_path if args.encoder_type == "amix" else args.pretrained_encoder
    model, tokenizer = init_model(args.encoder_type, encoder_path, args.dec_hidden_dim)
    # Init optimizer
    optim = partial(Adam, lr=args.lr)

    # ================== #
    # ====== Data ====== #
    # ================== #
    datamodule = ProteinsDataModule(
        csv_file=args.data_file,
        tokenizer=tokenizer,
        train_batch_size=args.batch_size,
        valid_batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # ==================== #
    #  ====== Model ====== #
    # ==================== #
    if args.encoder_type == "amix":
        module = AMixDecoderModule(model, optim)
    else:
        module = ESM2DecoderModule(model, optim)

    # ====================== #
    # ====== Training ====== #
    # ====================== #
    logger_list = [
        loggers.CSVLogger(args.output_dir),
        loggers.WandbLogger(save_dir=args.output_dir,
                            project=args.wandb_project)
    ]
    if args.encoder_type == "amix":
        prefix = f"amix-dec_{args.dec_hidden_dim}"
    else:
        prefix = args.pretrained_encoder.split("/")[-1] + f"-dec_{args.dec_hidden_dim}"
    callback_list = [
        callbacks.RichModelSummary(),
        callbacks.RichProgressBar(),
        callbacks.ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, "checkpoints"),
            filename=f"{prefix}-{args.dataset_name}_" +
                     "{epoch:02d}-{train_loss:.3f}-{val_loss:.3f}",
            monitor="val_loss",
            verbose=True,
            save_top_k=args.num_ckpts,
            save_weights_only=False,
            every_n_epochs=1,
        )
    ]

    trainer = Trainer(
        accelerator=accelerator,
        devices=[int(d) for d in args.devices.split(",")],
        max_epochs=args.num_epochs,
        log_every_n_steps=args.log_interval,
        accumulate_grad_batches=args.grad_accum_steps,
        deterministic=not args.set_seed_only,
        default_root_dir=args.output_dir,
        logger=logger_list,
        callbacks=callback_list,
    )

    trainer.fit(module, datamodule=datamodule, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    args = parse_args()
    train(args)
