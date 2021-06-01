import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import utils
import glob
from system import TextGenerationTrainer

def train():
    parser = argparse.ArgumentParser()
    # experiment related
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--model-name-or-path", type=str, default="facebook/bart-base")
    parser.add_argument("--dataset-name", type=str, choices=['nyt_opinion', 'cmv'], required=True)
    parser.add_argument("--marginalization", type=str, choices=['seq', 'tok'],
                        help="Similar to RagToken and RagSequence models.")

    # input data related
    parser.add_argument("--max-sent-num", type=int, default=10)
    parser.add_argument("--max-entity-per-sentence", type=int, default=20)
    parser.add_argument("--max-concept-per-sentence", type=int, default=20)
    parser.add_argument("--no-entity", action="store_true")
    parser.add_argument("--no-claim", action="store_true")
    parser.add_argument("--no-concept", action="store_true")
    parser.add_argument("--no-pred-concept", action="store_true")

    # training related
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--weight-decay", type=float, default=0)

    parser.add_argument("--max-target-len", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--fp16-opt-level", type=str, default="O2")
    parser.add_argument("--multi-gpus", action="store_true")

    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=f'checkpoints/{args.exp_name}/' + '{epoch}-{val_loss:.2f}',
        monitor="val_loss",
        mode="min", save_top_k=5 if not args.debug else 1,
    )
    logger = TensorBoardLogger('tb_logs', name=args.exp_name)

    trainer_args = {
        "logger": logger,
        "gpus": 1,
        'checkpoint_callback': checkpoint_callback,
        "log_gpu_memory": True,
    }
    if args.fp16:
        trainer_args['precision'] = 16
        trainer_args['amp_level'] = args.fp16_opt_level

    if args.debug:
        trainer_args['log_every_n_steps'] = 5
        trainer_args['flush_logs_every_n_steps'] = 20

    if args.multi_gpus:
        trainer_args['gpus'] = 2

    old_ckpt = glob.glob(f"checkpoints/{args.exp_name}/epoch*")
    if len(old_ckpt) > 0:
        ckpt_path, _ = utils.load_latest_ckpt(args.exp_name)
        print(f"start training from {ckpt_path}")
        trainer_args['resume_from_checkpoint'] = ckpt_path
        model = TextGenerationTrainer.load_from_checkpoint(ckpt_path)
    else:
        model = TextGenerationTrainer(args)

    trainer = pl.Trainer.from_argparse_args(args, **trainer_args)
    trainer.fit(model)



if __name__=='__main__':
    train()
