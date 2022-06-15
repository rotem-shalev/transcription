import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, '/home/nlp/rotemsh/transcription')

from shared.collator import zero_pad_collator
from text_to_pose.args import args
from text_to_pose.data import get_dataset
from text_to_pose.model import IterativeTextGuidedPoseGenerationModel
from text_to_pose.tokenizers.hamnosys.hamnosys_tokenizer import HamNoSysTokenizer
from text_to_pose.pred import pred

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"#"0,1,2,3"
    # args.gpus = 4

    LOGGER = None
    if not args.no_wandb:
        LOGGER = WandbLogger(project="text-to-pose", log_model=False, offline=False)
        if LOGGER.experiment.sweep_id is None:
            LOGGER.log_hyperparams(args)

    args.batch_size = 4
    args.num_steps = 100

    train_dataset = get_dataset(name=args.dataset, poses=args.pose, fps=args.fps,
                                components=args.pose_components,
                                max_seq_size=args.max_seq_size, split="train[10:]")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, #num_workers=8,
                              shuffle=True, collate_fn=zero_pad_collator)

    validation_dataset = get_dataset(name=args.dataset, poses=args.pose, fps=args.fps,
                                     components=args.pose_components,
                                     max_seq_size=args.max_seq_size, split="train[:10]")
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size,
                                   collate_fn=zero_pad_collator)#, num_workers=8)

    _, num_pose_joints, num_pose_dims = train_dataset[0]["pose"]["data"].shape

    # Model Arguments
    model_args = dict(tokenizer=HamNoSysTokenizer(),
                      pose_dims=(num_pose_joints, num_pose_dims),
                      hidden_dim=args.hidden_dim,
                      text_encoder_depth=args.text_encoder_depth,
                      pose_encoder_depth=args.pose_encoder_depth,
                      encoder_heads=args.encoder_heads,
                      max_seq_size=args.max_seq_size,
                      num_steps=args.num_steps)

    experiment_name = "new_step_method2"
    args.checkpoint = f"/home/nlp/rotemsh/transcription/models/{experiment_name}/model.ckpt"
    if args.checkpoint is not None:
        model = IterativeTextGuidedPoseGenerationModel.load_from_checkpoint(args.checkpoint, **model_args)
    else:
        model = IterativeTextGuidedPoseGenerationModel(**model_args)

    callbacks = []
    if LOGGER is not None:
        os.makedirs("models", exist_ok=True)
        callbacks.append(ModelCheckpoint(
            dirpath="models/" + experiment_name,#LOGGER.experiment.id,
            filename="model",
            verbose=True,
            save_top_k=1,
            monitor='train_loss',
            mode='min'
        ))

    trainer = pl.Trainer(
        max_epochs=100,
        logger=LOGGER,
        callbacks=callbacks,
        gpus=args.gpus,
        accumulate_grad_batches=8,
        strategy="ddp"
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=validation_loader)
    output_dir = f"/home/nlp/rotemsh/transcription/text_to_pose/videos/{experiment_name}"
    pred(model, train_dataset, output_dir)
    pred(model, validation_dataset, output_dir)
