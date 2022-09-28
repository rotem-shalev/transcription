import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import tensorflow_datasets as tfds
from torch.optim import Adam, SGD

import sys
sys.path.insert(0, '/home/nlp/rotemsh/transcription')

from shared.collator import zero_pad_collator
from text_to_pose.args import args
from text_to_pose.data import get_dataset
from text_to_pose.model import IterativeTextGuidedPoseGenerationModel
from text_to_pose.tokenizers.hamnosys.hamnosys_tokenizer import HamNoSysTokenizer
from text_to_pose.pred import pred, vis_label_only
from text_to_pose.constants import num_steps_to_batch_size, batch_size_to_accumulate, DATASET_SIZE


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"#"0"
    args.gpus = 1
    experiment_name = "reproduce_exclude_sep_2"
    print("experiment_name:", experiment_name)

    if experiment_name != "test":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        args.gpus = 4

    LOGGER = None
    if not args.no_wandb:
        LOGGER = WandbLogger(project="text-to-pose", log_model=False, offline=False, id=experiment_name)
        if LOGGER.experiment.sweep_id is None:
            LOGGER.log_hyperparams(args)
    args.batch_size = num_steps_to_batch_size[args.num_steps]
    # args.pose_components = ['hand_left_keypoints_2d', 'hand_right_keypoints_2d']

    test_size = int(0.1*DATASET_SIZE)
    print("test size", test_size)
    train_split = f'test[{test_size}:]+train'
    test_split = f'test[:{test_size}]'
    if experiment_name == "test":
        train_split = f"test[{test_size}:{test_size+50}]"
        test_split = train_split
    train_dataset = get_dataset(name=args.dataset, poses=args.pose, fps=args.fps,
                                components=args.pose_components, exclude=True,
                                max_seq_size=args.max_seq_size, split=train_split)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=zero_pad_collator)
    print("train set size:", len(train_dataset))

    validation_dataset = get_dataset(name=args.dataset, poses=args.pose, fps=args.fps,
                                     components=args.pose_components, exclude=True,
                                     max_seq_size=args.max_seq_size, split=test_split)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size,
                                   collate_fn=zero_pad_collator)

    _, num_pose_joints, num_pose_dims = train_dataset[0]["pose"]["data"].shape

    # Model Arguments
    model_args = dict(tokenizer=HamNoSysTokenizer(),
                      pose_dims=(num_pose_joints, num_pose_dims),
                      hidden_dim=128,#256,#args.hidden_dim,
                      text_encoder_depth=args.text_encoder_depth,
                      pose_encoder_depth=args.pose_encoder_depth,
                      encoder_heads=args.encoder_heads, #4,
                      # encoder_dim_feedforward=512,
                      max_seq_size=args.max_seq_size,
                      num_steps=args.num_steps,
                      tf_p=args.tf_p,
                      masked_loss=args.masked_loss,
                      optimizer_fn=Adam,  # TODO- convert to arg
                      separate_positional_embedding=args.separate_positional_embedding,
                      num_pose_projection_layers=1,#2,
                      do_pose_self_attention=False,#True,
                      use_transformer_decoder=False,#True,
                      concat=True)

    # args.checkpoint = f"models/{experiment_name}/model.ckpt"
    if args.checkpoint is not None:
        model = IterativeTextGuidedPoseGenerationModel.load_from_checkpoint(args.checkpoint, **model_args)
    else:
        model = IterativeTextGuidedPoseGenerationModel(**model_args)

    callbacks = []
    if LOGGER is not None:
        os.makedirs("models", exist_ok=True)
        callbacks.append(ModelCheckpoint(
            dirpath="models/" + experiment_name,
            filename="model",
            verbose=True,
            save_top_k=1,
            monitor='train_loss',
            mode='min'
        ))

    trainer = pl.Trainer(
        max_epochs=2000,
        logger=LOGGER,
        callbacks=callbacks,
        gpus=args.gpus,
        accumulate_grad_batches=batch_size_to_accumulate[args.batch_size],
        strategy="ddp"
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=validation_loader)

    # eval part- take best model saved
    args.checkpoint = f"models/{experiment_name}/model.ckpt"
    model = IterativeTextGuidedPoseGenerationModel.load_from_checkpoint(args.checkpoint, **model_args)
    model.eval()

    output_dir = f"/home/nlp/rotemsh/transcription/text_to_pose/videos/{experiment_name}"
    pred(model, train_dataset, os.path.join(output_dir, "train"), use_learned_first_pose=False)
    pred(model, validation_dataset, os.path.join(output_dir, "val"), use_learned_first_pose=False)
