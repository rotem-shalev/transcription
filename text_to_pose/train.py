import os
import numpy as np
import tensorflow_datasets as tfds
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

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
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    args.gpus = 1
    experiment_name = "reproduce_exclude_sep_seq_len_6"
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
        train_split = f"test[{test_size}:{test_size+20}]"
        test_split = train_split
    if "leave_out" in experiment_name:
        train_dataset, test_dataset = get_dataset(name=args.dataset, poses=args.pose, fps=args.fps,
                                    components=args.pose_components, exclude=True, leave_out="lsf",
                                    max_seq_size=args.max_seq_size, split=train_split)
    else:
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
                      num_pose_projection_layers=1,
                      do_pose_self_attention=False,
                      use_transformer_decoder=False,#True,
                      concat=True)

    # args.checkpoint = f"/home/nlp/rotemsh/transcription/models/{experiment_name}/model.ckpt"
    if args.checkpoint is not None:
        model = IterativeTextGuidedPoseGenerationModel.load_from_checkpoint(args.checkpoint, **model_args)
    else:
        model = IterativeTextGuidedPoseGenerationModel(**model_args)

    callbacks = []
    if LOGGER is not None:
        os.makedirs("models", exist_ok=True)
        callbacks.append(ModelCheckpoint(
            dirpath="/home/nlp/rotemsh/transcription/models/" + experiment_name,
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
        accelerator='gpu',
        devices=args.gpus,
        accumulate_grad_batches=batch_size_to_accumulate[args.batch_size],
        strategy="ddp"
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=validation_loader)

    # eval part- take best model saved
    args.checkpoint = f"/home/nlp/rotemsh/transcription/models/{experiment_name}/model.ckpt"
    model = IterativeTextGuidedPoseGenerationModel.load_from_checkpoint(args.checkpoint, **model_args)
    model.eval()

    # test seq_len_predictor
    diffs = []
    for d in validation_dataset:
        _, seq_len = model.encode_text([d["text"]])
        real_seq_len = len(d["pose"]["data"])
        diff = np.abs(real_seq_len-seq_len.item())
        if diff > 100:
            print(d["id"])
            print("real vs pred:", real_seq_len, seq_len.item())
        diffs.append(diff)
    print(np.mean(diffs), np.median(diffs), np.max(diffs))

    output_dir = f"/home/nlp/rotemsh/transcription/text_to_pose/videos/{experiment_name}"
    pred(model, train_dataset, os.path.join(output_dir, "train"))
    pred(model, validation_dataset, os.path.join(output_dir, "val"))
    # if "leave_out" in experiment_name:
    #     pred(model, test_dataset, os.path.join(output_dir, "test"))
