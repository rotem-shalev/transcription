from text_to_pose.pred import visualize_seq

from typing import List
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import os

EPSILON = 1e-4
START_LEARNING_RATE = 1e-3
MAX_SEQ_LEN = 200
output_dir_base = f"/home/nlp/rotemsh/transcription/videos"


def masked_mse_loss(pose: torch.Tensor, pose_hat: torch.Tensor, confidence: torch.Tensor, model_num_steps: int):
    # Loss by confidence. If missing joint, no loss. If less likely joint, less gradients.
    sq_error = torch.pow(pose - pose_hat, 2).sum(-1)
    num_steps_norm = np.log(model_num_steps) ** 2  # normalization of the loss by the model's num_steps
    return (sq_error * confidence).mean() * num_steps_norm


def mse_loss(pose: torch.Tensor, pose_hat: torch.Tensor, model_num_steps: int):
    # Loss by confidence. If missing joint, no loss. If less likely joint, less gradients.
    sq_error = torch.pow(pose - pose_hat, 2).sum(-1)
    num_steps_norm = np.log(model_num_steps) ** 2  # normalization of the loss by the model's num_steps
    return sq_error.mean() * num_steps_norm


class IterativeTextGuidedPoseGenerationModel(pl.LightningModule):
    def __init__(
            self,
            tokenizer,
            pose_dims: (int, int) = (137, 2),
            hidden_dim: int = 128,
            text_encoder_depth: int = 2,
            pose_encoder_depth: int = 4,
            encoder_heads: int = 2,
            encoder_dim_feedforward: int = 2048,
            max_seq_size: int = MAX_SEQ_LEN,
            num_steps: int = 50,
            tf_p: float = 0.5,
            lr: int = START_LEARNING_RATE,
            masked_loss: bool = True
    ):
        super().__init__()
        self.lr = lr
        self.tf_p = tf_p
        self.tokenizer = tokenizer
        self.max_seq_size = max_seq_size
        self.num_steps = num_steps
        self.hidden_dim = hidden_dim
        self.pose_dims = pose_dims
        self.masked_loss = masked_loss
        # self.parallel = parallel
        pose_dim = int(np.prod(pose_dims))

        # Embedding layers
        self.positional_embeddings = nn.Embedding(
            num_embeddings=max_seq_size, embedding_dim=hidden_dim
        )
        # positional embedding scalar
        self.alpha_pose = nn.Parameter(torch.randn(1))
        self.alpha_text = nn.Parameter(torch.randn(1))

        self.step_embedding = nn.Embedding(
            num_embeddings=num_steps, embedding_dim=hidden_dim
        )

        self.embedding = nn.Embedding(
            num_embeddings=len(tokenizer),
            embedding_dim=hidden_dim,
            padding_idx=tokenizer.pad_token_id,
        )

        self.first_pose = nn.Parameter(torch.randn(pose_dims))

        self.pose_projection = nn.Linear(pose_dim, hidden_dim)
        # if self.parallel:
        #     self.pose_embedding_projection = nn.Linear(hidden_dim * MAX_SEQ_LEN, hidden_dim)
        #     self.text_projection = nn.Linear(hidden_dim * MAX_SEQ_LEN, hidden_dim)
        #
        #     # Encoder
        #     text_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=encoder_heads,
        #                                                     dim_feedforward=encoder_dim_feedforward)
        #     pose_encoder_layer = nn.TransformerEncoderLayer(d_model=3*hidden_dim, nhead=encoder_heads,
        #                                                     dim_feedforward=encoder_dim_feedforward)  # , batch_first=True)
        #     self.text_encoder = nn.TransformerEncoder(text_encoder_layer, num_layers=text_encoder_depth)
        #     self.pose_encoder = nn.TransformerEncoder(pose_encoder_layer, num_layers=pose_encoder_depth)
        #
        # else:
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=encoder_heads,
                                                    dim_feedforward=encoder_dim_feedforward)  # , batch_first=True)
        self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=text_encoder_depth)
        self.pose_encoder = nn.TransformerEncoder(encoder_layer, num_layers=pose_encoder_depth)

        # step encoder
        self.step_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )

        # Predict sequence length
        self.seq_length = nn.Linear(hidden_dim, 1)

        # Predict pose difference
        pose_diff_projection_input_size = hidden_dim  # 3*hidden_dim if self.parallel else hidden_dim
        pose_diff_projection_output_size = pose_dim  # MAX_SEQ_LEN * pose_dim if self.parallel else pose_dim
        self.pose_diff_projection = nn.Sequential(
            nn.Linear(pose_diff_projection_input_size, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, pose_diff_projection_output_size),
        )

    def encode_text(self, texts: List[str]):
        tokenized = self.tokenizer(texts, device=self.device)
        positional_embedding = self.positional_embeddings(tokenized["positions"])
        embedding = self.embedding(tokenized["tokens_ids"]) + self.alpha_text * positional_embedding
        encoded = self.text_encoder(embedding.transpose(0, 1),
                                    src_key_padding_mask=tokenized["attention_mask"]).transpose(0, 1)
        seq_length = self.seq_length(encoded).mean(axis=1)
        return {"data": encoded, "mask": tokenized["attention_mask"]}, seq_length

    def forward(self, text: str, first_pose: torch.Tensor = None):
        first_pose = self.first_pose
        text_encoding, sequence_length = self.encode_text([text])
        sequence_length = round(float(sequence_length))

        pose_sequence = {
            "data": first_pose.expand(1, sequence_length, *self.pose_dims),
            "mask": torch.zeros([1, sequence_length], dtype=torch.bool),
        }
        positions = torch.arange(0, min(sequence_length, self.max_seq_size), dtype=torch.long,
                                 device=self.device)
        positional_embedding = self.positional_embeddings(positions)

        step_num = 0
        while True:
            yield pose_sequence["data"][0]
            # if self.parallel:
            #     pose_sequence["data"] = self.refinement_step_parallel(step_num, pose_sequence,
            #                                                 text_encoding, positional_embedding)[0]
            # else:
            pose_sequence["data"] = self.refinement_step(step_num, pose_sequence, text_encoding,
                                                         positional_embedding)[0]
            step_num += 1

    # def refinement_step_parallel(self, step_num, pose_sequence, text_encoding, positional_embedding):
    #     pose_sequence["data"] = pose_sequence["data"].detach()  # Detach from graph
    #     seq_len = pose_sequence["data"].size(1)
    #     change_pred = self.refine_pose_sequence_parallel(pose_sequence, text_encoding, positional_embedding,
    #                                             step_num)[:, :seq_len]
    #     cur_step_size = self.get_step_size(step_num)
    #     prev_step_size = self.get_step_size(step_num - 1) if step_num > 1 else 0
    #     step_size = cur_step_size - prev_step_size
    #     pred = (1 - step_size) * pose_sequence["data"] + step_size * change_pred
    #     return pred, cur_step_size

    def refinement_step(self, step_num, pose_sequence, text_encoding, positional_embedding):
        batch_size = pose_sequence["data"].shape[0]
        pose_sequence["data"] = pose_sequence["data"].detach()  # Detach from graph
        batch_step_num = torch.repeat_interleave(torch.LongTensor([step_num]),
                                                 batch_size).unsqueeze(1).to(self.device)
        step_encoding = self.step_encoder(self.step_embedding(batch_step_num))
        change_pred = self.refine_pose_sequence(pose_sequence, text_encoding, positional_embedding,
                                                 step_encoding)
        cur_step_size = self.get_step_size(step_num)
        prev_step_size = self.get_step_size(step_num-1) if step_num > 1 else 0
        step_size = cur_step_size-prev_step_size
        pred = (1-step_size) * pose_sequence["data"] + step_size * change_pred
        return pred, cur_step_size

    # def refine_pose_sequence_parallel(self, pose_sequence, text_encoding, positional_embedding, step_num):
    #     batch_size, seq_length, _, _ = pose_sequence["data"].shape
    #     flat_pose_data = pose_sequence["data"].reshape(batch_size, seq_length, -1)
    #
    #     # Encode pose sequence
    #     pose_embedding = self.pose_projection(flat_pose_data) + self.alpha_pose * positional_embedding
    #
    #     flat_pose_embedding = pose_embedding.reshape(batch_size, 1, seq_length, -1)
    #     padded_flat_pose_embedding = F.pad(flat_pose_embedding, (0, 0, 0, MAX_SEQ_LEN - seq_length),
    #                                        "constant", 0).reshape(batch_size, 1, -1)
    #     flat_pose_embedding_proj = self.pose_embedding_projection(padded_flat_pose_embedding)
    #
    #     flat_padded_text = F.pad(text_encoding["data"], (0, 0, 0, MAX_SEQ_LEN - text_encoding["data"].size(1)),
    #                              "constant", 0).repeat(1, 1, 1, 1).transpose(0, 1). \
    #         reshape(batch_size, 1, -1)
    #     flat_padded_text_projection = self.text_projection(flat_padded_text)
    #
    #     batch_step_num = torch.LongTensor([step_num], device=self.device).repeat(batch_size, 1)
    #     encoded_step = self.step_encoder(self.step_embedding(batch_step_num))
    #     pose_text_step_seq = torch.cat([flat_pose_embedding_proj, flat_padded_text_projection, encoded_step],
    #                                    dim=2)
    #
    #     mask = torch.ones(1, 1, dtype=torch.bool, device=self.device).fill_diagonal_(False)
    #     pose_encoding = self.pose_encoder(pose_text_step_seq.transpose(0, 1), mask=mask).transpose(0, 1)
    #
    #     # Predict desired change
    #     flat_pose_projection = self.pose_diff_projection(pose_encoding)
    #     return flat_pose_projection.reshape(batch_size, MAX_SEQ_LEN, *self.pose_dims)

    def refine_pose_sequence(self, pose_sequence, text_encoding, positional_embedding, step_encoding):
        batch_size, seq_length, _, _ = pose_sequence["data"].shape
        flat_pose_data = pose_sequence["data"].reshape(batch_size, seq_length, -1)

        # Encode pose sequence
        pose_embedding = self.pose_projection(flat_pose_data) + self.alpha_pose * positional_embedding
        pose_text_sequence = torch.cat([pose_embedding, text_encoding["data"], step_encoding], dim=1)
        step_mask = torch.zeros([step_encoding.size(0), 1], dtype=torch.bool, device=self.device)
        pose_text_mask = torch.cat(
            [pose_sequence["mask"], text_encoding["mask"], step_mask], dim=1
        )
        pose_encoding = self.pose_encoder(
            pose_text_sequence.transpose(0, 1), src_key_padding_mask=pose_text_mask
        ).transpose(0, 1)[:, :seq_length, :]

        # Predict desired change
        flat_pose_projection = self.pose_diff_projection(pose_encoding)
        return flat_pose_projection.reshape(batch_size, seq_length, *self.pose_dims)

    def get_step_size(self, step_num):
        if step_num < 2:
            return 0.1
        else:
            return np.log(step_num) / np.log(self.num_steps)

    def training_step(self, batch, *unused_args):
        return self.step(batch, *unused_args, phase="train")

    def validation_step(self, batch, *unused_args):
        return self.step(batch, *unused_args, phase="validation")

    # def parallel_step(self, batch, gamma, phase):
    #     text_encoding, sequence_length = self.encode_text(batch["text"])
    #     pose = batch["pose"]
    #
    #     # Calculate sequence length loss
    #     sequence_length_loss = F.mse_loss(sequence_length, pose["length"]) / 10000
    #
    #     # Repeat the first frame for initial prediction
    #     batch_size, pose_seq_length, _, _ = pose["data"].shape
    #     pose_sequence = {
    #         "data": torch.stack([pose["data"][:, 0]] * pose_seq_length, dim=1),
    #         "mask": torch.logical_not(pose["inverse_mask"])
    #     }
    #     all_steps_size = [self.get_step_size(0), self.get_step_size(1)] + \
    #                      [self.get_step_size(step_num) - self.get_step_size(step_num - 1)
    #                       for step_num in range(2, self.num_steps)]
    #
    #     all_steps_label = torch.stack([step_size * pose["data"] +
    #                                    (1 - step_size) * pose_sequence["data"]
    #                                    for step_size in all_steps_size], dim=1)
    #     flat_pose_data = all_steps_label.reshape(batch_size, self.num_steps, pose_seq_length, -1)
    #     if phase == "train":  # add noise during training
    #         flat_pose_data += torch.randn_like(flat_pose_data) * EPSILON
    #     positions = torch.arange(0, pose_seq_length, dtype=torch.long,
    #                              device=self.device).repeat(self.num_steps, 1)
    #     positional_embedding = self.positional_embeddings(positions)
    #     pose_embedding = self.pose_projection(flat_pose_data) + self.alpha_pose * positional_embedding
    #
    #     flat_pose_embedding = pose_embedding.reshape(batch_size, self.num_steps, pose_seq_length, -1)
    #     padded_flat_pose_embedding = F.pad(flat_pose_embedding, (0, 0, 0, MAX_SEQ_LEN - pose_seq_length),
    #                                        "constant", 0).reshape(batch_size, self.num_steps, -1)
    #     flat_pose_embedding_proj = self.pose_embedding_projection(padded_flat_pose_embedding)
    #
    #     flat_padded_text = F.pad(text_encoding["data"], (0, 0, 0, MAX_SEQ_LEN - text_encoding["data"].size(1)),
    #                              "constant", 0).repeat(self.num_steps, 1, 1, 1).transpose(0, 1). \
    #         reshape(batch_size, self.num_steps, -1)
    #     flat_padded_text_projection = self.text_projection(flat_padded_text)
    #     batch_step_num = torch.arange(self.num_steps, device=self.device).repeat(batch_size, 1)
    #     encoded_steps = self.step_encoder(self.step_embedding(batch_step_num))
    #     pose_text_step_seq = torch.cat([flat_pose_embedding_proj, flat_padded_text_projection, encoded_steps],
    #                               dim=2)
    #
    #     mask = torch.ones(self.num_steps, self.num_steps, dtype=torch.bool, device=self.device).fill_diagonal_(False)
    #     pose_encoding = self.pose_encoder(pose_text_step_seq.transpose(0, 1), mask=mask).transpose(0, 1)
    #
    #     # Predict desired change
    #     flat_pose_projection = self.pose_diff_projection(pose_encoding)
    #     preds = flat_pose_projection.reshape(batch_size, self.num_steps, MAX_SEQ_LEN,
    #                                          *self.pose_dims)[:, :, :pose_seq_length]
    #
    #     confidence = pose["confidence"].repeat(self.num_steps, 1, 1, 1).transpose(0, 1)
    #     refinement_loss = masked_mse_loss(all_steps_label, preds, confidence, self.num_steps)
    #     loss = refinement_loss + gamma * sequence_length_loss
    #
    #     self.log(phase + "_seq_length_loss", sequence_length_loss, batch_size=batch_size)
    #     self.log(phase + "_refinement_loss", refinement_loss, batch_size=batch_size)
    #     self.log(phase + "_loss", loss, batch_size=batch_size)
    #
    #     return loss

    def step(self, batch, *unused_args, phase: str, gamma: float = 0.2, k: int = 5):
        """
        @param batch: data batch
        @param phase: either "train" or "validation"
        @param gamma: float between 0 and 1, determines the weight of the sequence length loss. default is 0.2.
        @param k: train seq len every k epochs. default is 5.
        """
        if self.global_step == 0:
            self.first_pose = nn.Parameter(torch.mean(batch["pose"]["data"][:, 0], axis=0))

        # if self.parallel:
        #     return self.parallel_step(batch, gamma, phase)

        text_encoding, sequence_length = self.encode_text(batch["text"])
        pose = batch["pose"]

        # Repeat the first frame for initial prediction
        batch_size, pose_seq_length, _, _ = pose["data"].shape
        teacher_forcing = torch.rand(1) < self.tf_p  # teacher force at the sequence level
        if teacher_forcing or phase == "validation":
            pose_sequence = {
                "data": torch.stack([pose["data"][:, 0]] * pose_seq_length, dim=1),
                "mask": torch.logical_not(pose["inverse_mask"])
            }
        else:
            pose_sequence = {
                "data": torch.stack([torch.repeat_interleave(self.first_pose.unsqueeze(0), batch_size, 0)] *
                                    pose_seq_length, dim=1),
                "mask": torch.logical_not(pose["inverse_mask"])
            }

        positions = torch.arange(0, pose_seq_length, dtype=torch.long, device=self.device)
        positional_embedding = self.positional_embeddings(positions)

        refinement_loss = 0
        for i in range(self.num_steps):
            pred, step_size = self.refinement_step(i, pose_sequence, text_encoding, positional_embedding)
            l1_gold = step_size * pose["data"] + (1 - step_size) * pose_sequence["data"]

            if self.masked_loss:
                refinement_loss += masked_mse_loss(l1_gold, pred, pose["confidence"], self.num_steps)
            else:
                refinement_loss += mse_loss(l1_gold, pred, self.num_steps)

            teacher_forcing_step_level = torch.rand(1) < self.tf_p
            pose_sequence["data"] = l1_gold if phase == "validation" or teacher_forcing_step_level else pred

            if phase == "train":  # add just a little noise while training
                pose_sequence["data"] = pose_sequence["data"] + torch.randn_like(pose_sequence["data"]) * EPSILON

        loss = refinement_loss

        if self.current_epoch % k == 0:
            # Calculate sequence length loss
            sequence_length_loss = F.mse_loss(sequence_length, pose["length"]) / 10000
            loss += gamma*sequence_length_loss
            self.log(phase + "_seq_length_loss", sequence_length_loss, batch_size=batch_size)

        self.log(phase + "_refinement_loss", refinement_loss, batch_size=batch_size)
        self.log(phase + "_loss", loss, batch_size=batch_size)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
