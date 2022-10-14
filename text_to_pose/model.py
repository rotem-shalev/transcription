from typing import List
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from torch.optim import optimizer, Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau


EPSILON = 1e-4
START_LEARNING_RATE = 1e-3
MAX_SEQ_LEN = 200
DATA_LENGTH_MEAN = 90


def masked_mse_loss(pose: torch.Tensor, pose_hat: torch.Tensor, confidence: torch.Tensor, model_num_steps: int = 10):
    # Loss by confidence. If missing joint, no loss. If less likely joint, less gradients.
    sq_error = torch.pow(pose - pose_hat, 2).sum(-1)
    num_steps_norm = np.log(model_num_steps) ** 2 if model_num_steps != 1 else 1  # normalization of the loss by the
    # model's step number
    return (sq_error * confidence).mean() * num_steps_norm


def high_conf_mse_loss(pose: torch.Tensor, pose_hat: torch.Tensor, confidence: torch.Tensor, model_num_steps: int = 10):
    # If missing joint, no loss. otherwise- equal importance.
    sq_error = torch.pow(pose - pose_hat, 2).sum(-1)
    num_steps_norm = np.log(model_num_steps) ** 2  # normalization of the loss by the model's num_steps
    confidence_mask = (confidence > 0.2).int()  # disregard low confidence joints
    return (sq_error * confidence_mask).mean() * num_steps_norm


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
            min_seq_size: int = 20,
            num_steps: int = 10,
            tf_p: float = 0.5,
            lr: int = START_LEARNING_RATE,
            masked_loss: bool = True,
            optimizer_fn: optimizer = torch.optim.Adam,
            separate_positional_embedding: bool = False,
            patience: int = 30,
            lr_th: float = 1e-4,
            num_pose_projection_layers: int = 1,
            use_transformer_decoder: bool = False,
            do_pose_self_attention: bool = False,
            concat: bool = True,
            same_encoding_layer: bool = True,
            no_scheduler: bool = True,
            use_seqlen_pred: bool = False,
            pred_prob_seq_length: bool = False
    ):
        super().__init__()
        self.lr = lr
        self.lr_th = lr_th
        self.tf_p = tf_p
        self.patience = patience
        self.tokenizer = tokenizer
        self.max_seq_size = max_seq_size
        self.min_seq_size = min_seq_size
        self.num_steps = num_steps
        self.hidden_dim = hidden_dim
        self.pose_dims = pose_dims
        self.masked_loss = masked_loss
        self.optimizer_fn = optimizer_fn
        self.separate_positional_embedding = separate_positional_embedding
        self.use_transformer_decoder = use_transformer_decoder
        self.do_pose_self_attention = do_pose_self_attention
        self.best_loss = np.inf
        self.concat = concat
        self.no_scheduler = no_scheduler
        self.use_seqlen_pred = use_seqlen_pred
        self.pred_prob_seq_length = pred_prob_seq_length

        pose_dim = int(np.prod(pose_dims))

        # Embedding layers

        self.embedding = nn.Embedding(
            num_embeddings=len(tokenizer),
            embedding_dim=hidden_dim,
            padding_idx=tokenizer.pad_token_id,
        )

        self.step_embedding = nn.Embedding(
            num_embeddings=num_steps, embedding_dim=hidden_dim
        )

        if separate_positional_embedding:
            self.pos_positional_embeddings = nn.Embedding(
                num_embeddings=max_seq_size, embedding_dim=hidden_dim
            )
            self.text_positional_embeddings = nn.Embedding(
                num_embeddings=max_seq_size, embedding_dim=hidden_dim
            )

        else:
            self.positional_embeddings = nn.Embedding(
                num_embeddings=max_seq_size, embedding_dim=hidden_dim
            )

            # positional embedding scalars
            self.alpha_pose = nn.Parameter(torch.randn(1))
            self.alpha_text = nn.Parameter(torch.randn(1))

        if num_pose_projection_layers == 1:
            self.pose_projection = nn.Linear(pose_dim, hidden_dim)
        else:  # TODO- change based on num layers
            self.pose_projection = nn.Sequential(
                nn.Linear(pose_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        # encoding layers
        if same_encoding_layer:
            encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=encoder_heads,
                                                   dim_feedforward=encoder_dim_feedforward)
            text_encoder_layer = encoder_layer
            pose_encoder_layer = encoder_layer
        else:
            text_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=encoder_heads,
                                                       dim_feedforward=encoder_dim_feedforward)
            pose_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=encoder_heads,
                                                            dim_feedforward=encoder_dim_feedforward)

        self.text_encoder = nn.TransformerEncoder(text_encoder_layer, num_layers=text_encoder_depth)
        self.pose_encoder = nn.TransformerEncoder(pose_encoder_layer, num_layers=pose_encoder_depth)

        if self.do_pose_self_attention:
            self.text_pose_encoder = self.pose_encoder
            pose_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=encoder_heads,
                                                            dim_feedforward=hidden_dim)
            self.pose_encoder = nn.TransformerEncoder(pose_encoder_layer, num_layers=pose_encoder_depth)

        # step encoder
        self.step_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )

        # Predict sequence length
        if self.pred_prob_seq_length:
            self.seq_length = nn.Sequential(nn.Linear(hidden_dim, self.max_seq_size-self.min_seq_size),  # length from 20 to 200
                                        torch.nn.Softmax())
        else:
            self.seq_length = nn.Linear(hidden_dim, 1)

        # Predict pose difference
        pose_diff_projection_output_size = pose_dim

        if self.use_transformer_decoder:
            pose_decoder_out_dim = 512
            decoder_layer = nn.TransformerDecoderLayer(hidden_dim, nhead=4, dim_feedforward=pose_decoder_out_dim)
            self.pose_decoder = nn.TransformerDecoder(decoder_layer, num_layers=pose_encoder_depth)

        self.pose_diff_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, pose_diff_projection_output_size),
        )

    def encode_text(self, texts: List[str]):
        tokenized = self.tokenizer(texts, device=self.device)
        if self.separate_positional_embedding:
            positional_embedding = self.text_positional_embeddings(tokenized["positions"])
        else:
            positional_embedding = self.alpha_text * self.positional_embeddings(tokenized["positions"])
        embedding = self.embedding(tokenized["tokens_ids"]) + positional_embedding
        encoded = self.text_encoder(embedding.transpose(0, 1),
                                    src_key_padding_mask=tokenized["attention_mask"]).transpose(0, 1)

        if self.pred_prob_seq_length:
            seq_length = torch.argmax(self.seq_length(encoded).mean(axis=1), dim=1) + self.min_seq_size
        else:
            seq_length = self.seq_length(encoded).mean(axis=1).int()
            seq_length = torch.minimum(seq_length, torch.full_like(seq_length, self.max_seq_size))
            seq_length = torch.maximum(seq_length, torch.full_like(seq_length, self.min_seq_size))
        return {"data": encoded, "mask": tokenized["attention_mask"]}, seq_length

    def forward(self, text: str, first_pose: torch.Tensor = None, sequence_length: int = -1):
        # if first_pose is None:
        #     first_pose = self.first_pose

        text_encoding, seq_len = self.encode_text([text])
        sequence_length = seq_len if sequence_length == -1 else sequence_length
        # sequence_length = min(sequence_length, self.max_seq_size)
        pose_sequence = {
            "data": first_pose.expand(1, sequence_length, *self.pose_dims),
            "mask": torch.zeros([1, sequence_length], dtype=torch.bool, device=self.device),
        }

        if self.num_steps == 1:
            pred = self.refine_pose_sequence(pose_sequence, text_encoding)
            yield pred
        else:
            step_num = 0
            while True:
                yield pose_sequence["data"][0]
                pose_sequence["data"] = self.refinement_step(step_num, pose_sequence, text_encoding)[0]
                step_num += 1

    def refinement_step(self, step_num, pose_sequence, text_encoding):
        batch_size = pose_sequence["data"].shape[0]
        pose_sequence["data"] = pose_sequence["data"].detach()  # Detach from graph
        batch_step_num = torch.repeat_interleave(torch.LongTensor([step_num]),
                                                 batch_size).unsqueeze(1).to(self.device)
        step_encoding = self.step_encoder(self.step_embedding(batch_step_num))
        change_pred = self.refine_pose_sequence(pose_sequence, text_encoding, step_encoding)
        cur_step_size = self.get_step_size(step_num+1)
        prev_step_size = self.get_step_size(step_num) if step_num > 0 else 0
        step_size = cur_step_size-prev_step_size
        pred = (1-step_size) * pose_sequence["data"] + step_size * change_pred
        return pred, cur_step_size

    def embed_pose(self, pose_sequence_data):
        batch_size, seq_length, _, _ = pose_sequence_data.shape
        flat_pose_data = pose_sequence_data.reshape(batch_size, seq_length, -1)
        positions = torch.arange(0, seq_length, dtype=torch.long, device=self.device)
        if self.separate_positional_embedding:
            positional_embedding = self.pos_positional_embeddings(positions)
        else:
            positional_embedding = self.alpha_pose * self.positional_embeddings(positions)

        # Encode pose sequence
        pose_embedding = self.pose_projection(flat_pose_data) + positional_embedding
        return pose_embedding

    def encode_pose(self, pose_sequence, text_encoding, step_encoding=None):
        batch_size, seq_length, _, _ = pose_sequence["data"].shape

        # Encode pose sequence
        pose_embedding = self.embed_pose(pose_sequence["data"])
        if self.do_pose_self_attention:
            pose_embedding = self.pose_encoder(pose_embedding.transpose(0, 1),
                                               src_key_padding_mask=pose_sequence["mask"]).transpose(0, 1)
        if step_encoding is not None:
            step_mask = torch.zeros([step_encoding.size(0), 1], dtype=torch.bool, device=self.device)

        if self.use_transformer_decoder:
            if step_encoding is not None:
                if self.concat:
                    pose_step = torch.cat([pose_embedding, step_encoding], dim=1)
                    pose_step_mask = torch.cat([pose_sequence["mask"], step_mask], dim=1)
                else:
                    pose_step = pose_embedding + step_encoding
                    pose_step_mask = pose_sequence["mask"]
                pose_step_encoding = self.__get_text_pose_encoder()(pose_step.transpose(0, 1),
                                                            src_key_padding_mask=pose_step_mask)
                # pose_step_encoding += pose_step.transpose(0, 1)  # skip connection
                return pose_step_encoding, pose_step_mask
            else:
                pose_encoding = self.__get_text_pose_encoder()(pose_embedding.transpose(0, 1),
                                                       src_key_padding_mask=pose_sequence["mask"])
                return pose_encoding, pose_sequence["mask"]

        else:
            pose_text_sequence = torch.cat([pose_embedding, text_encoding["data"], step_encoding], dim=1)
            pose_text_mask = torch.cat(
                [pose_sequence["mask"], text_encoding["mask"], step_mask], dim=1
            )

            pose_encoding = self.__get_text_pose_encoder()(
                pose_text_sequence.transpose(0, 1), src_key_padding_mask=pose_text_mask
            ).transpose(0, 1)[:, :seq_length, :]

            # pose_encoding += pose_text_sequence[:, :seq_length, :]  # skip connection
        return pose_encoding

    def __get_text_pose_encoder(self):
        if hasattr(self, "text_pose_encoder"):
            return self.text_pose_encoder
        else:
            return self.pose_encoder

    def refine_pose_sequence(self, pose_sequence, text_encoding, step_encoding=None):
        batch_size, seq_length, _, _ = pose_sequence["data"].shape
        pose_encoding = self.encode_pose(pose_sequence, text_encoding, step_encoding)
        if self.use_transformer_decoder:
            pose_step_encoding, pose_step_mask = pose_encoding
            pose_text_cross_attention = self.pose_decoder(tgt=pose_step_encoding,
                                                          memory=text_encoding["data"],
                                                          memory_key_padding_mask=text_encoding["mask"],
                                                          tgt_key_padding_mask=pose_step_mask)[:seq_length].transpose(0, 1)
            pose_encoding = pose_text_cross_attention + pose_step_encoding.transpose(0, 1)  # skip connection

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

    def step(self, batch, *unused_args, phase: str, gamma: float = 2e-5):
        """
        @param batch: data batch
        @param phase: either "train" or "validation"
        @param gamma: float between 0 and 1, determines the weight of the sequence length loss. default is 0.2.
        @param k: train seq len every k epochs. default is 1: train every epoch. if k==-1: don't train seq_len.
        """
        text_encoding, sequence_length = self.encode_text(batch["text"])
        pose = batch["pose"]

        # Repeat the first frame for initial prediction
        batch_size, pose_seq_length, num_keypoints, _ = pose["data"].shape
        rand = np.random.rand(5)[0]
        teacher_forcing_seq_length = rand < self.tf_p
        seq_length = max(sequence_length).item() if self.use_seqlen_pred and teacher_forcing_seq_length else \
            pose_seq_length
        pose_inverse_mask = pose["inverse_mask"][:, :seq_length] if pose_seq_length > seq_length else \
            torch.cat([pose["inverse_mask"], torch.zeros((batch_size, seq_length-pose_seq_length),
                                                         dtype=torch.bool,
                                                         device=self.device)], dim=1)

        pose_sequence = {
            "data": torch.stack([pose["data"][:, 0]] * seq_length, dim=1),
            "mask": torch.logical_not(pose_inverse_mask)
        }

        if self.use_seqlen_pred and teacher_forcing_seq_length:
            if seq_length > pose_seq_length:
                pose["data"] = torch.cat([pose["data"], torch.stack([pose["data"][:, 0]] *
                                                                    (seq_length - pose_seq_length), dim=1)], dim=1)
                pose["confidence"] = torch.cat([pose["confidence"], torch.zeros((batch_size, seq_length -
                                                pose_seq_length, num_keypoints), device=self.device)], dim=1)
            else:
                pose["data"] = pose["data"][:, :seq_length]
                pose["confidence"] = pose["confidence"][:, :seq_length]

        if self.num_steps == 1:
            pred = self.refine_pose_sequence(pose_sequence, text_encoding)
            l1_gold = pose["data"]
            refinement_loss = masked_mse_loss(l1_gold, pred, pose["confidence"], self.num_steps)
        else:
            refinement_loss = 0
            for i in range(self.num_steps):
                pred, step_size = self.refinement_step(i, pose_sequence, text_encoding)
                l1_gold = step_size * pose["data"] + (1 - step_size) * pose_sequence["data"]

                if self.masked_loss:
                    refinement_loss += masked_mse_loss(l1_gold, pred, pose["confidence"], self.num_steps)
                else:
                    refinement_loss += high_conf_mse_loss(l1_gold, pred, pose["confidence"], self.num_steps)

                teacher_forcing_step_level = np.random.rand(1)[0] < self.tf_p
                # l1_gold = torch.where(pose["data"] == 0, pred, l1_gold)
                pose_sequence["data"] = l1_gold if phase == "validation" or teacher_forcing_step_level else pred
                if self.use_seqlen_pred and teacher_forcing_seq_length:
                    pose_sequence["data"] = pred
                if phase == "train":  # add just a little noise while training
                    pose_sequence["data"] = pose_sequence["data"] + torch.randn_like(pose_sequence["data"]) * EPSILON

        loss = refinement_loss

        all_seq_lens = torch.Tensor([torch.where(pose["inverse_mask"][i] == False)[0][0].item()
                                     if len(torch.where(pose["inverse_mask"][i] == False)[0]) > 0
                                     else pose["inverse_mask"].size(1) for i in range(batch_size)]).unsqueeze(1).to(self.device)
        sequence_length_loss = F.mse_loss(sequence_length.unsqueeze(1), all_seq_lens)
        loss += gamma*sequence_length_loss

        self.log(phase + "_seq_length_loss", sequence_length_loss, batch_size=batch_size)
        self.log(phase + "_refinement_loss", refinement_loss, batch_size=batch_size)
        self.log(phase + "_loss", loss, batch_size=batch_size)

        return loss

    # def on_train_epoch_start(self):
    #     # self.trainer.accelerator.setup_optimizers(self)
    #     if self.current_epoch == 20:  # switch to SGD
    #         print("switch to SGD")
    #         optimizer = SGD(self.parameters(), lr=self.lr)
    #         # scheduler = [ReduceLROnPlateau(optimizer, patience=self.patience, verbose=True)]
    #         self.trainer.optimizers = [optimizer]
    #         # self.trainer.lr_schedulers = [{
    #         #     "scheduler": scheduler,
    #         #     "monitor": "train_loss",
    #         #     "interval": "epoch",
    #         #     "frequency": 1,
    #         # }]

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        if self.no_scheduler:
            return optimizer

        scheduler = ReduceLROnPlateau(optimizer, patience=self.patience, verbose=True)
        return [optimizer], [{
                "scheduler": scheduler,
                "monitor": "train_loss",
                "interval": "epoch",
                "frequency": 1,
                "threshold": self.lr_th
            }]

        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler2,
        #         "monitor": "train_loss",
        #         "interval": "epoch",
        #         "frequency": 1,
        #         "threshold": self.lr_th
        #     }
        # }

#
# class MyScheduler(torch.optim.lr_scheduler._LRScheduler):
#     def __init__(self, optimizer, factor=0.1, patience=100, last_epoch=-1, verbose=True):
#         """
#         Args:
#             optimizer (Optimizer): Wrapped optimizer.
#             factor (float): Factor by which the learning rate will be reduced and increased. Default: 0.1.
#             patience (int): Number of epochs with no improvement after which learning rate will be
#             increased/decreased. Default: 50.
#         """
#         self.optimizer = optimizer
#         self.factor = factor
#         self.saved_in_cur_lr = False
#         self.patience = patience
#         self.num_steps = 0
#         super(MyScheduler, self).__init__(optimizer, last_epoch, verbose)
#
#     def get_lr(self):
#         if self.saved_in_cur_lr and self.num_steps >= self.patience:
#             self.saved_in_cur_lr = False
#             self.num_steps = 0
#             return [group['lr'] * self.factor for group in self.optimizer.param_groups]
#         elif self.num_steps >= self.patience:
#             self.num_steps = 0
#             self.saved_in_cur_lr = False
#             return [group['lr'] / self.factor for group in self.optimizer.param_groups]
#         else:
#             self.num_steps += 1
#             return [group['lr'] for group in self.optimizer.param_groups]
#
