import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
from torchmetrics.regression.mse import MeanSquaredError
from torchmetrics.regression.mae import MeanAbsoluteError
from typing import Any, List
from .decoder import Decoder
from transformers import EsmModel, AutoTokenizer, EsmTokenizer
import torch.nn.functional as F
from de.samplers.models.amix_utils import load_amix_model


class ESM2_Attention(nn.Module):
    def __init__(self,
                 pretrained_model_name_or_path: str = "facebook/esm2_t12_35M_UR50D",
                 hidden_dim: int = 512):
        super().__init__()
        self.esm = EsmModel.from_pretrained(pretrained_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        input_dim = self.esm.config.hidden_size
        self.decoder = Decoder(input_dim, hidden_dim)

    def freeze_encoder(self):
        for param in self.esm.parameters():
            param.requires_grad = False

    def forward(self, x):
        enc_out = self.esm(x).last_hidden_state
        output = self.decoder(enc_out)
        return output


class ESM2DecoderModule(LightningModule):
    def __init__(self,
                 net: nn.Module,
                 optimizer: torch.optim.Optimizer):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(ignore=["net"])
        self.net = net
        # loss function
        self.criterion = torch.nn.MSELoss()

        # metric objects for calculating and averaging error
        self.train_mae = MeanAbsoluteError()
        self.valid_mae = MeanAbsoluteError()
        self.valid_mse = MeanSquaredError()

        # averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        # for tracking best so far
        self.val_mae_best = MinMetric()
        self.val_mse_best = MinMetric()

    def forward(self, x):
        return self.net(x)

    def on_train_start(self):
        self.val_loss.reset()
        self.valid_mae.reset()
        self.valid_mse.reset()
        self.val_mse_best.reset()
        self.val_mae_best.reset()

    def model_step(self, batch):
        x, y = batch["input_ids"], batch["fitness"]
        y = y.unsqueeze(1)
        pred = self.forward(x)
        loss = self.criterion(pred, y)
        return loss, pred, y

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_mae(preds, targets)
        self.log("train_loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_mae", self.train_mae, on_step=True, on_epoch=True, prog_bar=True)

        # return loss
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.valid_mae(preds, targets)
        self.valid_mse(preds, targets)
        self.log("val_loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mae", self.valid_mae, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        mae = self.valid_mae.compute()  # get current mae
        mse = self.valid_mse.compute()  # get current mse
        self.val_mae_best(mae)
        self.val_mse_best(mse)
        self.log("val_mae_best", self.val_mae_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val_mse_best", self.val_mse_best.compute(), sync_dist=True, prog_bar=True)

    def configure_optimizers(self) -> Any:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        return {"optimizer": optimizer}

    def predict_fitness(self, representation: torch.Tensor):
        fitness = self.net.decoder(representation)
        return fitness

    def infer_fitness(self, seqs: List[str]):
        with torch.inference_mode():
            inputs = self.net.tokenizer(seqs, return_tensors="pt").to(self.device)
            repr = self.net.esm(**inputs).last_hidden_state
            outputs = self.predict_fitness(repr)
            return outputs.cpu()


class AMix_Attention(nn.Module):
    def __init__(self,
                 ckpt_path: str,
                 hidden_dim: int = 512):
        super().__init__()
        
        # Load AMix model using utility function
        self.amix_model = load_amix_model(ckpt_path, device='cpu')
        self.tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t30_150M_UR50D')
        
        # Get hidden size from the AMix model's ESM encoder
        input_dim = self.amix_model.bfn.net.esm.config.hidden_size
        self.decoder = Decoder(input_dim, hidden_dim)

    def freeze_encoder(self):
        for param in self.amix_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Convert input_ids to one-hot embeddings for BFN
        inputs_embeds = F.one_hot(x, num_classes=len(self.tokenizer)).float()
        attention_mask = (x != self.tokenizer.pad_token_id)
        
        # Set timestep to 1.0 for inference
        t = torch.ones_like(attention_mask).float()
        
        # Get encoder output
        with torch.no_grad():
            outputs = self.amix_model.bfn.net(
                t=t,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=False
            )
        
        enc_out = outputs["last_hidden_state"]
        output = self.decoder(enc_out)
        return output


class AMixDecoderModule(LightningModule):
    def __init__(self,
                 net: nn.Module,
                 optimizer: torch.optim.Optimizer):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(ignore=["net"])
        self.net = net
        # loss function
        self.criterion = torch.nn.MSELoss()

        # metric objects for calculating and averaging error
        self.train_mae = MeanAbsoluteError()
        self.valid_mae = MeanAbsoluteError()
        self.valid_mse = MeanSquaredError()

        # averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        # for tracking best so far
        self.val_mae_best = MinMetric()
        self.val_mse_best = MinMetric()

    def forward(self, x):
        return self.net(x)

    def on_train_start(self):
        self.val_loss.reset()
        self.valid_mae.reset()
        self.valid_mse.reset()
        self.val_mse_best.reset()
        self.val_mae_best.reset()

    def model_step(self, batch):
        x, y = batch["input_ids"], batch["fitness"]
        y = y.unsqueeze(1)
        pred = self.forward(x)
        loss = self.criterion(pred, y)
        return loss, pred, y

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_mae(preds, targets)
        self.log("train_loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_mae", self.train_mae, on_step=True, on_epoch=True, prog_bar=True)

        # return loss
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.valid_mae(preds, targets)
        self.valid_mse(preds, targets)
        self.log("val_loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mae", self.valid_mae, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        mae = self.valid_mae.compute()  # get current mae
        mse = self.valid_mse.compute()  # get current mse
        self.val_mae_best(mae)
        self.val_mse_best(mse)
        self.log("val_mae_best", self.val_mae_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val_mse_best", self.val_mse_best.compute(), sync_dist=True, prog_bar=True)

    def configure_optimizers(self) -> Any:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        return {"optimizer": optimizer}

    def predict_fitness(self, representation: torch.Tensor):
        fitness = self.net.decoder(representation)
        return fitness

    def infer_fitness(self, seqs: List[str]):
        with torch.inference_mode():
            inputs = self.net.tokenizer(seqs, return_tensors="pt", padding=True).to(self.device)
            input_ids = inputs["input_ids"]
            
            # Convert to one-hot embeddings
            inputs_embeds = F.one_hot(input_ids, num_classes=len(self.net.tokenizer)).float()
            attention_mask = (input_ids != self.net.tokenizer.pad_token_id)
            
            # Set timestep to 1.0 for inference
            t = torch.ones_like(attention_mask).float()
            
            # Get encoder output
            outputs = self.net.amix_model.bfn.net(
                t=t,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=False
            )
            
            repr = outputs["last_hidden_state"]
            outputs = self.predict_fitness(repr)
            return outputs.cpu()

