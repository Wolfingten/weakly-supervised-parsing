import os
from math import ceil
import torch
import datasets
import numpy as np
import pandas as pd

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer, logging

from onnxruntime import InferenceSession
from scipy.special import softmax

from weakly_supervised_parser.model.data_module_loader import DataModule
from weakly_supervised_parser.model.span_classifier import LightningModel


# Disable model checkpoint warnings
logging.set_verbosity_error()


# class MetricsCallback(Callback):
#    """PyTorch Lightning metric callback."""
#
#    def __init__(self):
#        super().__init__()
#        self.metrics = []
#
#    def on_validation_end(self, trainer, pl_module):
#        self.metrics.append(trainer.callback_metrics)


class InsideOutsideStringClassifier:
    def __init__(
        self, model_name_or_path: str, num_labels: int = 2, max_seq_length: int = 256
    ):

        self.model_name_or_path = model_name_or_path
        self.num_labels = num_labels
        self.max_seq_length = max_seq_length

    def fit(
        self,
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        outputdir: str,
        filename: str,
        devices: int = 1,
        enable_progress_bar: bool = True,
        enable_model_summary: bool = True,
        enable_checkpointing: bool = True,
        logger: bool = False,
        accelerator: str = "auto",
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        learning_rate: float = 5e-6,
        max_epochs: int = 10,
        dataloader_num_workers: int = 16,
        seed: int = 42,
        run_name: str = "InsideOutsideStringClassifier",
    ):

        wandb_logger = WandbLogger(
            project="wsparser InsideOutsideStringClassifier",
            name=run_name,
            log_model="all",
        )

        data_module = DataModule(
            model_name_or_path=self.model_name_or_path,
            train_df=train_df,
            eval_df=eval_df,
            max_seq_length=self.max_seq_length,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=dataloader_num_workers,
        )

        model = LightningModel(
            model_name_or_path=self.model_name_or_path,
            lr=learning_rate,
            num_labels=self.num_labels,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
        )

        wandb_logger.watch(model, log="all")

        class TrainingLossLoggerCallback(Callback):
            def on_train_epoch_end(self, trainer, pl_module, outputs):
                train_loss = outputs.get("loss")
                if train_loss is not None:
                    trainer.logger.log_metrics(
                        {"train_loss": train_loss.item()}, step=trainer.global_step
                    )

        seed_everything(seed, workers=True)

        callbacks = []
        callbacks.append(
            EarlyStopping(monitor="val_loss", patience=2, mode="min", check_finite=True)
        )
        #        callbacks.append(MetricsCallback())
        callbacks.append(TrainingLossLoggerCallback())
        # callbacks.append(ModelCheckpoint(monitor="val_loss", dirpath=outputdir, filename=filename, save_top_k=1, save_weights_only=True, mode="min"))

        trainer = Trainer(
            accelerator=accelerator,
            devices=devices,
            max_epochs=max_epochs,
            callbacks=callbacks,
            enable_progress_bar=enable_progress_bar,
            enable_model_summary=enable_model_summary,
            enable_checkpointing=enable_checkpointing,
            track_grad_norm=2,
            logger=wandb_logger,
            log_every_n_steps=10,
        )
        trainer.fit(model, data_module)
        trainer.validate(model, data_module.val_dataloader())

        train_batch = next(iter(data_module.train_dataloader()))

        model.to_onnx(
            file_path=f"{outputdir}/{filename}.onnx",
            input_sample=(
                train_batch["input_ids"].cuda(),
                train_batch["attention_mask"].cuda(),
            ),
            export_params=True,
            opset_version=11,
            input_names=["input", "attention_mask"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "attention_mask": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )

    def load_model(self, pre_trained_model_path):
        self.model = InferenceSession(
            pre_trained_model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, use_fast=True
        )

    def preprocess_function(self, data):
        features = self.tokenizer(
            data["sentence"],
            max_length=self.max_seq_length,
            padding="max_length",
            add_special_tokens=True,
            truncation=True,
            return_tensors="np",
        )
        return features

    def process_spans(self, spans, scale_axis):
        spans_dataset = datasets.Dataset.from_pandas(spans)
        processed = spans_dataset.map(
            self.preprocess_function, batched=True, batch_size=None
        )
        inputs = {
            "input": processed["input_ids"],
            "attention_mask": processed["attention_mask"],
        }
        with torch.no_grad():
            return softmax(self.model.run(None, inputs)[0], axis=scale_axis)

    #    def predict_proba(self, spans, scale_axis, predict_batch_size):
    #        if spans.shape[0] > predict_batch_size:
    #            output = []
    #            span_batches = np.array_split(spans, ceil(spans.shape[0] / predict_batch_size))
    #            for span_batch in span_batches:
    #                output.extend(self.process_spans(span_batch, scale_axis))
    #            return np.vstack(output)
    #        else:
    #            return self.process_spans(spans, scale_axis)

    def predict_proba(self, spans, scale_axis, predict_batch_size):
        # n_spans = 150
        n_spans = spans.shape[0]
        if n_spans > predict_batch_size:
            # add padding so all inputs can be passed to the model
            n_padding = -(-n_spans // predict_batch_size) * predict_batch_size - n_spans
            spans = pd.concat(
                [spans, pd.DataFrame(["<pad>"] * n_padding, columns=spans.columns)],
                ignore_index=True,
            )
            output = []
            # spans.shape[0] = 160 because padding
            span_batches = np.array_split(spans, spans.shape[0] // predict_batch_size)
            for span_batch in span_batches:
                output.extend(self.process_spans(span_batch, scale_axis))
            return np.vstack(output)[:n_spans:]  # truncate padding
        else:
            n_padding = predict_batch_size - n_spans
            spans = pd.concat(
                [spans, pd.DataFrame(["na"] * n_padding, columns=spans.columns)],
                ignore_index=True,
            )
            return self.process_spans(spans, scale_axis)[:n_spans:]  # truncate padding

    def predict(self, spans, scale_axis, predict_batch_size):
        return self.predict_proba(spans, scale_axis, predict_batch_size).argmax(axis=1)
