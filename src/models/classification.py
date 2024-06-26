import torch
from pytorch_lightning import LightningModule
from sklearn.metrics import f1_score, accuracy_score
from torch import nn
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import AdamW, Adam
from transformers import get_linear_schedule_with_warmup, AutoModel, AutoTokenizer, AutoModelForSequenceClassification, \
    AutoConfig, T5EncoderModel
from utils.prompting import CPUAutomaticVerbalizer

from utils.seed_util import seed_all


class FrozenClassification(LightningModule):

    def __init__(self, input_dim, num_classes, dropout_rate, hyperparameter):
        super().__init__()

        seed_all(hyperparameter["seed"])

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(input_dim, num_classes)

        if num_classes == 1:
            self.loss = MSELoss()
            self.problem_type = "regression"
        elif num_classes > 1:
            self.loss = CrossEntropyLoss()
            self.problem_type = "single_label_classification"
        else:
            self.loss = BCEWithLogitsLoss()
            self.problem_type = "multi_label_classification"

        self.hyperparameter = hyperparameter

    def forward(self, x):
        x = self.dropout(x)
        y_hat = self.classifier(x)

        return y_hat


    def training_epoch_end(self, losses):
        all_lengths = sum([ele["length"] for ele in losses])
        summed_loss = sum([ele["length"] * ele["loss"] for ele in losses])

        self.log("train/loss",summed_loss/all_lengths)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        y_hat = self(batch[0]).softmax(axis=1).argmax(axis=1)
        return y_hat

    def training_step(self, batch, batch_index):
        x, y = batch[0]

        y_hat = self.forward(x)
        y_hat = y_hat.softmax(axis=1)

        loss = self.loss(y_hat, y)

        return {
            "loss": loss,
            "length": x.shape[0]
        }

    def validation_step(self, batch, batch_index):
        x, y = batch
        y_hat = self.forward(x)
        y_hat = y_hat.softmax(axis=1)

        return {
            "y_hat": y_hat.cpu(),
            "y": y.cpu()
        }

    def validation_epoch_end(self, outputs):
        y = torch.concat([x["y"] for x in outputs])
        y_hat = torch.concat([x["y_hat"] for x in outputs])
        val_loss = self.loss(y_hat, y)

        y_hat = y_hat.argmax(axis=1)

        f1 = f1_score(y, y_hat, average="macro")
        accuracy = accuracy_score(y, y_hat)

        self.log_dict({
            "eval/loss": val_loss,
            "eval/f1-macro": f1,
            "eval/accuracy": accuracy,
        }, on_step=False, prog_bar=True)


    def test_step(self, batch, batch_index):
        x, y = batch
        y_hat = self.forward(x)
        y_hat = y_hat.softmax(axis=1)

        return {
            "y_hat": y_hat.cpu(),
            "y": y.cpu()
        }

    def test_epoch_end(self, outputs):

        y = torch.concat([x["y"] for x in outputs])
        y_hat = torch.concat([x["y_hat"] for x in outputs])
        test_loss = self.loss(y_hat, y)

        y_hat = y_hat.argmax(axis=1)

        f1 = f1_score(y, y_hat, average="macro")
        accuracy = accuracy_score(y, y_hat)

        self.log_dict({
            "test/loss": test_loss,
            "test/f1-macro": f1,
            "test/accuracy": accuracy,
        })


    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hyperparameter["learning_rate"])

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hyperparameter["warmup_steps"],
            num_training_steps=self.hyperparameter["training_steps"]
        )

        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval='step'
            )
        )


class TemplateTuningClassification(LightningModule):

    def __init__(self, prompt_model, hyperparameter):
        super().__init__()

        seed_all(hyperparameter["seed"])

        self.loss = CrossEntropyLoss()
        self.prompt_model = prompt_model
        self.hyperparameter = hyperparameter

    def forward(self, features):
        logits = self.prompt_model(features)

        return logits


    def training_epoch_end(self, losses):
        all_lengths = sum([ele["length"] for ele in losses])
        summed_loss = sum([ele["length"] * ele["loss"] for ele in losses])

        self.log("train/loss",summed_loss/all_lengths)

    def predict_step(self, features, batch_idx, dataloader_idx=0):

        logits = self.forward(features)
        y_hat = logits.softmax(axis=1).argmax(axis=1)
        return y_hat

    def training_step(self, batch, batch_index):

        features = batch[0]
        y = features["label"]

        logits = self.forward(features)
        y_hat = logits.softmax(axis=1)

        loss = self.loss(y_hat, y)

        return {
            "loss": loss,
            "length": y.shape[0]
        }

    def training_epoch_end(self, outputs):
        if type(self.prompt_model.verbalizer) == CPUAutomaticVerbalizer:
            self.prompt_model.verbalizer.optimize_to_initialize()


    def validation_step(self, features, batch_index):
        y = features["label"]

        logits = self.forward(features)
        y_hat = logits.softmax(axis=1)

        return {
            "y_hat": y_hat.cpu(),
            "y": y.cpu()
        }

    def validation_epoch_end(self, outputs):
        y = torch.concat([x["y"] for x in outputs])
        y_hat = torch.concat([x["y_hat"] for x in outputs])
        val_loss = self.loss(y_hat, y)

        y_hat = y_hat.argmax(axis=1)

        f1 = f1_score(y, y_hat, average="macro")
        accuracy = accuracy_score(y, y_hat)

        self.log_dict({
            "eval/loss": val_loss,
            "eval/f1-macro": f1,
            "eval/accuracy": accuracy,
        }, on_step=False, prog_bar=True)


    def test_step(self, features, batch_index):
        y = features["label"]

        logits = self.forward(features)
        y_hat = logits.softmax(axis=1)

        return {
            "y_hat": y_hat.cpu(),
            "y": y.cpu()
        }

    def test_epoch_end(self, outputs):

        y = torch.concat([x["y"] for x in outputs])
        y_hat = torch.concat([x["y_hat"] for x in outputs])
        test_loss = self.loss(y_hat, y)

        y_hat = y_hat.argmax(axis=1)

        f1 = f1_score(y, y_hat, average="macro")
        accuracy = accuracy_score(y, y_hat)

        self.log_dict({
            "test/loss": test_loss,
            "test/f1-macro": f1,
            "test/accuracy": accuracy,
        })


    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']

        optimizers = []
        schedulers = []

        if self.hyperparameter["optimize_plm"]:

            optimizer_grouped_parameters_plm = [
                {'params': [p for n, p in self.prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in self.prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            optimizer_plm = AdamW(optimizer_grouped_parameters_plm, lr=self.hyperparameter["learning_rate"])

            scheduler_plm = get_linear_schedule_with_warmup(
                optimizer_plm,
                num_warmup_steps=self.hyperparameter["warmup_steps"],
                num_training_steps=self.hyperparameter["training_steps"]
            )

            optimizers.append(optimizer_plm)
            schedulers.append(scheduler_plm)


        optimizer_grouped_parameters_template = [
            {'params': [p for n, p in self.prompt_model.template.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.prompt_model.template.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer_template = Adam(optimizer_grouped_parameters_template, lr=self.hyperparameter["template_learning_rate"])

        scheduler_template = get_linear_schedule_with_warmup(
            optimizer_template,
            num_warmup_steps=self.hyperparameter["warmup_steps"],
            num_training_steps=self.hyperparameter["training_steps"]
        )

        optimizers.append(optimizer_template)
        schedulers.append(scheduler_template)

        #return optimizers, schedulers

        return dict(
            optimizer=optimizer_template,
            lr_scheduler=dict(
                scheduler=scheduler_template,
                interval='step'
            )
        )



class SoftVerbalizingTuningClassification(LightningModule):

    def __init__(self, prompt_model, hyperparameter):
        super().__init__()

        seed_all(hyperparameter["seed"])

        self.loss = CrossEntropyLoss()
        self.prompt_model = prompt_model
        self.hyperparameter = hyperparameter

    def forward(self, features):
        logits = self.prompt_model(features)

        return logits


    def training_epoch_end(self, losses):
        all_lengths = sum([ele["length"] for ele in losses])
        summed_loss = sum([ele["length"] * ele["loss"] for ele in losses])

        self.log("train/loss",summed_loss/all_lengths)

    def predict_step(self, features, batch_idx, dataloader_idx=0):

        logits = self.forward(features)
        y_hat = logits.softmax(axis=1).argmax(axis=1)
        return y_hat

    def training_step(self, batch, batch_index):

        features = batch[0]
        y = features["label"]

        logits = self.forward(features)
        y_hat = logits.softmax(axis=1)

        loss = self.loss(y_hat, y)

        return {
            "loss": loss,
            "length": y.shape[0]
        }

#    def training_epoch_end(self, outputs):
#        self.prompt_model.verbalizer.optimize_to_initialize()


    def validation_step(self, features, batch_index):
        y = features["label"]

        logits = self.forward(features)
        y_hat = logits.softmax(axis=1)

        return {
            "y_hat": y_hat.cpu(),
            "y": y.cpu()
        }

    def validation_epoch_end(self, outputs):
        y = torch.concat([x["y"] for x in outputs])
        y_hat = torch.concat([x["y_hat"] for x in outputs])
        val_loss = self.loss(y_hat, y)

        y_hat = y_hat.argmax(axis=1)

        f1 = f1_score(y, y_hat, average="macro")
        accuracy = accuracy_score(y, y_hat)

        self.log_dict({
            "eval/loss": val_loss,
            "eval/f1-macro": f1,
            "eval/accuracy": accuracy,
        }, on_step=False, prog_bar=True)


    def test_step(self, features, batch_index):
        y = features["label"]

        logits = self.forward(features)
        y_hat = logits.softmax(axis=1)

        return {
            "y_hat": y_hat.cpu(),
            "y": y.cpu()
        }

    def test_epoch_end(self, outputs):

        y = torch.concat([x["y"] for x in outputs])
        y_hat = torch.concat([x["y_hat"] for x in outputs])
        test_loss = self.loss(y_hat, y)

        y_hat = y_hat.argmax(axis=1)

        f1 = f1_score(y, y_hat, average="macro")
        accuracy = accuracy_score(y, y_hat)

        self.log_dict({
            "test/loss": test_loss,
            "test/f1-macro": f1,
            "test/accuracy": accuracy,
        })


    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']

        optimizers = []
        schedulers = []

        if self.hyperparameter["optimize_plm"]:

            optimizer_grouped_parameters_plm = [
                {'params': [p for n, p in self.prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in self.prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            optimizer_plm = AdamW(optimizer_grouped_parameters_plm, lr=self.hyperparameter["learning_rate"])

            scheduler_plm = get_linear_schedule_with_warmup(
                optimizer_plm,
                num_warmup_steps=self.hyperparameter["warmup_steps"],
                num_training_steps=self.hyperparameter["training_steps"]
            )

            optimizers.append(optimizer_plm)
            schedulers.append(scheduler_plm)


        optimizer_grouped_parameters_verbalizer = [
            {'params': [p for n, p in self.prompt_model.verbalizer.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.prompt_model.verbalizer.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer_verbalizer = Adam(optimizer_grouped_parameters_verbalizer, lr=self.hyperparameter["verbalizer_learning_rate"])

        scheduler_verbalizer = get_linear_schedule_with_warmup(
            optimizer_verbalizer,
            num_warmup_steps=self.hyperparameter["warmup_steps"],
            num_training_steps=self.hyperparameter["training_steps"],
        )

        optimizers.append(optimizer_verbalizer)
        schedulers.append(scheduler_verbalizer)

        #return optimizers, schedulers

        return dict(
            optimizer=optimizer_verbalizer,
            lr_scheduler=dict(
                scheduler=scheduler_verbalizer,
                interval='step'
            )
        )


class PromptClassification(LightningModule):

    def __init__(self, hyperparameter, prompt_model):
        super().__init__()

        seed_all(hyperparameter["seed"])

        self.loss = CrossEntropyLoss()
        self.prompt_model = prompt_model
        self.hyperparameter = hyperparameter

    def forward(self, features):
        logits = self.prompt_model(features)

        return logits


    def training_epoch_end(self, losses):
        all_lengths = sum([ele["length"] for ele in losses])
        summed_loss = sum([ele["length"] * ele["loss"] for ele in losses])

        self.log("train/loss", summed_loss/all_lengths)

        if type(self.prompt_model.verbalizer) == CPUAutomaticVerbalizer:
            self.prompt_model.verbalizer.optimize_to_initialize()

    def predict_step(self, features, batch_idx, dataloader_idx=0):

        logits = self.forward(features)
        y_hat = logits.softmax(axis=1).argmax(axis=1)
        return y_hat

    def training_step(self, batch, batch_index):

        features = batch[0]
        y = features["label"]

        logits = self.forward(features)
        y_hat = logits.softmax(axis=1)

        loss = self.loss(y_hat, y)

        return {
            "loss": loss,
            "length": y.shape[0]
        }


    def validation_step(self, features, batch_index):
        y = features["label"]

        logits = self.forward(features)
        y_hat = logits.softmax(axis=1)

        return {
            "y_hat": y_hat.cpu(),
            "y": y.cpu()
        }

    def validation_epoch_end(self, outputs):
        y = torch.concat([x["y"] for x in outputs])
        y_hat = torch.concat([x["y_hat"] for x in outputs])
        val_loss = self.loss(y_hat, y)

        y_hat = y_hat.argmax(axis=1)

        f1 = f1_score(y, y_hat, average="macro")
        accuracy = accuracy_score(y, y_hat)

        self.log_dict({
            "eval/loss": val_loss,
            "eval/f1-macro": f1,
            "eval/accuracy": accuracy,
        }, on_step=False, prog_bar=True)


    def test_step(self, features, batch_index):
        y = features["label"]

        logits = self.forward(features)
        y_hat = logits.softmax(axis=1)

        return {
            "y_hat": y_hat.cpu(),
            "y": y.cpu()
        }

    def test_epoch_end(self, outputs):

        y = torch.concat([x["y"] for x in outputs])
        y_hat = torch.concat([x["y_hat"] for x in outputs])
        test_loss = self.loss(y_hat, y)

        y_hat = y_hat.argmax(axis=1)

        f1 = f1_score(y, y_hat, average="macro")
        accuracy = accuracy_score(y, y_hat)

        self.log_dict({
            "test/loss": test_loss,
            "test/f1-macro": f1,
            "test/accuracy": accuracy,
        })


    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hyperparameter["learning_rate"])

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hyperparameter["warmup_steps"],
            num_training_steps=self.hyperparameter["training_steps"]
        )

        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval='step'
            )
        )



class SanityPromptClassification(LightningModule):

    def __init__(self, hyperparameter, prompt_model):
        super().__init__()

        seed_all(hyperparameter["seed"])

        self.loss = CrossEntropyLoss()
        self.prompt_model = prompt_model
        self.hyperparameter = hyperparameter

    def forward(self, features):
        logits = self.prompt_model(features)

        return logits


    def training_epoch_end(self, losses):
        all_lengths = sum([ele["length"] for ele in losses])
        summed_loss = sum([ele["length"] * ele["loss"] for ele in losses])

        self.log("train/loss", summed_loss/all_lengths)


    def predict_step(self, features, batch_idx, dataloader_idx=0):

        logits = self.forward(features)
        y_hat = logits.softmax(axis=1).argmax(axis=1)
        return y_hat

    def training_step(self, batch, batch_index):

        features = batch[0]
        y = features["label"]

        logits = self.forward(features)
        y_hat = logits.softmax(axis=1)

        loss = self.loss(y_hat, y)

        return {
            "loss": loss,
            "length": y.shape[0]
        }

#    def on_validation_epoch_start(self):
#        self.prompt_model.verbalizer.optimize_to_initialize()

    def validation_step(self, features, batch_index):
        y = features["label"]

        logits = self.forward(features)
        y_hat = logits.softmax(axis=1)

        return {
            "y_hat": y_hat.cpu(),
            "y": y.cpu()
        }

    def validation_epoch_end(self, outputs):
        y = torch.concat([x["y"] for x in outputs])
        y_hat = torch.concat([x["y_hat"] for x in outputs])
        val_loss = self.loss(y_hat, y)

        y_hat = y_hat.argmax(axis=1)

        f1 = f1_score(y, y_hat, average="macro")
        accuracy = accuracy_score(y, y_hat)

        self.log_dict({
            "eval/loss": val_loss,
            "eval/f1-macro": f1,
            "eval/accuracy": accuracy,
        }, on_step=False, prog_bar=True)


    def test_step(self, features, batch_index):
        y = features["label"]

        logits = self.forward(features)
        y_hat = logits.softmax(axis=1)

        return {
            "y_hat": y_hat.cpu(),
            "y": y.cpu()
        }

    def test_epoch_end(self, outputs):

        y = torch.concat([x["y"] for x in outputs])
        y_hat = torch.concat([x["y_hat"] for x in outputs])
        test_loss = self.loss(y_hat, y)

        y_hat = y_hat.argmax(axis=1)

        f1 = f1_score(y, y_hat, average="macro")
        accuracy = accuracy_score(y, y_hat)

        self.log_dict({
            "test/loss": test_loss,
            "test/f1-macro": f1,
            "test/accuracy": accuracy,
        })


    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hyperparameter["learning_rate"])

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hyperparameter["warmup_steps"],
            num_training_steps=self.hyperparameter["training_steps"]
        )

        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval='step'
            )
        )



class FineTuningClassification(LightningModule):

    def __init__(self, include_label_for_forward = False):
        super().__init__()

        self.include_label_for_forward = include_label_for_forward



    def training_epoch_end(self, losses):
        all_lengths = sum([ele["length"] for ele in losses])
        summed_loss = sum([ele["length"] * ele["loss"] for ele in losses])

        self.log("train/loss",summed_loss/all_lengths)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        y_hat = self(batch).softmax(axis=1).argmax(axis=1)
        return y_hat

    def training_step(self, batch, batch_index):
        if type(batch) == list:
            batch = batch[0]

        y = batch["labels"]

        X = batch

        if not self.include_label_for_forward:
            del X["labels"]

        y_hat = self.forward(X)

        y_hat = y_hat.softmax(axis=1)

        try:
            loss = self.loss(y_hat, y)
        except:
            print()
        return {
            "loss": loss,
            "length": len(y)
        }

    def validation_step(self, batch, batch_index):
        if type(batch) == list:
            batch = batch[0]

        y = batch["labels"]

        X = batch

        if not self.include_label_for_forward:
            del X["labels"]

        y_hat = self.forward(X)
        y_hat = y_hat.softmax(axis=1)

        return {
            "y_hat": y_hat.cpu(),
            "y": y.cpu()
        }

    def validation_epoch_end(self, outputs):
        y = torch.concat([x["y"] for x in outputs])
        y_hat = torch.concat([x["y_hat"] for x in outputs])
        val_loss = self.loss(y_hat, y)

        y_hat = y_hat.argmax(axis=1)

        f1 = f1_score(y, y_hat, average="macro")
        accuracy = accuracy_score(y, y_hat)

        self.log_dict({
            "eval/loss": val_loss,
            "eval/f1-macro": f1,
            "eval/accuracy": accuracy,
        }, on_step=False, prog_bar=True)



    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if type(batch) == list:
            batch = batch[0]

        X = batch

        if not self.include_label_for_forward:
            del X["labels"]

        y_hat = self.forward(X)
        y_hat = y_hat.softmax(axis=1).argmax(axis=1)

        return y_hat

    def test_step(self, batch, batch_index):
        if type(batch) == list:
            batch = batch[0]

        y = batch["labels"]

        X = batch

        if not self.include_label_for_forward:
            del X["labels"]

        y_hat = self.forward(X)
        y_hat = y_hat.softmax(axis=1)

        return {
            "y_hat": y_hat.cpu(),
            "y": y.cpu()
        }

    def test_epoch_end(self, outputs):

        y = torch.concat([x["y"] for x in outputs])
        y_hat = torch.concat([x["y_hat"] for x in outputs])
        test_loss = self.loss(y_hat, y)

        y_hat = y_hat.argmax(axis=1)

        f1 = f1_score(y, y_hat, average="macro")
        accuracy = accuracy_score(y, y_hat)

        self.log_dict({
            "test/loss": test_loss,
            "test/f1-macro": f1,
            "test/accuracy": accuracy,
        })


    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hyperparameter["learning_rate"], weight_decay=0.01)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hyperparameter["warmup_steps"],
            num_training_steps=self.hyperparameter["training_steps"]
        )

        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval='step'
            )
        )




class MeanFineTuningClassification(FineTuningClassification):

    def __init__(self, num_classes, hyperparameter, include_label_for_forward=False):
        super().__init__(include_label_for_forward=include_label_for_forward)

        seed_all(hyperparameter["seed"])

        self.hyperparameter = hyperparameter
        self.tokenizer = AutoTokenizer.from_pretrained(self.hyperparameter["model_name"])

        if "t5" in str(self.tokenizer).lower():
            T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
            self.base_model = T5EncoderModel.from_pretrained(self.hyperparameter["model_name"])
        else:
            self.base_model = AutoModel.from_pretrained(self.hyperparameter["model_name"])


        self.dropout = nn.Dropout(self.hyperparameter["dropout_rate"])
        self.classifier = nn.Linear(self.base_model.config.hidden_size, num_classes)

        if num_classes == 1:
            self.loss = MSELoss()
            self.problem_type = "regression"
        elif num_classes > 1:
            self.loss = CrossEntropyLoss()
            self.problem_type = "single_label_classification"


    def mean_pooling(self, token_embeddings, attention_mask):
        # following https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Pooling.py

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        pooled = sum_embeddings / sum_mask

        return pooled

    def forward(self, features):

        encodings = self.base_model(**features)

        if "last_hidden_state" in encodings:
            encodings = encodings["last_hidden_state"]

        encodings = self.mean_pooling(token_embeddings=encodings, attention_mask=features["attention_mask"])

        encodings = self.dropout(encodings)
        y_hat = self.classifier(encodings)

        return y_hat




class CLSFineTuningClassification(FineTuningClassification):

    def __init__(self, num_classes, hyperparameter, include_label_for_forward = False):
        super().__init__(include_label_for_forward=include_label_for_forward)

        seed_all(hyperparameter["seed"])

        self.hyperparameter = hyperparameter

        config = AutoConfig.from_pretrained(self.hyperparameter["model_name"])
        config.classifier_dropout = 0.1
        config.cls_dropout = 0.1
        config.num_labels = num_classes

        self.base_model = AutoModelForSequenceClassification.from_pretrained(self.hyperparameter["model_name"], config=config)

        if num_classes == 1:
            self.loss = MSELoss()
            self.problem_type = "regression"
        elif num_classes > 1:
            self.loss = CrossEntropyLoss()
            self.problem_type = "single_label_classification"
        else:
            self.loss = BCEWithLogitsLoss()
            self.problem_type = "multi_label_classification"


    def forward(self, features):
        results = self.base_model(**features)
        y_hat = results.logits
        return y_hat

