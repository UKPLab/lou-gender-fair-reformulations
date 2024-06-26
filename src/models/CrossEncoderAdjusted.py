import os
from typing import Dict, Type, Callable

import numpy
import pandas
import torch
import transformers
import wandb
from sentence_transformers import CrossEncoder, SentenceTransformer
from sentence_transformers.evaluation import SentenceEvaluator
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm, trange


class EvaluatorWrapper:
    def __init__(self,evaluators):
        self.evaluators = evaluators

    def __call__(self, model, output_path, epoch, steps):

        scores = []

        if self.store_model and self.use_mlflow:
            os.system("mkdir -p " + self.model_dump_path + "/models/" + str(epoch))
            os.system("mkdir -p " + self.model_dump_path + "/preds/" + str(epoch))

        for evaluator in self.evaluators:
            scores.append(evaluator(model, output_path, epoch, steps))

        if self.use_wandb:
            wandb.log({
                "epoch": epoch
            }, commit=True)

        #if self.store_model and self.use_mlflow:
        #    if epoch == 4:
        #        mlflow.log_artifacts(self.model_dump_path, artifact_path="artifacts")
        #        os.system("rm -rf " + self.model_dump_path)

        return 0

class Evaluator:
    def __init__(self):
        self.best_score = 0

    def save_preds(self, ids, preds, epoch, prefix):
        pred_frame = pandas.DataFrame([
            [id, pred]
            for id, pred in zip(ids, preds)
        ], columns=["id", "pred"])

        file_path = os.path.join(wandb.run.dir, prefix + "preds-" + str(epoch) +".csv")

        pred_frame.to_csv(file_path)

        return file_path

class CrossEvaluator(Evaluator):
    def __init__(self, train_samples, dev_samples, test_samples, dev_threshold=False, name="", train_loss=None, use_wandb=False, use_mlflow=False, model_dump_path=None, sample_columns=[], store_model=False):
        super().__init__()

        self.train_samples = train_samples
        self.dev_samples = dev_samples
        self.test_samples = test_samples

        if type(train_samples[0].texts[0]) == tuple:
            self.train_pairs, self.train_labels = numpy.array([[ele.texts, ele.label] for ele in train_samples]).T
            self.dev_pairs, self.dev_labels = numpy.array([[ele.texts, ele.label] for ele in dev_samples]).T
            self.test_pairs, self.test_labels = numpy.array([[ele.texts, ele.label] for ele in test_samples]).T
        else:
            self.train_pairs, self.train_labels = numpy.array([
                [[ele.texts[i] for i in range(len(sample_columns))], ele.label]
                for ele in train_samples
            ]).T

            self.dev_pairs, self.dev_labels = numpy.array([
                [[ele.texts[i] for i in range(len(sample_columns))], ele.label]
                for ele in dev_samples
            ]).T

            self.test_pairs, self.test_labels = numpy.array([
                [[ele.texts[i] for i in range(len(sample_columns))], ele.label]
                for ele in test_samples
            ]).T

        self.name = name
        self.train_loss = train_loss
        self.use_wandb = use_wandb
        self.use_mlflow = use_mlflow
        self.dev_threshold = dev_threshold

        self.model_dump_path = model_dump_path
        self.store_model = store_model
        self.sample_columns = sample_columns

        if self.store_model:
            self.model_path = self.model_dump_path + "/models/"
            self.preds_path = self.model_dump_path + "/preds/"

        self.evalutor_samples = {
            "train": [self.train_pairs, self.train_labels],
            "dev": [self.dev_pairs, self.dev_labels],
            "test": [self.test_pairs, self.test_labels]
        }




class CrossF1Evaluator(CrossEvaluator):


    def __call__(self, model, output_path, epoch, steps):

        performance = {}

        loss_function = nn.CrossEntropyLoss()

        for name, (pairs, labels) in self.evalutor_samples.items():

            if type(pairs[0][0]) == tuple:
                raw_predictions = model.predict(pairs, convert_to_numpy=True)
                raw_predictions = torch.nn.utils.rnn.pad_sequence([torch.Tensor(ele) for ele in raw_predictions.tolist()], batch_first=True)
                raw_predictions, labels = model.clean_inference_predictions(pairs, raw_predictions, labels)
                labels = labels.numpy().astype(int)
            else:
                raw_predictions = model.predict(pairs, convert_to_tensor=True).cpu().detach()

            loss = loss_function(raw_predictions, torch.Tensor(labels.tolist()).long())
            predictions = numpy.argmax(raw_predictions.numpy(), axis=1).tolist()
            raw_predictions = nn.Softmax(dim=1)(raw_predictions).numpy().tolist()

            performance[name + " f1-macro"] = round(f1_score(labels.tolist(), predictions, average="macro"), 3)
            performance[name + " accuracy"] = round(accuracy_score(labels.tolist(), predictions), 3)
            performance[name + " loss"] = float(loss)

            if type(pairs[0][0]) == tuple:
                pairs_length = [len(pair[0]) for pair in pairs]
                pairs = [[str(ele)] for ele in pairs]

                resulting_preds = []
                raw_resulting_preds = []
                resulting_labels = []
                last_length = 0
                for length in pairs_length:
                    resulting_preds.append(tuple(predictions[last_length:length]))
                    raw_resulting_preds.append(tuple(raw_predictions[last_length:length]))
                    resulting_labels.append(tuple(labels[last_length:length]))



            else:
                resulting_preds = predictions
                raw_resulting_preds = raw_predictions
                resulting_labels = labels


            predictions = pandas.DataFrame([
                id + [str(pred),  raw_pred, str(label), epoch]
                for id, pred, raw_pred, label in zip(pairs, resulting_preds, raw_resulting_preds, resulting_labels)
            ], columns = self.sample_columns + ["prediction", "raw_predictions", "label", "epoch"])



        if self.use_wandb:
            wandb.log(performance, commit=False)




            if self.store_model:

                if performance["dev f1-macro"] > self.best_score:
                    self.best_score = performance["dev f1-macro"]

                    if self.store_model:
                        model.save(self.model_path + "/")

                    if self.store_model and self.use_mlflow:
                        predictions.to_csv(self.preds_path + "/" + name + ".csv", index=False)


class CrossEncoderAdjusted(CrossEncoder):

    def fit(self,
            train_dataloader: DataLoader,
            evaluator: SentenceEvaluator = None,
            epochs: int = 1,
            loss_fct = None,
            activation_fct = nn.Identity(),
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = transformers.AdamW,
            optimizer_params: Dict[str, object] = {'lr': 2e-5, 'eps': 1e-6, 'correct_bias': False},
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            use_amp: bool = False,
            callback: Callable[[float, int, int], None] = None,
            accumulation_steps: int = 1
            ):
        """
        Train the model with the given training objective
        Each training objective is sampled in turn for one batch.
        We sample only as many batches from each objective as there are in the smallest one
        to make sure of equal training with each dataset.

        :param train_dataloader: DataLoader with training InputExamples
        :param evaluator: An evaluator (sentence_transformers.evaluation) evaluates the model performance during training on held-out dev data. It is used to determine the best model that is saved to disc.
        :param epochs: Number of epochs for training
        :param loss_fct: Which loss function to use for training. If None, will use nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss()
        :param acitvation_fct: Activation function applied on top of logits output of model.
        :param scheduler: Learning rate scheduler. Available schedulers: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        :param warmup_steps: Behavior depends on the scheduler. For WarmupLinear (default), the learning rate is increased from o up to the maximal learning rate. After these many training steps, the learning rate is decreased linearly back to zero.
        :param optimizer_class: Optimizer
        :param optimizer_params: Optimizer parameters
        :param weight_decay: Weight decay for model parameters
        :param evaluation_steps: If > 0, evaluate the model using evaluator after each number of training steps
        :param output_path: Storage path for the model and evaluation files
        :param save_best_model: If true, the best model (according to evaluator) is stored at output_path
        :param max_grad_norm: Used for gradient normalization.
        :param use_amp: Use Automatic Mixed Precision (AMP). Only for Pytorch >= 1.6.0
        :param callback: Callback function that is invoked after each evaluation.
                It must accept the following three parameters in this order:
                `score`, `epoch`, `steps`
        """
        train_dataloader.collate_fn = self.smart_batching_collate

        if use_amp:
            scaler = torch.cuda.amp.GradScaler()

        self.model.to(self._target_device)

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)

        self.best_score = -9999999
        num_train_steps = int(len(train_dataloader) * epochs)

        # Prepare optimizers
        param_optimizer = list(self.model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        if isinstance(scheduler, str):
            scheduler = SentenceTransformer._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

        if loss_fct is None:
            loss_fct = nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss()


        skip_scheduler = False
        for epoch in trange(epochs, desc="Epoch"):
            training_steps = 0
            self.model.zero_grad()
            self.model.train()

            for features, labels in tqdm(train_dataloader, desc="Iteration", smoothing=0.05):
                model_predictions = self.model(**features, return_dict=True)
                logits = activation_fct(model_predictions.logits)
                if self.config.num_labels == 1:
                    logits = logits.view(-1)

                if len(logits.shape) == 3:
                    logits, labels = self.filter_predictions_for_token_classifications(logits, labels)

                loss_value = loss_fct(logits, labels)
                loss_value.backward()

                if (training_steps+1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()

                    optimizer.zero_grad()

                    if not skip_scheduler:
                        scheduler.step()

                training_steps += 1

                if evaluator is not None and evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    self._eval_during_training(evaluator, output_path, save_best_model, epoch, training_steps, callback)

                    self.model.zero_grad()
                    self.model.train()

            if evaluator is not None:
                self.model.eval()
                self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1, callback)

    def filter_predictions_for_token_classifications(self, logits, labels):

        filtered_logits = torch.flatten(logits, 0, 1)
        filtered_labels = torch.flatten(labels)

        filtered_logits = filtered_logits[(filtered_labels > -100)]
        filtered_labels = filtered_labels[(filtered_labels > -100)]

        return filtered_logits, filtered_labels

    def clean_inference_predictions(self, texts, predictions, labels):
        tokenized = self.tokenizer([ele[0] for ele in texts], padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_length, is_split_into_words=True, return_offsets_mapping=True)
        encoded_labels = self.encode_labels(labels, tokenized)

        cleaned_logits, cleaned_labels = self.filter_predictions_for_token_classifications(predictions, torch.Tensor(encoded_labels))

        return cleaned_logits, cleaned_labels

    def smart_batching_collate(self, batch):
        texts = [[] for _ in range(len(batch[0].texts))]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                if type(text) == str:
                    texts[idx].append(text.strip())
                else:
                    texts[idx].append(text)

            labels.append(example.label)


        if type(texts[0][0]) == tuple:
            tokenized = self.tokenizer(*texts, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_length, is_split_into_words=True, return_offsets_mapping=True)
            labels = self.encode_labels(labels, tokenized)
            del tokenized.data["offset_mapping"]
        else:
            tokenized = self.tokenizer(*texts, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_length)


        labels = torch.tensor(labels, dtype=torch.float if self.config.num_labels == 1 else torch.long).to(self._target_device)
        for name in tokenized:
            tokenized[name] = tokenized[name].to(self._target_device)


        return tokenized, labels

    def smart_batching_collate_text_only(self, batch):
        texts = [[] for _ in range(len(batch[0]))]

        for example in batch:
            for idx, text in enumerate(example):
                if type(text) == str:
                    texts[idx].append(text.strip())
                else:
                    texts[idx].append(text)

        if type(texts[0][0]) == tuple:
            tokenized = self.tokenizer(*texts, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_length, is_split_into_words=True, return_offsets_mapping=True)
            del tokenized.data["offset_mapping"]
        else:
            tokenized = self.tokenizer(*texts, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_length)


        for name in tokenized:
            tokenized[name] = tokenized[name].to(self._target_device)

        return tokenized


    def encode_labels(self, labels, encodings):
        encoded_labels = []
        for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
            # create an empty array of -100
            doc_enc_labels = numpy.ones(len(doc_offset),dtype=int) * -100
            arr_offset = numpy.array(doc_offset)

            # set labels whose first offset position is 0 and the second is not 0
            doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
            encoded_labels.append(doc_enc_labels.tolist())

        return encoded_labels