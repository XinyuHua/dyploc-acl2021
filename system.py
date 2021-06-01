import argparse
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from transformers import (
    AdamW,
    BartTokenizer,
)
from transformers.optimization import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from dataset import TextGenDataset
from dyploc import DyplocModel

class TextGenerationTrainer(pl.LightningModule):

    def __init__(self, hparams):
        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)

        super().__init__()
        self.output_dir = f"checkpoints/{hparams.exp_name}/"
        self.step_count = 0
        self.save_hyperparameters(hparams)
        self.debug = hparams.debug
        self.batch_size = hparams.batch_size
        self.dataset_name = hparams.dataset_name

        self.max_sent_num = hparams.max_sent_num
        self.max_entity_per_sentence = hparams.max_entity_per_sentence
        self.max_concept_per_sentence = hparams.max_concept_per_sentence
        self.no_claim = hparams.no_claim
        self.no_concept = hparams.no_concept
        self.no_entity = hparams.no_entity
        self.no_pred_concept = hparams.no_pred_concept

        self.marginalization = hparams.marginalization
        self.fixed_k_size = False

        self.tokenizer = BartTokenizer.from_pretrained(
            self.hparams.model_name_or_path,
        )
        self.pad_token_id = self.tokenizer.pad_token_id
        self.model = DyplocModel(hparams)

    def get_dataloader(self, set_type, batch_size, shuffle, system_setup='oracle'):
        dataset = TextGenDataset(
            dataset_name=self.dataset_name,
            set_type=set_type,
            tokenizer=self.tokenizer,
            debug=self.debug,
            max_sent_num=self.max_sent_num,
            max_entity_per_sentence=self.max_entity_per_sentence,
            max_concept_per_sentence=self.max_concept_per_sentence,
            no_claim=self.no_claim,
            no_entity=self.no_entity,
            no_concept=self.no_concept,
            no_pred_concept=self.no_pred_concept,
            system_setup=system_setup,
        )
        return DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collater,
                          shuffle=shuffle, num_workers=0 if self.debug else 16)


    def get_lr_scheduler(self):
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps(),
        )
        scheduler = {"scheduler": scheduler, "interval": "step",
                     "frequency": 1}
        return scheduler


    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.hparams.weight_decay,
            },
            {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon
        )
        self.opt = optimizer

        scheduler = self.get_lr_scheduler()
        return [optimizer], [scheduler]

    def total_steps(self):
        return (self.dataset_size / self.hparams.batch_size) * self.hparams.max_epochs



    def train_dataloader(self):
        return self.train_loader

    def setup(self, mode):
        self.train_loader = self.get_dataloader("train", self.batch_size, shuffle=True)
        self.dataset_size = len(self.train_loader.dataset)

    def val_dataloader(self):
        return self.get_dataloader("dev", self.batch_size, shuffle=False)

    def test_dataloader(self, set_type, system_setup):
        return self.get_dataloader(set_type, self.batch_size, shuffle=False, system_setup=system_setup)

    def _step(self, batch):
        return self.model(
            context_input_ids=batch['input_ids'],
            context_attention_mask=batch['input_attn_mask'],
            context_input_scores=batch['context_input_scores'],
            decoder_input_ids=batch['dec_in'],
            decoder_attention_mask=batch['dec_attn_mask'],
            decoder_labels=batch['dec_out'],
            k_sizes=batch['k_sizes'] if not self.fixed_k_size else None,
            use_cache=False,
            pad_token_id=self.tokenizer.pad_token_id,
            marginalization=self.marginalization,
        )

    def calculate_token_acc(self, probs, target):
        preds = probs.argmax(-1).view(-1)
        flat_target = target.view(-1)
        preds_no_pad = preds[flat_target != self.tokenizer.pad_token_id]
        flat_target_no_pad = flat_target[flat_target != self.tokenizer.pad_token_id]
        acc = (preds_no_pad == flat_target_no_pad).float().mean()
        return acc

    def training_step(self, batch, batch_idx):
        outputs = self._step(batch)
        loss = outputs.loss
        # ppl = loss.exp().cpu()
        # probs = outputs.probs
        # token_acc = self.calculate_token_acc(probs, batch['dec_out'])
        scoring_loss = outputs.scoring_loss
        loss = loss + scoring_loss


        # logs = {'train_loss': loss.cpu(), 'bs': batch['dec_in'].shape[0], 'train_ppl': ppl}
        self.log('train_loss', loss.cpu(), on_step=True, prog_bar=True, logger=True)
        self.log('train_scoring_loss', scoring_loss.cpu(), on_step=True, prog_bar=False, logger=True)
        # self.log('train_acc', token_acc.cpu(), on_step=False, prog_bar=False, logger=True)
        # self.log('train_ppl', ppl, on_step=False, prog_bar=False, logger=True)

        # for i, param in enumerate(self.opt.param_groups):
        #     self.log(f"lr_group_{i}", param["lr"], on_step=True, prog_bar=False, logger=True)

        # return {"loss": loss, 'logs': logs}
        return {"loss": loss}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        outputs = self._step(batch)
        loss = outputs.loss
        ppl = loss.exp().cpu()
        probs = outputs.probs
        token_acc = self.calculate_token_acc(probs, batch['dec_out'])
        scoring_loss = outputs.scoring_loss
        loss = loss + scoring_loss

        self.log('val_loss', loss.cpu(), on_step=False, prog_bar=False, logger=True)
        self.log('val_scoring_loss', scoring_loss.cpu(), on_step=False, prog_bar=False, logger=True)
        self.log('val_acc', token_acc, on_step=False, prog_bar=False, logger=True)
        self.log('val_ppl', ppl, on_step=False, prog_bar=False, logger=True)
        return {'loss': loss, 'acc': token_acc, 'ppl': ppl}

    def calculate_teacher_forcing_accuracy(self, target, lm_logits):
        """Calculate the % of tokens that has the highest probabilities in lm_logits.

        Args:
            target [batch_size x max_len]
            lm_logits [batch_size x max_len x vocab_size]
        Returns:
            acc: a number between 0-1
        """
        preds = lm_logits.argmax(dim=-1)
        matched = target.eq(preds).long()
        matched.masked_fill_(target.eq(self.tokenizer.pad_token_id), value=0)
        total_matched = matched.sum().item()
        total_tokens = target.ne(self.tokenizer.pad_token_id).sum().item()
        return total_matched / total_tokens


    def calculate_scaled_cross_entropy(self, logits, scores, dec_out, ignore_index):
        """Calculate scaled cross-entropy loss with pre-computed sequence weights.

        Args:
             logits: [bsz, k_size, seq_len, vocab_size]
             dec_out: [bsz, seq_len]
             scores: [bsz x k_size]

        """

        bsz, k_size, seq_len, vocab_size = logits.shape
        logits = logits.view(bsz * k_size, seq_len, vocab_size)
        flat_log_probs = F.log_softmax(logits, dim=-1).view(-1, vocab_size)

        flat_target = dec_out.repeat_interleave(repeats=k_size, dim=0).view(-1, 1)

        expanded_weights = scores.repeat_interleave(repeats=seq_len, dim=0)
        ce = flat_log_probs.gather(index=flat_target, dim=-1)
        # ce = ce * expanded_weights
        ce = ce[flat_target != ignore_index]
        loss = -1 * ce.mean()
        ppl = loss.exp()

        pred = flat_log_probs.argmax(-1)
        pred = pred[flat_target.squeeze() != ignore_index]
        non_pad = flat_target[flat_target != ignore_index]
        acc = (pred == non_pad.squeeze()).float().mean()
        return loss, ppl, acc


    def calculate_cross_entropy(self, probs, dec_out, ignore_index, use_logits):
        """Calculate cross-entropy for tokens that are not ignore_index.
        Args:
            probs: (bsz, seq_len, vocab_size)
            dec_out: (bsz, seq_len)
            use_logits: if True, probs is actually logits (this is needed to
                leverage numerical stability of log_softmax)
        """
        if len(probs.shape) == 4:
            assert use_logits, "only when --trainig-method=single there is un-marginalized weights"
            bsz, k_size, seq_len, vocab_size = probs.shape
            dec_out = dec_out.repeat_interleave(repeats=k_size, dim=0)
            logits = probs.view(bsz * k_size, seq_len, vocab_size)
            flat_log_probs = F.log_softmax(logits, dim=-1).view(-1, vocab_size)
        else:
            bsz, seq_len, vocab_size = probs.shape
            flat_log_probs = probs.log().view(-1, vocab_size)
        flat_target = dec_out.view(-1, 1)

        ce = flat_log_probs.gather(index=flat_target, dim=-1)
        ce = ce[flat_target != ignore_index]

        loss = -1 * ce.mean()
        ppl = loss.exp()

        pred = flat_log_probs.argmax(-1)
        pred = pred[flat_target.squeeze() != ignore_index]
        non_pad = flat_target[flat_target != ignore_index]
        acc = (pred == non_pad.squeeze()).float().mean()
        return loss, ppl, acc
