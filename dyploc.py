from dataclasses import dataclass
import torch
from transformers import BartForConditionalGeneration
from transformers.modeling_outputs import ModelOutput
import torch.nn as nn
from typing import List, Optional, Tuple
import random
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss

@dataclass
class DyplocModelLMOutput(ModelOutput):
    logits : torch.Tensor = None
    loss: torch.Tensor = None
    scoring_loss: torch.Tensor = None
    probs: torch.Tensor = None
    past_key_values : Optional[List[torch.FloatTensor]] = None
    context_input_ids : Optional[torch.FloatTensor] = None
    context_attention_mask : Optional[torch.LongTensor] = None
    generator_dec_hidden_states : Optional[Tuple[torch.FloatTensor]] = None
    generator_dec_attentions : Optional[Tuple[torch.FloatTensor]] = None


class ScorePredictionHead(nn.Module):
    """Head for score prediction for encoder sequence hidden states,
    similar to `BartClassificationHead`, essentially one FFN layer with
    the input [last_enc_hidden_states; current_dec_hidden_states],
    and logits for the scores (which will be normalized across all sequences)."""

    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, 1)

    def forward(self, enc_hidden_states, dec_hidden_states):
        """
        enc_hidden_states (effective_bsz, dim)
        dec_hidden_states (effective_bsz, seq_len, dim)
        """
        seq_len = dec_hidden_states.shape[1]
        enc_hidden_exp = enc_hidden_states.unsqueeze(1).expand(-1, seq_len, -1)
        concat_hidden_states = torch.cat([enc_hidden_exp, dec_hidden_states], dim=2)
        hidden_states = self.dropout(concat_hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class DyplocModel(nn.Module):

    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.bart = BartForConditionalGeneration.from_pretrained(config.model_name_or_path)
        self.fixed_k_size = False
        self.learn_scoring = True
        self.scoring_head = ScorePredictionHead(
            input_dim=768 * 2,
            inner_dim=768,
            pooler_dropout=0.1,
        )
        self.cross_entropy = CrossEntropyLoss(reduction="sum")


    def forward(
        self,
        context_input_ids,
        decoder_input_ids,
        context_input_scores,
        context_attention_mask=None,
        decoder_attention_mask=None,
        decoder_labels=None,
        k_sizes=None,
        use_cache=False,
        past_key_values=None,
        pad_token_id=1,
        marginalization='seq',
    ):
        """Run forward pass with parallel op_k inputs.

        Args:
             context_input_ids (effective_bsz x in_seq_len)
             decoder_input_ids (effective_bsz x out_seq_len)
             context_input_scores (effective_bsz x 1) or (effective_bsz x out_seq_len)
             decoder_labels (real_bsz x out_seq_len )
        """

        effective_bsz = context_input_ids.shape[0]
        real_bsz = decoder_labels.shape[0]

        if self.fixed_k_size:
            k_sizes = effective_bsz // real_bsz
        else:
            assert k_sizes is not None, "`k_sizes` must be provided if not fixed"

        gen_outputs = self.bart(
            input_ids=context_input_ids,
            attention_mask=context_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
            output_hidden_states=True,
        )

        if self.learn_scoring:
            context_input_scores_logits = self.scoring_head(
                # gen_outputs.encoder_last_hidden_state[:, -1, :],
                gen_outputs.encoder_last_hidden_state[:, 0, :],
                gen_outputs.decoder_hidden_states[-1]
            )

            # calculate cross entropy by iterating over all samples in the batch
            # due to the variable op_k number
            cur_low = 0
            scoring_loss = 0.0
            num_toks = 0
            for sample_ix, chunk_size in enumerate(k_sizes):
                chunk_size = chunk_size.item()
                cur_high = cur_low + chunk_size
                cur_chunk = context_input_scores[cur_low: cur_high]
                cur_labels = cur_chunk.argmax(0)
                cur_logits = context_input_scores_logits[cur_low: cur_high]
                cur_logits = cur_logits.squeeze(2).transpose(0, 1)

                cur_real_seq_len = (decoder_input_ids[cur_low] != 1).sum().item()
                cur_labels = cur_labels[:cur_real_seq_len]
                cur_logits = cur_logits[:cur_real_seq_len, :]

                cur_low = cur_high

                scoring_loss += self.cross_entropy(cur_logits, cur_labels)
                num_toks += cur_real_seq_len
            scoring_loss = scoring_loss / num_toks
        else:
            scoring_loss = 0.0

        raw_logits = gen_outputs.logits
        # TODO: check if this is correct in inference time
        combined_log_probs = self.marginalize(
            raw_logits,
            k_sizes,
            context_input_scores.type_as(raw_logits),
            marginalization
        )

        if decoder_labels is not None:
            assert decoder_labels.shape[0] == real_bsz
            vocab_size = raw_logits.shape[-1]
            flat_target = decoder_labels.view(-1, 1)
            flat_log_probs = combined_log_probs.view(-1, vocab_size)
            ce = flat_log_probs.gather(index=flat_target, dim=-1)
            ce = ce[flat_target != pad_token_id]
            loss = -1 * ce.mean()
        else:
            loss = None

        return DyplocModelLMOutput(
            logits=raw_logits,
            loss=loss,
            scoring_loss=scoring_loss,
            probs=combined_log_probs.exp(),
            past_key_values=gen_outputs.past_key_values,
            context_input_ids=context_input_ids,
            context_attention_mask=context_attention_mask,
            generator_dec_hidden_states=gen_outputs.decoder_hidden_states,
            generator_dec_attentions=gen_outputs.decoder_attentions,
        )


    def marginalize(self, raw_logits, k_sizes, context_input_scores, method='seq'):
        """Run marginalization and calculate combined_log_probs.

        Args:
            raw_logits: (effective_bsz x out_seq_len x V)
            k_sizes: either an integer (fixed_k_size) or a tensor with k_sizes
            context_input_scores: (effective_bsz x 1) or (effective_bsz x out_seq_len),
                if the second dimension is 1, the scoring is on sequence level,
                otherwise it is on token level.
            method: either `seq` or `tok`
        Returns:
            combined_log_probs: (real_bsz x out_seq_len x V) log probabilities
                ready to use for loss calculation
        """
        effective_bsz = raw_logits.shape[0]
        raw_log_probs = F.log_softmax(raw_logits, -1)
        # prevent log(0)
        context_input_scores[context_input_scores == 0.0] = 1e-5
        seq_log_scores = context_input_scores.log()

        seq_log_socres_shape = seq_log_scores.shape
        if isinstance(k_sizes, int):
            # on sequence level
            if method == 'seq':
                assert seq_log_socres_shape[1] == 1, \
                    f"seq_log_scores shape must be ({seq_log_socres_shape[0]}, 1) but is {seq_log_socres_shape} instead"
                first_token_scores = raw_log_probs[:, :1, :]
                remainder_scores = raw_log_probs[:, 1:, :]
                scored_log_probs = torch.cat(
                    [first_token_scores + seq_log_scores.unsqueeze(-1), remainder_scores],
                    dim=1
                )
                scored_log_probs = scored_log_probs.view(
                    effective_bsz // k_sizes, k_sizes, raw_logits.shape[1], -1
                )
                combined_log_probs = torch.logsumexp(scored_log_probs, dim=1)
            else:
                scored_log_probs = raw_log_probs + seq_log_scores.unsqueeze(2)
                combined_log_probs = torch.logsumexp(scored_log_probs.view(
                    effective_bsz // k_sizes, k_sizes, raw_logits.shape[1], -1
                ), dim=1)

        else:
            if method == 'seq':
                first_token_scores = raw_log_probs[:, :1, :]
                remainder_scores = raw_log_probs[:, 1:, :]
                scored_log_probs = torch.cat(
                    [first_token_scores + seq_log_scores.unsqueeze(-1), remainder_scores],
                    dim=1
                )
                combined_log_probs = []
                ptr = 0
                for k_size in k_sizes:
                    cur_lower, cur_upper = ptr, ptr + k_size.item()
                    ptr = cur_upper

                    # 1 x seq_len x V
                    weighted_logprobs = torch.logsumexp(scored_log_probs[cur_lower: cur_upper, :, :], dim=0)
                    combined_log_probs.append(weighted_logprobs.unsqueeze(0))
                combined_log_probs = torch.cat(combined_log_probs, dim=0)
            else:
                scored_log_probs = raw_log_probs + seq_log_scores.unsqueeze(2)
                combined_log_probs = []
                ptr = 0
                for k_size in k_sizes:
                    cur_lower, cur_upper = ptr, ptr + k_size.item()
                    ptr = cur_upper
                    weighted_logprobs = torch.logsumexp(scored_log_probs[cur_lower: cur_upper, :, :], dim=0)
                    combined_log_probs.append(weighted_logprobs.unsqueeze(0))
                combined_log_probs = torch.cat(combined_log_probs, dim=0)

        return combined_log_probs


    def prepare_inputs_for_generation(self, input_ids, past, use_cache):
        encoder_outputs, decoder_cached_states = past
        return {
            'input_ids': None,
            'encoder_outputs': encoder_outputs,
            'decoder_cached_states': decoder_cached_states,
            'decoder_input_ids': input_ids,
            'use_cache': use_cache,
            'output_hidden_states': True,
        }

    def sample_or_greedy(self, next_token_logits, do_sample, temperature, top_k, top_p):
        """Sampling or greedy decoding.

        Returns:
            next_token: (batch_size)
            probs: (batch_size x vocab_size)
        """
        if do_sample:
            scores = next_token_logits
            # Temperature (higher temperature => more likely to sample low probability tokens)
            if temperature != 1.0:
                scores = scores / temperature
            # Top-p/top-k filtering
            next_token_logscores = top_k_top_p_filtering(scores, top_k=top_k, top_p=top_p)
            # Sample
            probs = F.softmax(next_token_logscores, dim=-1).squeeze()
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            # Greedy decoding
            probs = F.softmax(next_token_logits, -1).squeeze()
            next_token = torch.argmax(next_token_logits, dim=-1).squeeze(1)
        return next_token, probs

    @torch.no_grad()
    def generate(
        self,
        real_batch_size,
        k_sizes,
        context_input_ids,
        context_attention_mask,
        max_length,
        do_sample=False,
        top_k=10,
        top_p=0.9,
        hard_selection=False,
        rand_selection=False,
        tokenizer=None,
        quiet=True,
    ):
        if hard_selection: assert not rand_selection, "`hard-selection` and `random-selection` cannot be both set to True"
        bos_token_id = tokenizer.bos_token_id
        eos_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id

        fixed_k = isinstance(k_sizes, int)

        encoder = self.bart.get_encoder()
        encoder_outputs = encoder(context_input_ids, attention_mask=context_attention_mask)

        effective_batch_size = context_input_ids.shape[0]
        unfinished_sents = context_input_ids.new(real_batch_size).fill_(1)
        past = (encoder_outputs, None)
        cur_len = 0

        scoring_results = []

        # input_ids is always the effective_batch_size, after combining the
        # branches, set the selected token to all outputs in the same sample
        input_ids = torch.full((effective_batch_size, 1),
                               fill_value=bos_token_id,
                               dtype=torch.long, device=next(self.parameters()).device)
        if fixed_k:
            top_beam_ids = [i * k_sizes for i in range(real_batch_size)]
        else:
            _k_sizes_shift = torch.cat([torch.LongTensor([0]).cuda(), k_sizes[:-1]])
            top_beam_ids = torch.cumsum(_k_sizes_shift, dim=0)

        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, past=past, use_cache=True,
            )
            model_inputs['attention_mask'] = context_attention_mask
            outputs = self.bart(**model_inputs)
            past = outputs[1]
            next_token_probs = F.softmax(outputs.logits, -1)


            context_input_scores_logits = self.scoring_head(
                # gen_outputs.encoder_last_hidden_state[:, -1, :],
                outputs.encoder_last_hidden_state[:, 0, :],
                outputs.decoder_hidden_states[-1]
            )

            context_input_scores = self.calculate_branch_scores(context_input_scores_logits, k_sizes)
            scoring_results.append(context_input_scores)
            if not hard_selection and not rand_selection:
                weighted_probs = next_token_probs * context_input_scores.unsqueeze(-1)

            # combine probabilities with weights
            if fixed_k:
                weighted_probs = weighted_probs.view(real_batch_size, k_sizes, -1).sum(1)
            else:
                _combined_weighted_probs = []
                cur_lower = 0
                for sample_ix, chunk_size in enumerate(k_sizes):
                    cur_upper = cur_lower + chunk_size.item()

                    if hard_selection:
                        selected_branch = context_input_scores[cur_lower : cur_upper].argmax().item()
                        #print(selected_branch)
                        _combined_weighted_probs.append(next_token_probs[cur_lower + selected_branch, :, :])

                    elif rand_selection:
                        selected_branch = random.randrange(cur_lower, cur_upper)
                        _combined_weighted_probs.append(next_token_probs[selected_branch, :, :])

                    else:
                        cur_chunk = weighted_probs[cur_lower : cur_upper, :, :].sum(0)
                        _combined_weighted_probs.append(cur_chunk)
                    cur_lower = cur_upper
                weighted_probs = torch.cat(_combined_weighted_probs, dim=0)

            if do_sample:
                # Top-p/top-k filtering
                next_token_probs = top_k_top_p_filtering(weighted_probs, top_k=top_k, top_p=top_p, use_logits=False)
                next_token = torch.multinomial(next_token_probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_token = torch.argmax(weighted_probs, dim=-1)

            chosen_probs = weighted_probs.gather(index=next_token.view(-1, 1), dim=-1)

            # update generations and finished sentences
            if eos_token_id is not None:
                # pad finished sentences if eos_token_id exist
                tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
                #print('tokens_to_add:', tokens_to_add.tolist())
            else:
                tokens_to_add = next_token

            if not quiet:
                cur_next_toks = tokenizer.convert_ids_to_tokens(tokens_to_add)
                # print first three samples
                print(f"{cur_len} |", end="")
                for j in range(min(3, len(cur_next_toks))):
                    tok = cur_next_toks[j]
                    tok_prob = chosen_probs[j].item()
                    print(f"{tok:<10s} ({tok_prob:.2f}) |", end="")
                print()
                if cur_len == 100:
                    print()


            # expand to all branches
            if fixed_k:
                tokens_to_add = tokens_to_add.unsqueeze(1).repeat((1, k_sizes)).view(effective_batch_size, 1)
            else:
                _tokens_to_add = [_tok.repeat(_k_size.item()) for _tok, _k_size in zip(tokens_to_add, k_sizes)]
                tokens_to_add = torch.cat(_tokens_to_add, 0).view(effective_batch_size, 1)


            input_ids = torch.cat([input_ids, tokens_to_add], dim=-1)
            cur_len = cur_len + 1
            if eos_token_id is not None:
                eos_in_sents = tokens_to_add == eos_token_id
                eos_in_sents = eos_in_sents[top_beam_ids]
                eos_in_sents = eos_in_sents.squeeze(1)

                # unfinished_sents is set to zero if eos in sentence
                unfinished_sents.mul_((~eos_in_sents).long())
                #print('unfinished_sents:', unfinished_sents.tolist())

                # stop when there is a </s> in each sentence, or if we exceed the maximul length
                if unfinished_sents.max() == 0:
                    break

        scoring_results = torch.cat(scoring_results, 1)
        scoring_results_chunks = []
        cur_lower = 0
        for chunk_size in k_sizes:
            cur_upper = cur_lower + chunk_size
            scoring_results_chunks.append(scoring_results[cur_lower:cur_upper])
            cur_lower = cur_upper
        return input_ids, scoring_results_chunks

            # return self.generate_combined_branches(
            #     real_batch_size, k_sizes, context_input_ids, context_attention_mask,
            #     context_input_scores, max_length, min_length, do_sample,
            #     temperature, top_k, top_p, tokenizer, quiet
            # )


    def calculate_branch_scores(self, score_logits, k_sizes):
        cur_low = 0
        scores = []
        for sample_ix, chunk_size in enumerate(k_sizes):
            chunk_size = chunk_size.item()
            cur_high = cur_low + chunk_size
            cur_chunk = score_logits[cur_low: cur_high, 0]
            cur_scores = F.softmax(cur_chunk, 0)
            scores.append(cur_scores)
            cur_low = cur_high
        return torch.cat(scores, dim=0)

    def generate_combined_branches(
        self,
        real_batch_size,
        k_sizes,
        context_input_ids,
        context_attention_mask,
        context_input_scores,
        max_length,
        min_length,
        do_sample=False,
        temperature=1.0,
        top_k=10,
        top_p=0.9,
        tokenizer=None,
        quiet=True
    ):
        """Combine probabilities according to the input sequence scores."""
        bos_token_id = tokenizer.bos_token_id
        eos_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id

        fixed_k = isinstance(k_sizes, int)

        encoder = self.bart.get_encoder()
        encoder_outputs = encoder(context_input_ids, attention_mask=context_attention_mask)

        effective_batch_size = context_input_ids.shape[0]
        unfinished_sents = context_input_ids.new(real_batch_size).fill_(1)
        past = (encoder_outputs, None)
        cur_len = 0

        scoring_results = []

        # input_ids is always the effective_batch_size, after combining the
        # branches, set the selected token to all outputs in the same sample
        input_ids = torch.full((effective_batch_size, 1),
                               fill_value=bos_token_id,
                               dtype=torch.long, device=next(self.parameters()).device)
        if fixed_k:
            top_beam_ids = [i * k_sizes for i in range(real_batch_size)]
        else:
            _k_sizes_shift = torch.cat([torch.LongTensor([0]).cuda(), k_sizes[:-1]])
            top_beam_ids = torch.cumsum(_k_sizes_shift, dim=0)

        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, past=past, use_cache=True,
            )
            model_inputs['attention_mask'] = context_attention_mask
            outputs = self.bart(**model_inputs)
            past = outputs[1]
            next_token_probs = F.softmax(outputs.logits, -1)


            context_input_scores_logits = self.scoring_head(
                # gen_outputs.encoder_last_hidden_state[:, -1, :],
                outputs.encoder_last_hidden_state[:, 0, :],
                outputs.decoder_hidden_states[-1]
            )

            context_input_scores = self.calculate_branch_scores(context_input_scores_logits, k_sizes)
            scoring_results.append(context_input_scores)
            weighted_probs = next_token_probs * context_input_scores.unsqueeze(-1)

            # combine probabilities with weights
            if fixed_k:
                weighted_probs = weighted_probs.view(real_batch_size, k_sizes, -1).sum(1)
            else:
                _combined_weighted_probs = []
                cur_lower = 0
                for sample_ix, chunk_size in enumerate(k_sizes):
                    cur_upper = cur_lower + chunk_size
                    cur_chunk = weighted_probs[cur_lower : cur_upper, :, :].sum(0)
                    _combined_weighted_probs.append(cur_chunk)
                    cur_lower = cur_upper
                weighted_probs = torch.cat(_combined_weighted_probs, dim=0)

            if do_sample:
                # Top-p/top-k filtering
                next_token_probs = top_k_top_p_filtering(weighted_probs, top_k=top_k, top_p=top_p, use_logits=False)
                next_token = torch.multinomial(next_token_probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_token = torch.argmax(weighted_probs, dim=-1)

            chosen_probs = weighted_probs.gather(index=next_token.view(-1, 1), dim=-1)

            # update generations and finished sentences
            if eos_token_id is not None:
                # pad finished sentences if eos_token_id exist
                tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
                #print('tokens_to_add:', tokens_to_add.tolist())
            else:
                tokens_to_add = next_token

            if not quiet:
                cur_next_toks = tokenizer.convert_ids_to_tokens(tokens_to_add)
                # print first three samples
                print(f"{cur_len} |", end="")
                for j in range(min(3, len(cur_next_toks))):
                    tok = cur_next_toks[j]
                    tok_prob = chosen_probs[j].item()
                    print(f"{tok:<10s} ({tok_prob:.2f}) |", end="")
                print()
                if cur_len == 100:
                    print()


            # expand to all branches
            if fixed_k:
                tokens_to_add = tokens_to_add.unsqueeze(1).repeat((1, k_sizes)).view(effective_batch_size, 1)
            else:
                _tokens_to_add = [_tok.repeat(_k_size.item()) for _tok, _k_size in zip(tokens_to_add, k_sizes)]
                tokens_to_add = torch.cat(_tokens_to_add, 0).view(effective_batch_size, 1)


            input_ids = torch.cat([input_ids, tokens_to_add], dim=-1)
            cur_len = cur_len + 1
            if eos_token_id is not None:
                eos_in_sents = tokens_to_add == eos_token_id
                eos_in_sents = eos_in_sents[top_beam_ids]
                eos_in_sents = eos_in_sents.squeeze(1)

                # unfinished_sents is set to zero if eos in sentence
                unfinished_sents.mul_((~eos_in_sents).long())
                #print('unfinished_sents:', unfinished_sents.tolist())

                # stop when there is a </s> in each sentence, or if we exceed the maximul length
                if unfinished_sents.max() == 0:
                    break

        scoring_results = torch.cat(scoring_results, 1)
        scoring_results_chunks = []
        cur_lower = 0
        for chunk_size in k_sizes:
            cur_upper = cur_lower + chunk_size
            scoring_results_chunks.append(scoring_results[cur_lower:cur_upper])
            cur_lower = cur_upper
        return input_ids, scoring_results_chunks



    def generate_individual_branches(
        self,
        context_input_ids,
        context_attention_mask,
        max_length,
        min_length,
        do_sample=False,
        temperature=1.0,
        top_k=10,
        top_p=0.9,
        tokenizer=None,
        quiet=True
    ):
        """Generate output for each encoder outputs, no merging branches."""

        bos_token_id = tokenizer.bos_token_id
        eos_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id

        encoder = self.bart.get_encoder()
        encoder_outputs = encoder(context_input_ids, attention_mask=context_attention_mask)
        batch_size = context_input_ids.shape[0]
        unfinished_sents = context_input_ids.new(batch_size).fill_(1)

        past = (encoder_outputs, None)
        cur_len = 0
        input_ids = torch.full((batch_size, 1),
                               fill_value=bos_token_id,
                               dtype=torch.long, device=next(self.parameters()).device)
        while cur_len < max_length:

            model_inputs = self.prepare_inputs_for_generation(
                input_ids, past=past, use_cache=True,
            )
            outputs = self.bart(**model_inputs)
            past = outputs[1]
            next_token, probs = self.sample_or_greedy(outputs.logits, do_sample, temperature, top_k, top_p)
            chosen_probs = probs.gather(index=next_token.unsqueeze(-1), dim=-1)

            if not quiet:
                cur_next_toks = tokenizer.convert_ids_to_tokens(next_token)
                # print first three samples
                print_str = ""
                for j in range(min(3, len(cur_next_toks))):
                    tok = cur_next_toks[j]
                    tok_prob = chosen_probs[j].item()
                    print_str += f"{tok:<10s} ({tok_prob:.2f}) |"
                print(print_str)

            # update generations and finished sentences
            if eos_token_id is not None:
                # pad finished sentences if eos_token_id exist
                tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
            else:
                tokens_to_add = next_token


            # tokens_to_add = tokens_to_add.repeat((1, k_size)).view(effective_batch_size, 1)


            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(1)], dim=-1)
            cur_len = cur_len + 1
            if eos_token_id is not None:
                eos_in_sents = tokens_to_add == eos_token_id
                # unfinished_sents is set to zero if eos in sentence
                unfinished_sents.mul_((~eos_in_sents).long())
                # stop when there is a </s> in each sentence, or if we exceed the maximul length
                if unfinished_sents.max() == 0:
                    break

        return input_ids


    def _generate_combined_branches(
        self,
        real_batch_size,
        context_input_ids,
        context_attention_mask,
        context_input_scores,
        max_length,
        min_length,
        do_sample=False,
        temperature=1.0,
        top_k=10,
        top_p=0.9,
        combine_branches=True,
        bos_token_id=0,
        pad_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=0,
        no_weighted_sum=False,
        tokenizer=None,
        quiet=True,
        oracle_op=False,
    ):
        """
        Greedy or sampling based decoding.

        Args:
            context_input_ids: (effective_bsz x seq_len)
            context_attention_mask: (effective_bsz x seq_len)

        """
        encoder = self.bart.get_encoder()
        encoder_outputs = encoder(context_input_ids, attention_mask=context_attention_mask)

        effective_batch_size = context_input_ids.shape[0]
        k_size = effective_batch_size // real_batch_size
        top_beam_ids = [i * k_size for i in range(real_batch_size)]


        if no_weighted_sum:
            unfinished_sents = context_input_ids.new(effective_batch_size).fill_(1)
        else:
            unfinished_sents = context_input_ids.new(real_batch_size).fill_(1)

        past = (encoder_outputs, None)
        cur_len = 0
        input_ids = torch.full((effective_batch_size, 1),
                               fill_value=decoder_start_token_id,
                               dtype=torch.long, device=next(self.parameters()).device)

        while cur_len < max_length:

            model_inputs = self.prepare_inputs_for_generation(
                input_ids, past=past, use_cache=True,
            )
            outputs = self.bart(**model_inputs)

            all_beams_logits = outputs.logits
            vocab_size = all_beams_logits.shape[-1]

            past = outputs[1]

            if no_weighted_sum:
                next_token_logits = all_beams_logits.view(effective_batch_size, vocab_size)
                next_token, probs = self.sample_or_greedy(next_token_logits, do_sample, temperature, top_k, top_p)
            elif context_input_scores is None:
                # uniformly combine logits from different beams
                combined_logits = all_beams_logits.view(real_batch_size, k_size, -1, vocab_size)
                next_token_logits = combined_logits.mean(dim=1).squeeze(1)
                next_token, probs = self.sample_or_greedy(next_token_logits, do_sample, temperature, top_k, top_p)
            else:
                next_token_probs = F.softmax(all_beams_logits, -1)
                weighted_probs = next_token_probs * context_input_scores.unsqueeze(-1)
                probs = weighted_probs.view(real_batch_size, k_size, -1).sum(1)

                if do_sample:
                    probs = top_k_filtering_on_probs(probs, top_k)
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_token = torch.argmax(probs, dim=-1)

            chosen_probs = probs.gather(index=next_token.unsqueeze(-1), dim=-1)

            if not quiet:
                cur_next_toks = tokenizer.convert_ids_to_tokens(next_token)
                # print first three samples
                print_str = ""
                for j in range(min(3, len(cur_next_toks))):
                    tok = cur_next_toks[j]
                    tok_prob = chosen_probs[j].item()
                    print_str += f"{tok:<10s} ({tok_prob:.2f}) |"
                print(print_str)

            # update generations and finished sentences
            if eos_token_id is not None:
                # pad finished sentences if eos_token_id exist
                tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
            else:
                tokens_to_add = next_token

            if not no_weighted_sum:
                tokens_to_add = tokens_to_add.repeat((1, k_size)).view(effective_batch_size, 1)
            else:
                tokens_to_add = tokens_to_add.view(effective_batch_size, 1)

            input_ids = torch.cat([input_ids, tokens_to_add], dim=-1)
            cur_len = cur_len + 1
            if eos_token_id is not None:
                eos_in_sents = tokens_to_add == eos_token_id
                if not no_weighted_sum:
                    eos_in_sents = eos_in_sents[top_beam_ids]
                eos_in_sents = eos_in_sents.squeeze(1)

                # unfinished_sents is set to zero if eos in sentence
                unfinished_sents.mul_((~eos_in_sents).long())

                # stop when there is a </s> in each sentence, or if we exceed the maximul length
                if unfinished_sents.max() == 0:
                    break

        return input_ids

def top_k_filtering_on_probs(
    probs,
    top_k,
    filter_value=0.0,
):
    top_k = min(max(top_k, 1), probs.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = probs < torch.topk(probs, top_k)[0][..., -1, None]
    probs[indices_to_remove] = filter_value
    return probs

def top_k_top_p_filtering(
    scores,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
    use_logits=True,
):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317

        If `use_logits` is True, scores are logits, otherwise they are probs
    """
    if not use_logits:
        filter_value = 0.0

    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), scores.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
        scores[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        if use_logits:
            cumulative_probs = torch.cumsum(F.softmax(sorted_scores, dim=-1), dim=-1)
        else:
            cumulative_probs = torch.cumsum(sorted_scores, dim=-1)


        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores[indices_to_remove] = filter_value
    return scores
