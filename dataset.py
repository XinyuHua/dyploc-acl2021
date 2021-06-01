import torch
from torch.utils.data import Dataset
import json
import numpy as np
from tqdm import tqdm
import time

SYSTEM_DATA_PATH = './trainable/'


class TextGenDataset(Dataset):

    def __init__(
        self,
        dataset_name,
        set_type,
        tokenizer,
        debug=False,
        max_sent_num=8,
        max_entity_per_sentence=20,
        max_concept_per_sentence=20,
        no_entity=False,
        no_claim=False,
        no_concept=False,
        no_pred_concept=False,
        system_setup='oracle',
    ):

        super().__init__()
        self.dataset_name = dataset_name
        self.debug = debug
        self.tokenizer = tokenizer
        self.set_type = set_type
        self.system_setup = system_setup

        self.max_sent_num = max_sent_num
        self.max_entity_per_sentence = max_entity_per_sentence
        self.max_concept_per_sentence = max_concept_per_sentence
        #self.max_sent_num = 10
        #self.max_entity_per_sentence = 20
        #self.max_concept_per_sentence = 20
        
        self.no_entity = no_entity
        self.no_concept = no_concept
        self.no_claim = no_claim
        self.no_pred_concept = no_pred_concept

        self.pad_token_id = tokenizer.pad_token_id
        self.ent_sep = '//'
        self.title_sep = '///'
        self.concept_sep = '////'
        # use sep_token (</s>) to split opinion sentences
        self.ent_sep_id = self.tokenizer._convert_token_to_id(self.ent_sep)
        self.title_sep_id = self.tokenizer._convert_token_to_id(self.title_sep)
        self.concept_sep_id = self.tokenizer._convert_token_to_id(self.concept_sep)

        self.ID = []
        # branch inputs
        self.encoder_in = []
        self.encoder_in_str = []
        # input scores (goldstandard)
        self.encoder_scores = []
        self.entities = []
        self.title = []
        self.output = []

        self._load_data(self.set_type)
        self.oracle = False


    def _load_data(self, set_type):
        t0 = time.time()
        input_len, output_len = [], []
        concept_per_branch = []
        data_path = SYSTEM_DATA_PATH + f'{self.dataset_name}.{set_type}.with_op.jsonl'
        data_lns = [json.loads(ln) for ln in open(data_path)]

        if self.system_setup == 'system':
            concept_path = f'./concept_predictions/{self.dataset_name}.{set_type}.concept-generation.sent.core3.top5.jsonl'

            if not self.no_concept:
                concept_dict = dict()
                for ln in open(concept_path):
                    cur_obj = json.loads(ln)
                    branch_input_dict = dict()
                    branch_preds = cur_obj['branch_predictions'] if 'branch_predictions' in cur_obj else cur_obj['branch_inputs']
                    for item in branch_preds:
                        if len(item) == 0:
                            continue
                        cur_branch_concepts = item['input_concepts']
                        if not self.no_pred_concept:
                            cur_branch_concepts += item['predicted_concepts']
                        branch_input_dict[item['branch_id']] = cur_branch_concepts
                    concept_dict[cur_obj['id']] = branch_input_dict

        elif self.no_pred_concept:
            # load core concepts, only these will be used
            core_concept_path = SYSTEM_DATA_PATH + f'core_concepts/{self.dataset_name}.{set_type}.cset=top10k.core_concepts.method=3.top5.jsonl'
            core_concepts = dict()
            for ln in open(core_concept_path):
                cur_obj = json.loads(ln)
                core_concepts[cur_obj['id']] = cur_obj['core_concepts']




        for ln_id, data_obj in tqdm(enumerate(data_lns), desc=f'loading {set_type} data'):
            post_id = data_obj['id']
            self.ID.append(post_id)
            title = data_obj['title']
            cur_title_tok_ids = self.tokenizer.encode(title, add_special_tokens=False)

            encoder_branches = []
            encoder_branch_str = []
            reference_sent_tok_ids = []
            branch_mask_ids = []


            if self.system_setup == 'system' and not self.no_concept:
                post_concept_dict = concept_dict[post_id] if post_id in concept_dict else dict()

            for sent_id, (sent_type, ref_sent, branch_input) in enumerate(zip(data_obj['sentence_types'], data_obj['reference_sentences'], data_obj['branch_input'])):
                """
                reference_sent_tok_ids = ['this', 'is', 'a', 'sentence', '.', 'here', 'is', 'another', 'one', '.']
                branch_mask_ids = [(0, 5), (5, 10)]
                """
                if sent_id == self.max_sent_num:
                    break

                if sent_id > 0:
                    ref_sent = ' ' + ref_sent
                cur_ref_sent_tok = self.tokenizer.encode(ref_sent, add_special_tokens=False)

                ref_lower_id = len(reference_sent_tok_ids)
                ref_upper_id = ref_lower_id + len(cur_ref_sent_tok)
                reference_sent_tok_ids.extend(cur_ref_sent_tok)

                if sent_type == 'claims' and branch_input['target_entity'] is not None:
                    # opinion: title + entities + concepts + claims
                    cur_branch_tok_ids = cur_title_tok_ids + [self.title_sep_id]
                    cur_branch_tok_str = title + " "

                    if not self.no_entity:
                        ents = [branch_input['target_entity']] + branch_input['other_entities']
                        for ent in ents[:self.max_entity_per_sentence]:
                            ent = get_wiki_surface(ent)
                            ent_id = self.tokenizer.encode(ent, add_special_tokens=False)
                            cur_branch_tok_ids.extend(ent_id)
                            cur_branch_tok_ids.append(self.ent_sep_id)
                            cur_branch_tok_str += ent + self.ent_sep

                    if not self.no_concept:
                        if self.system_setup == 'oracle':
                            _concepts = branch_input['concepts']
                            if self.no_pred_concept: # remove other concepts
                                cur_core = core_concepts[post_id] if post_id in core_concepts else []
                                _concepts = [item for item in _concepts if item in cur_core]
                        else:
                            _concepts = post_concept_dict[sent_id] if sent_id in post_concept_dict else []
                        _seen = set()
                        for con in _concepts[:self.max_concept_per_sentence]:
                            if con in _seen:
                                continue
                            _seen.add(con)
                            con_id = self.tokenizer.encode(con, add_special_tokens=False)
                            cur_branch_tok_ids.extend(con_id)
                            cur_branch_tok_ids.append(self.concept_sep_id)
                            cur_branch_tok_str += con + ' [sep] '
                        concept_per_branch.append(len(_seen))

                    if not self.no_claim:
                        cur_claim_tok_ids = self.tokenizer.encode(branch_input['claims'], add_special_tokens=False)
                        cur_branch_tok_ids.extend(cur_claim_tok_ids)
                        cur_branch_tok_str += branch_input['claims']

                elif sent_type in ['claims', 'facts', 'quotes']:
                    # title + entities + concepts
                    cur_branch_tok_ids = cur_title_tok_ids + [self.title_sep_id]
                    cur_branch_tok_str = title + " "
                    if not self.no_entity:
                        if branch_input['target_entity'] is not None:
                            ents = [branch_input['target_entity']] + branch_input['other_entities']
                            for ent in ents[:self.max_entity_per_sentence]:
                                ent = get_wiki_surface(ent)
                                ent_id = self.tokenizer.encode(ent, add_special_tokens=False)
                                cur_branch_tok_ids.extend(ent_id)
                                cur_branch_tok_ids.append(self.ent_sep_id)
                                cur_branch_tok_str += ent + self.ent_sep

                    if not self.no_concept:
                        if self.system_setup == 'oracle':
                            _concepts = branch_input['concepts']
                            if self.no_pred_concept:
                                cur_core = core_concepts[post_id] if post_id in core_concepts else []
                                _concepts = [item for item in _concepts if item in cur_core]
                        else:
                            _concepts = post_concept_dict[sent_id] if sent_id in post_concept_dict else []
                        _seen = set()
                        for con in _concepts[:self.max_concept_per_sentence]:
                            if con in _seen:
                                continue
                            _seen.add(con)
                            con_id = self.tokenizer.encode(con, add_special_tokens=False)
                            cur_branch_tok_ids.extend(con_id)
                            cur_branch_tok_ids.append(self.concept_sep_id)
                            cur_branch_tok_str += con + ' [sep] '
                        concept_per_branch.append(len(_seen))
                elif sent_type == 'others':
                    continue

                encoder_branches.append(cur_branch_tok_ids)
                encoder_branch_str.append(cur_branch_tok_str)
                branch_mask_ids.append((ref_lower_id, ref_upper_id))
            if len(reference_sent_tok_ids) > 1022:
                reference_sent_tok_ids = reference_sent_tok_ids[:1022]

            tgt_toks = [self.tokenizer.bos_token_id] + reference_sent_tok_ids + [self.tokenizer.eos_token_id]
            self.output.append(tgt_toks)
            output_len.append(len(tgt_toks))

            branch_scores = self._make_branch_scores(branch_mask_ids, len(reference_sent_tok_ids))
            self.encoder_in.append(encoder_branches)
            self.encoder_in_str.append(encoder_branch_str)
            self.encoder_scores.append(branch_scores)
            self.title.append(data_obj['title'])
            if self.debug and len(self.ID) >= 20:
                break


        print(f'{len(self.ID)} samples loaded in {time.time() - t0:.2f} secs')
        print(f'output length: min={min(output_len)} max={max(output_len)} avg={np.mean(output_len):.2f}')
        print(f'average {np.mean(concept_per_branch):.2f} concepts per branch')

    
    def _make_branch_scores(self, branch_mask_ids, total_len):
        """Create binary mask for branches.
        Args:
            branch_mask_ids (List[Tuple[Int]]): list of start, end
                pairs of token ids
            total_len (int)
        Returns:
            result (List[List[Int]]): 1 for match, 0 otherwise
        """
        result = np.full(shape=(len(branch_mask_ids), total_len), fill_value=0, dtype=np.int)
        for b, mask in enumerate(branch_mask_ids):
            result[b][mask[0]: mask[1]] = 1
        return result


    def __len__(self):
        return len(self.ID)
    

    def __getitem__(self, index):
        result = {
            "id": self.ID[index],
            "encoder_branch": self.encoder_in[index],
            "encoder_branch_toks": self.encoder_in_str[index],
            "encoder_scores": self.encoder_scores[index],
            "title": self.title[index],
            "target": self.output[index],
            }
        return result


    def collater(self, samples):
        batch = dict()
        batch['id'] = [s['id'] for s in samples]
        real_batch_size = len(samples)
        k_sizes = [len(s['encoder_branch']) for s in samples]
        batch['k_sizes'] = torch.LongTensor(k_sizes)
        effective_batch_size = sum(k_sizes)

        input_len, tgt_len = [], []
        for sample in samples:
            for b in sample['encoder_branch']:
                input_len.append(len(b))
            tgt_len.append(len(sample['target']))

        input_ids = np.full([effective_batch_size, max(input_len)],
                fill_value=self.tokenizer.pad_token_id,
                dtype=np.int)
        seq_scores = np.full([effective_batch_size, max(tgt_len)],
                              fill_value=0.0, dtype=np.float)
        dec_in = np.full([effective_batch_size, max(tgt_len)],
                          fill_value=self.tokenizer.pad_token_id,
                          dtype=np.int)
        dec_out = np.full([real_batch_size, max(tgt_len)],
                          fill_value=self.tokenizer.pad_token_id,
                          dtype=np.int)
        global_ix = 0
        for s, sample in enumerate(samples):
            cur_tgt = sample['target']
            cur_k_size = len(sample['encoder_branch'])
            for ix, op in enumerate(sample['encoder_branch']):
                input_ids[global_ix, :len(op)] = op
                cur_score = sample['encoder_scores'][ix]
                seq_scores[global_ix, :len(cur_score)] = cur_score
                seq_scores[global_ix, len(cur_score) - 1] = 1/cur_k_size
                dec_in[global_ix, :len(cur_tgt) - 1] = cur_tgt[:-1]
                global_ix += 1
            dec_out[s, :len(cur_tgt) - 1] = cur_tgt[1:]

        input_ids_pt = torch.LongTensor(input_ids)
        batch['input_ids'] = input_ids_pt
        batch['input_attn_mask'] = (input_ids_pt != self.tokenizer.pad_token_id).long()
        batch['context_input_scores'] = torch.FloatTensor(seq_scores) #
        dec_in_pt = torch.LongTensor(dec_in)
        batch['dec_in'] = dec_in_pt
        batch['dec_attn_mask'] = (dec_in_pt != self.tokenizer.pad_token_id).long()
        batch['dec_out'] = torch.LongTensor(dec_out)
        batch['title'] = [item['title'] for item in samples]
        batch['encoder_input_str'] = [item['encoder_branch_toks'] for item in samples]
        return batch


def get_wiki_surface(ent_name):
    if '(' in ent_name:
        ent_name = ent_name[:ent_name.index('(')]
    ent_name = ent_name.replace("_", " ")
    return ent_name.strip()


