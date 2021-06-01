import json
import time
from tqdm import tqdm
import utils
import argparse
from system import TextGenerationTrainer

def generate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str)
    parser.add_argument("--dataset-name", type=str, choices=["cmv", "nyt_opinion"], required=True)
    parser.add_argument("--set-type", type=str, choices=['dev', 'test'], default='dev')
    parser.add_argument("--system-setup", type=str, choices=['oracle', 'system'])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--debug", action="store_true")

    # new decoding experiment for camera-ready
    parser.add_argument("--hard-selection", action="store_true",
                        help="If set to True, use 0/1 hard selection of content items instead of weighted sum.")
    parser.add_argument("--random-selection", action="store_true",
                        help="If set to True, use random selection of content items for each sentecne.")


    parser.add_argument("--max-sent-num", type=int, default=10)
    parser.add_argument("--max-entity-per-sentence", type=int, default=20)
    parser.add_argument("--max-concept-per-sentence", type=int, default=20)
    parser.add_argument("--no-entity", action="store_true")
    parser.add_argument("--no-claim", action="store_true")
    parser.add_argument("--no-concept", action="store_true")

    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-target-len", type=int, default=150)
    parser.add_argument("--min-target-len", type=int, default=5)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=-1)

    args = parser.parse_args()

    ckpt_path, ckpt_id = utils.load_latest_ckpt(args.exp_name, args.epoch)
    print(f'loading epoch {ckpt_id}')
    t0 = time.time()
    system = TextGenerationTrainer.load_from_checkpoint(ckpt_path).cuda()
    print(f"model loaded in {time.time() - t0:.2f} secs")
    system.eval()
    if args.fp16:
        system.half()
    system.batch_size = args.batch_size
    system.debug = args.debug
    system.dataset_name = args.dataset_name  # allow out-of-domain test
    system.max_sent_num = args.max_sent_num
    system.max_entity_per_sentence = args.max_entity_per_sentence
    system.max_concept_per_sentence = args.max_concept_per_sentence
    system.no_entity = args.no_entity
    system.no_claim = args.no_claim
    system.no_concept = args.no_concept

    tokenizer = system.tokenizer

    if args.do_sample:
        fout = open(f'outputs/camera_ready/{args.exp_name}_data={args.dataset_name}.{args.set_type}_epoch={args.epoch}_system={args.system_setup}_sampling-topk={args.top_k}-topp={args.top_p}-temp={args.temperature}.jsonl','w')
    else:
        if args.hard_selection:
            selection_method = 'hard'
        elif args.random_selection:
            selection_method = 'rand'
        else:
            selection_method = 'weighted'
        fout = open(f'outputs/camera_ready/{args.exp_name}_data={args.dataset_name}.{args.set_type}_epoch={args.epoch}_system={args.system_setup}_sel-{selection_method}_greedy.jsonl', 'w')

    dataloader = system.test_dataloader(args.set_type, system_setup=args.system_setup)
    for batch in tqdm(dataloader, total=len(dataloader.dataset) / dataloader.batch_size):

        batch = utils.move_to_cuda(batch)
        real_batch_size = len(batch['id'])

        if not args.quiet:
            branch_in = batch['encoder_input_str']
            for i in range(min(3, len(branch_in))):
                print(f'{i}-th branch inputs:')
                for sent_ix, sent in enumerate(branch_in[i]):
                    print(f'{sent_ix}: {sent}')
                print('\n')

        outputs, scoring_results = system.model.generate(
            real_batch_size=real_batch_size,
            k_sizes=batch['k_sizes'],
            context_input_ids=batch['input_ids'],
            context_attention_mask=batch['input_attn_mask'],
            # context_input_scores=batch['context_input_scores'],
            max_length=args.max_target_len,
            do_sample=args.do_sample,
            top_k=args.top_k,
            top_p=args.top_p,
            tokenizer=tokenizer,
            hard_selection=args.hard_selection,
            rand_selection=args.random_selection,
            quiet=args.quiet,
        )

        for b in range(real_batch_size):
            cur_id = batch['id'][b]
            if isinstance(cur_id, list):
                cur_id = f'{cur_id[0]}_{cur_id[1]}'

            title = batch['title'][b]
            reference = batch['dec_out'][b]
            reference = tokenizer.decode(reference, skip_special_tokens=True)
            cur_scoring_results = scoring_results[b].cpu().tolist()

            output_obj = dict(
                id=cur_id,
                title=title,
                reference=reference,
                branch_in=batch['encoder_input_str'][b],
                scores=cur_scoring_results,
            )
            cur_idx = batch['k_sizes'][:b].sum().item()
            cur_output_toks = outputs[cur_idx]
            cur_output_toks = cur_output_toks[cur_output_toks != tokenizer.pad_token_id]
            output_obj['output_toks'] = tokenizer.convert_ids_to_tokens(cur_output_toks)
            cur_output_toks = tokenizer.decode(cur_output_toks, skip_special_tokens=True)
            output_obj['output'] = cur_output_toks
            fout.write(json.dumps(output_obj) + "\n")
    fout.close()

if __name__=='__main__':
    generate()
