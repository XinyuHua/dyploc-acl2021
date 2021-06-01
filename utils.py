import torch
import os
import glob

def load_latest_ckpt(exp_name, epoch_id=-1):
    def get_epoch_num(path):
        base = os.path.basename(path)
        base = base.split('-')[0][6:]
        return int(base)

    if epoch_id == -1:
        all_ckpts = glob.glob(f"checkpoints/{exp_name}/epoch*")
    else:
        all_ckpts = glob.glob(f"checkpoints/{exp_name}/epoch={epoch_id}-*")

    ckpt_list = sorted(
        all_ckpts,
        key=lambda x: get_epoch_num(x),
        reverse=True
    )
    assert len(ckpt_list) > 0, f"no checkpoint found for {exp_name}, epoch={epoch_id}"
    return ckpt_list[0], get_epoch_num(ckpt_list[0])


def move_to_cuda(sample):

    def _move_to_cuda(tensor):
        return tensor.cuda()

    return apply_to_sample(_move_to_cuda, sample)

def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            r = {key: _apply(value) for key, value in x.items()}
            return r
            # return {
            #     key: _apply(value)
            #     for key, value in x.items()
            # }
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)