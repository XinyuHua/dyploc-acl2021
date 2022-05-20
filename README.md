### Code for DYPLOC 

This repository contains code for DYPLOC ver-0.1, which is the original code
we used for experiments in our ACL 2021 paper submission. If you find our work useful, please cite:

```bibtex
@inproceedings{hua-etal-2021-dyploc,
    title = "DYPLOC: Dynamic Planning of Content Using Mixed Language Modelsfor Text Generation",
    author = "Hua, Xinyu  and
      Sreevatsa, Ashwin  and
      Wang, Lu",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (ACL-IJCNLP)",
    month = aug,
    year = "2021",
    publisher = "Association for Computational Linguistics",
}
```

### Requirements

The original project is tested under the following environment:

```
pytorch==1.5.1
pytorch-lightning==1.0.0
transformers==3.2.0
```

### Project Structure

Please first download data from this [link](https://drive.google.com/drive/folders/1BGXJDXRGEoXZuKChjAXQgDcMHrJEJLVu?usp=sharing).

- `trainable/`: datasets ready to be loaded by `dataset.py`
- `checkpoints/`: folders to save checkpoints
- `train.py`: code for training, will save checkpoints to `checkpoints/` and statistics to `tb_logs/`.
- `generate.py`: code for inference, will save output to `outputs/`.
- `system.py`: code for implementation of the training, validation, data loading, optimization routines.
- `dataset.py`: code for datasets
- `dyploc.py`: code for model implementation, including decoding specific methods such as nucleus sampling

### To run

Our model is trained using the following command.

```shellscript 
EXP_NAME="demo_cmv"
CUDA_VISIBLE_DEVICES=0 python train.py \
    --exp-name=${EXP_NAME} \
    --dataset-name=cmv \
    --warmup-steps=1000 \
    --max-epochs=5 \
    --batch-size=4 \
    --fp16 \
    --marginalization=tok
```

For decoding:

```shellscript
EXP_NAME="demo_cmv"
CUDA_VISIBLE_DEVICES=0 python generate.py \
    --exp-name=${EXP_NAME} \
    --dataset-name=cmv \
    --set-type=test \
    --system-setup=oracle \
    [--quiet \]
    --epoch=5 \
    --batch-size=64 \
    --fp16
```
