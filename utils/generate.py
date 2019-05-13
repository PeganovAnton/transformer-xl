import argparse
import math
import os

import torch
import torch.nn.functional as F
import tqdm

from pytorch_pretrained_bert import GPT2Tokenizer


def hidden_to_softmax(model, hidden):
    """Turn a hidden projection into log softmax.
    
    Adapted from utils/proj_adaptive_softmax.py
    """
    self = model.crit
    logit = self._compute_logit(hidden, self.out_layers[0].weight,
                                            self.out_layers[0].bias, self.out_projs[0])
    softmax = F.softmax(logit, dim=-1)
    return softmax
    
def main():
    parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
    parser.add_argument('--work_dir', type=str, required=True,
                        help='path to the work_dir')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the best saved model.
    with open(os.path.join(args.work_dir, 'model-best.pt'), 'rb') as f:
        model = torch.load(f)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    NL = tokenizer.encode('\n')

    model = model.to(device)
    model.eval()

    data = torch.tensor(NL*70)
    # Turn into a batch
    data.unsqueeze_(1)
    mems = model.init_mems()

    tgt_len = data.size(0)
    hidden, new_mems = model._forward(data, mems=mems)
    pred_hid = hidden[-tgt_len:]
    softmax = hidden_to_softmax(model, pred_hid)


if __name__ == '__main__':
    main()
