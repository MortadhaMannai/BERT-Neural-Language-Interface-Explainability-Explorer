import argparse

import torch
from transformers import AutoTokenizer

from dataloader import get_dataloaders
from explain_evaluator import VerificationNetwork
from train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5, required=False)
    parser.add_argument('--batch_size', type=int, default=32, required=False)
    parser.add_argument('--eval_batch_size', type=int, default=64, required=False)
    parser.add_argument('--train_length', type=int, default=-1, required=False)
    parser.add_argument('--lr', type=float, default=5e-5, required=False)
    parser.add_argument('--lrdecay', type=float, default=0.9, required=False)
    parser.add_argument('--reg_strength', type=float, default=4e-4, required=False)
    parser.add_argument('--decaystep', type=int, default=500, required=False)
    parser.add_argument('--evalstep', type=int, default=500, required=False)
    parser.add_argument('--model_name',
                        type=str,
                        default='bert-base-uncased',
                        required=False)
    parser.add_argument('--max_length', type=int, default=64, required=False)
    parser.add_argument('--data_root',
                        type=str,
                        default='data/e-SNLI/dataset',
                        required=False)
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = VerificationNetwork(args.model_name,
                                mask_id=tokenizer.mask_token_id,
                                reg_strength=args.reg_strength).to(device)

    train_loader, valid_loader, test_loader = get_dataloaders(tokenizer, args)

    train((train_loader, valid_loader, test_loader), model, args, device)


if __name__ == '__main__':
    main()
