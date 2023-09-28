import os

import pandas as pd
from torch.utils.data import DataLoader, Dataset


def get_dataloaders(tokenizer, args):
    train_loader = DataLoader(NLIDataset(tokenizer, 'train', args),
                              args.batch_size,
                              shuffle=True)
    valid_loader = DataLoader(NLIDataset(tokenizer, 'dev', args),
                              args.eval_batch_size,
                              shuffle=False)
    test_loader = DataLoader(NLIDataset(tokenizer, 'test', args),
                             args.eval_batch_size,
                             shuffle=False)
    return train_loader, valid_loader, test_loader


class NLIDataset(Dataset):

    def __init__(self, tokenizer, mode, args):
        """
        mode \in [train, dev, test]
        """
        self.tokenizer = tokenizer
        self.args = args
        self.mode = mode

        csv_file_name = os.path.join(args.data_root, f'esnli_{mode}')
        if mode == 'train':
            df1 = pd.read_csv(csv_file_name + '_1.csv')
            df2 = pd.read_csv(csv_file_name + '_2.csv')
            self.df = pd.concat([df1, df2])
            if args.train_length > 0:
                n_samples = args.train_length * args.batch_size
                self.df = self.df.sample(n_samples)
        else:
            self.df = pd.read_csv(csv_file_name + '.csv')

        self.df.dropna(inplace=True)
        self.df = self.df[['gold_label', 'Sentence1', 'Sentence2']]

        self.label_map = {
            'neutral': 0,
            'entailment': 1,
            'contradiction': 2,
        }

        self.df['label'] = self.df['gold_label'].apply(self.label_map.get)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        item = self.df.iloc[index]
        label = item['label']
        try:
            inputs = self.tokenizer(
                item['Sentence1'],
                text_pair=item['Sentence2'],
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=self.args.max_length,
            )
        except:
            import pdb
            pdb.set_trace()
        inputs = {k: v[0] for k, v in inputs.items()}
        inputs['label'] = label

        return inputs
