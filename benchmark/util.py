import os
import pandas as pd
from datasets import load_dataset

def save_comparison_df(out_dir, name, timer1, timer2):
    df1 = timer1.dataframe().rename(columns={'timer': timer1.name})
    df2 = timer2.dataframe().rename(columns={'timer': timer2.name})
    merge_cols1 = [col for col in df1.columns if col != timer1.name]
    merge_cols2 = [col for col in df2.columns if col != timer2.name]
    assert merge_cols1 == merge_cols2, 'timers have different features'
    df = pd.merge(df1, df2, on=merge_cols1)
    df.to_csv(os.path.join(out_dir, name + '.csv'), index=False)

def plot_features(out_dir, name, timer1, timer2):
    features1 = set().union(*(f.keys() for f in timer1.features))
    features2 = set().union(*(f.keys() for f in timer2.features))
    all_features = features1.union(features2)
    
    for feature in all_features: 
        ax = timer1.plot_feature(feature, label=timer1.name)
        ax = timer2.plot_feature(feature, label=timer2.name, ax=ax)
        ax.set_title(f'{name} - {feature}')
        ax.legend()
        ax.figure.savefig(os.path.join(out_dir, f'{name}_{feature}.png'))

def get_wikitext():
    return '\n'.join(load_dataset("wikitext", "wikitext-2-raw-v1")['test']['text'])

def token_prefixes(text, tokenizer, prefix=''):
    """ Generates all token prefixes in text and prepends `prefix` """
    tokens = tokenizer.encode(prefix + text)
    prefix_len = len(tokenizer.encode(prefix)) if prefix else 0
    for i in range(prefix_len + 1, len(tokens)):
        yield tokens[max(0, i - tokenizer.model_max_length): i]

def prefix_batches(text, tokenizer, batch_size, prefix=''):
    """Gets batches of token prefixes.
    
    Args:
        text: Text to generate prefixes from
        tokenizer: HuggingFace tokenizer
        batch_size: Number of prefixes per batch
        prefix: Optional string to prepend to all sequences
    
    Yields:
        Batches of token prefix sequences
    """
    batch = []
    for prefix_tokens in token_prefixes(text, tokenizer, prefix):
        batch.append(prefix_tokens)
        if len(batch) == batch_size:
            yield batch
            batch = []