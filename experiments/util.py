from datasets import load_dataset

def get_wikitext():
    return '\n'.join(load_dataset("wikitext", "wikitext-2-raw-v1")['test']['text'])

def token_prefixes(text, tokenizer, prefix=''):
    """ Generate all token prefixes which include `prefix` """
    tokens = tokenizer.encode(prefix + text)
    prefix_len = len(tokenizer.encode(prefix)) if prefix else 0
    for i in range(prefix_len + 1, len(tokens)):
        yield tokens[max(0, i - tokenizer.model_max_length): i]
