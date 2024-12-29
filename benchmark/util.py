from datasets import load_dataset

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