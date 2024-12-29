import asyncio
from datasets import load_dataset

def get_wikitext():
    return '\n'.join(load_dataset("wikitext", "wikitext-2-raw-v1")['test']['text'])

def run_await_next_token_logprobs(benchmark, llm, sequences, rounds=200, warmup_rounds=10):
    loop = asyncio.new_event_loop()

    async def run():
        token_ids = next(sequences)
        assert token_ids
        await llm.next_token_logprobs(token_ids)

    benchmark.pedantic(
        lambda: loop.run_until_complete(run()), 
        iterations=1, 
        rounds=rounds, 
        warmup_rounds=warmup_rounds, 
    )

    loop.close()

def run_await_batch_next_token_logprobs(
    benchmark, llm, batches, rounds=20, warmup_rounds=10
):
    loop = asyncio.new_event_loop()

    async def run():
        token_ids = next(batches)
        assert token_ids
        await llm.batch_next_token_logprobs(token_ids)

    benchmark.pedantic(
        lambda: loop.run_until_complete(run()), 
        iterations=1, 
        rounds=rounds, 
        warmup_rounds=warmup_rounds, 
    )

    loop.close()

def token_prefixes(text, tokenizer, prepend=''):
    """ Generates all token prefixes in text and prepends `prepend` """
    tokens = tokenizer.encode(prepend + text)
    prepend_len = len(tokenizer.encode(prepend)) if prepend else 0
    for i in range(prepend_len + 1, len(tokens)):
        yield tokens[max(0, i - tokenizer.model_max_length): i]

def token_prefix_batches(text, tokenizer, batch_size, prepend=''):
    """Gets batches of token prefixes.
    
    Args:
        text: Text to generate prefixes from
        tokenizer: HuggingFace tokenizer
        batch_size: Number of prefixes per batch
        prepend: Optional string to prepend to all sequences
    
    Yields:
        Batches of token prefix sequences
    """
    batch = []
    for prefix_tokens in token_prefixes(text, tokenizer, prepend):
        batch.append(prefix_tokens)
        if len(batch) == batch_size:
            yield batch
            batch = []