import torch
from collections import OrderedDict
from vllm.model_executor.layers.sampler import SamplerOutput, Sampler
from vllm.sequence import SequenceOutput, CompletionSequenceGroupOutput, Logprob

class DeferredSampler(torch.nn.Module):
    """A custom vLLM sampler optimized for efficient next-token probability calculations.

    This sampler replaces vLLM's default sampling mechanism to optimize for scenarios
    where we only need the next token probabilities without actually sampling tokens.

    Note:
        While this sampler implements vLLM's expected interface, it intentionally
        avoids actual token sampling to optimize for probability calculation use cases.
        It should not be used in scenarios where actual token generation is needed.
    """
    def __init__(self): 
        super().__init__()

    def forward(self, logits, sampling_metadata):
        """Process model logits to create vLLM-compatible sampling outputs.

        This method implements the required vLLM sampler interface but optimizes for
        probability requests.

        Args:
            logits (torch.Tensor): Raw model logits with shape (num_tokens, vocab_size).
            sampling_metadata: vLLM metadata containing sequence grouping information.

        Returns:
            SamplerOutput: A vLLM-compatible output structure containing:
                - Sequence group outputs with lazy probability dictionaries
                - Placeholder values for unused sampling fields
                - No actual sampled tokens (uses dummy token_id=0)

        Note:
            The sampler uses token_id=0 as a placeholder.
        """
        assert logits is not None

        logprobs = logits.log_softmax(dim=-1, dtype=torch.float)

        sample_idx = 0
        sampler_output = []
        for seq_group in sampling_metadata.seq_groups:
            seq_ids = seq_group.seq_ids
            num_parent_seqs = len(seq_ids)
            logprobs_by_seq = logprobs[sample_idx : sample_idx + num_parent_seqs]

            assert len(logprobs_by_seq) == len(seq_ids)
            
            seq_outputs = []
            for (seq_id, seq_logprobs) in zip(seq_ids, logprobs_by_seq):
                seq_outputs.append(
                    SequenceOutput(seq_id, 0, LazyLogprobDict(seq_logprobs))
                )

            sampler_output.append(
                CompletionSequenceGroupOutput(samples=seq_outputs, prompt_logprobs=[])
            )
            
            sample_idx += 1

        sampler_outputs = SamplerOutput(
            outputs=sampler_output,
            sampled_token_probs=None,
            sampled_token_ids=None,
            logprobs=None,
            deferred_sample_results_args=None
        )

        return sampler_outputs


class LazyLogprobDict:
    """An efficient dictionary-like interface required by vLLM's output processing.
    
    vLLM's output processor expects token probabilities to be provided as a dictionary
    mapping token IDs to Logprob objects. However, creating this full dictionary is
    computationally expensive, especially when dealing with large vocabulary sizes
    (often 50k+ tokens).

    This class provides a compatible interface that satisfies vLLM's requirements while
    avoiding the overhead.
    """
    def __init__(self, logprobs):
        self.logprobs = logprobs

    def __getitem__(self, key):
        if 0 <= key < len(self.logprobs):
            return Logprob(self.logprobs[key])
        raise KeyError(key)

    def __contains__(self, key):
        return 0 <= key < len(self.logprobs)

    def __len__(self):
        return len(self.logprobs)

    def items(self):
        return ((i, Logprob(prob)) for i, prob in enumerate(self.logprobs))

    def keys(self):
        return range(len(self.logprobs))

    def values(self):
        return iter(map(Logprob, self.logprobs))

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default