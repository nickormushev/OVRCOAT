from typing import List, Tuple

import open_clip as clip
import torch
from torch import nn

class LearnablePromptExtractor(nn.Module):
    def __init__(self, prompt_dim: int, prompt_shape: Tuple[int, int]):
        super().__init__()
        self.prompt_dim = prompt_dim
        self.prompt_shape = prompt_shape
        self.prefix_prompt = self._init_prompt(self.n_prefix)
        self.suffix_prompt = self._init_prompt(self.n_suffix)
        self._buffer_init = False
        self.with_trainable_params = True
        self.noun_bucket = {}

    def _init_prompt(self, length):
        if length == 0:
            return None
        prompt_tensor = torch.empty(length, self.prompt_dim)
        nn.init.normal_(prompt_tensor, std=0.02)
        return nn.Parameter(prompt_tensor)
    
    def init_buffer(self, clip_model, tokenizer):
        sentence = "X."
        prompt = tokenizer([sentence])
        cast_dtype = clip_model.transformer.get_cast_dtype()
        with torch.no_grad():
            # Extract the text features using CLIP's encode_text method
            embedding = clip_model.token_embedding(prompt).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        # Extract components from the embedding
        self.register_buffer("start_signal", embedding[0, :1, :])  # 1,512
        self.register_buffer("dot_signal", embedding[0, 2:3, :])  # 1,512
        self.register_buffer("end_signal", embedding[0, 3:4, :])  # 1,512
        self.register_buffer("pad_signal", embedding[0, 4:5, :])  # 1,512

        self.noun_bucket = {}
        self._buffer_init = True


    def forward(self, noun_list: List[str], clip_model: nn.Module, text_tokenizer: nn.Module):
        if not self._buffer_init:
            raise RuntimeError(
                f"Buffer of {self.__class__.__name__} is not initialized"
            )

        self._update_noun_features(noun_list, clip_model, text_tokenizer)

        prefix = [self.start_signal]
        if self.prefix_prompt is not None:
            prefix.append(self.prefix_prompt)

        prefix = torch.cat(prefix)

        suffix = [self.dot_signal, self.end_signal]
        if self.suffix_prompt is not None:
            suffix.insert(0, self.suffix_prompt)

        suffix = torch.cat(suffix)
        # only process those which are not in bucket
        lengths = [
            len(prefix) + len(suffix) + len(self.noun_bucket[noun])
            for noun in noun_list
        ]
        embeddings = torch.stack(
            [
                torch.cat(
                    [prefix, self.noun_bucket[noun], suffix]
                    + [self.pad_signal.expand(77 - length, -1)]
                )
                for noun, length in zip(noun_list, lengths)
            ]
        )  # cls,77,512
        indices = torch.Tensor(lengths).long().to(embeddings.device) - 1

        return embeddings, indices
    
    def _update_noun_features(self, noun_list, clip_model, text_tokenizer):
        left_class_names = [noun for noun in noun_list if noun not in self.noun_bucket]
        
        cast_dtype = clip_model.transformer.get_cast_dtype()
        if len(left_class_names) > 0:
            with torch.no_grad():
                tokens = text_tokenizer(left_class_names)
                eot_token_id = text_tokenizer.eot_token_id
                text_embeddings = clip_model.token_embedding(
                    tokens.to(self.device)
                ).type(cast_dtype)
                eot_indices = (tokens == eot_token_id).nonzero(as_tuple=True)[1]  # Find EOT index for each sequence
                # Slice embeddings correctly
                text_embeddings = [
                    embedding[1:eot_idx.item()]  # Exclude SOT (index 0) and stop before EOT
                    for embedding, eot_idx in zip(text_embeddings, eot_indices)
                ]
            
            self.noun_bucket.update(
                {
                    name: embedding
                    for name, embedding in zip(left_class_names, text_embeddings)
                }
            )



    @property
    def n_prefix(self):
        return self.prompt_shape[0]

    @property
    def n_suffix(self):
        return self.prompt_shape[1]

    @property
    def device(self):
        return self.start_signal.device

    def extra_repr(self) -> str:
        r"""Set the extra representation of the module

        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """

        repr = f"prefix_prompt:{self.n_prefix},suffix_prompt:{self.n_suffix},dimension:{self.prompt_dim}\n"
        repr = repr + "[Normal_Init(mu=0,std=0.02)]"
        return repr