"""
Embeddings utilities for sentence encoding using Hugging Face models.

Model: sundeepdwd/muril-mlm-newa-finetuned
Reference: https://huggingface.co/sundeepdwd/muril-mlm-newa-finetuned
"""

from typing import List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModel


class SentenceEncoder:
    """
    Wraps a HF encoder model to produce sentence embeddings via mean pooling.
    """

    def __init__(
        self,
        model_name: str = "sundeepdwd/muril-mlm-newa-finetuned",
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, sentences: List[str], batch_size: int = 16) -> torch.Tensor:
        """
        Encode a list of sentences into embeddings using mean pooling.
        Returns a tensor of shape [num_sentences, hidden_size].
        """
        all_embeddings: List[torch.Tensor] = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state  # [B, T, H]
            attention_mask = inputs["attention_mask"].unsqueeze(-1)  # [B, T, 1]

            masked = last_hidden_state * attention_mask  # zero out pads
            sum_embeddings = masked.sum(dim=1)  # [B, H]
            lengths = attention_mask.sum(dim=1).clamp_min(1)  # [B, 1]
            mean_pooled = sum_embeddings / lengths

            all_embeddings.append(mean_pooled.detach().cpu())

        return torch.cat(all_embeddings, dim=0)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity between two batches of vectors.
    a: [N, D], b: [M, D] -> returns [N, M]
    """
    a_norm = torch.nn.functional.normalize(a, dim=-1)
    b_norm = torch.nn.functional.normalize(b, dim=-1)
    return a_norm @ b_norm.T


def top_k_similar(
    query_embeddings: torch.Tensor,
    corpus_embeddings: torch.Tensor,
    k: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return top-k (indices, scores) for each query against the corpus.
    indices: [Q, k], scores: [Q, k]
    """
    sims = cosine_similarity(query_embeddings, corpus_embeddings)
    scores, indices = torch.topk(sims, k=min(k, corpus_embeddings.shape[0]), dim=-1)
    return indices, scores
