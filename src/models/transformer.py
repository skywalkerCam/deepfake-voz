"""Transformer encoder (família de atenção) — A IMPLEMENTAR (integrante 3).

Ideia: self-attention relaciona qualquer par de frames diretamente, sem o
processamento passo a passo da RNN nem o campo receptivo local da CNN.
Serve como terceira família para contrastar com CNN (local) e RNN (sequencial).

Contrato (ver src/models/__init__.py):
    entrada x: (batch, n_features, n_frames)
    saída:     (batch,) logits, positivo = spoof

Sugestões (não obrigatórias):
    - Projeção linear n_features -> d_model por frame.
    - Codificação posicional (senoidal ou aprendida, tamanho n_frames).
    - nn.TransformerEncoder (batch_first=True); hparams em config.yaml -> models.transformer.
    - Agregação: token [CLS] aprendido ou média temporal.
    - 400 frames x atenção O(T²) é ok, mas reduzir T (ex.: conv/pooling
      inicial com stride) acelera bastante.
    - Uma camada Linear final -> 1 logit.

Teste rápido do contrato:
    python -m scripts.check_model --model transformer
"""
import torch
import torch.nn as nn


class TransformerClassifier(nn.Module):
    def __init__(self, n_features: int, n_frames: int,
                 d_model: int = 128, n_heads: int = 4, num_layers: int = 3,
                 dim_feedforward: int = 256, dropout: float = 0.1):
        super().__init__()
        # TODO: definir camadas
        raise NotImplementedError("TransformerClassifier ainda não implementado (integrante 3)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: x (B, F, T) -> logits (B,)
        raise NotImplementedError
