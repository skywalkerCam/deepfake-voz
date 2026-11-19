"""RNN (família recorrente) — A IMPLEMENTAR (integrante 2).

Ideia: tratar a locução como uma sequência de vetores de features (um por
frame) e modelar dependências temporais (prosódia, transições entre fonemas).

Contrato (ver src/models/__init__.py):
    entrada x: (batch, n_features, n_frames)
    saída:     (batch,) logits, positivo = spoof

Sugestões (não obrigatórias):
    - Reorganizar para (batch, n_frames, n_features) antes da LSTM/GRU
      (nn.LSTM com batch_first=True).
    - LSTM ou GRU, possivelmente bidirecional (hparams em config.yaml -> models.rnn).
    - Como resumir a sequência em um vetor: último estado oculto, média
      temporal, ou attention pooling. Vale comparar e justificar no relatório.
    - Uma camada Linear final -> 1 logit.
    - 400 frames é uma sequência longa para RNN: se o treino ficar lento ou
      instável, considere subamostrar no tempo ou usar gradient clipping.

Teste rápido do contrato:
    python -m scripts.check_model --model rnn
"""
import torch
import torch.nn as nn


class RNNClassifier(nn.Module):
    def __init__(self, n_features: int, n_frames: int,
                 hidden_size: int = 128, num_layers: int = 2,
                 bidirectional: bool = True, dropout: float = 0.3):
        super().__init__()
        # TODO: definir camadas
        raise NotImplementedError("RNNClassifier ainda não implementado (integrante 2)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: x (B, F, T) -> logits (B,)
        raise NotImplementedError
