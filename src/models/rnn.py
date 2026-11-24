"""RNN (família recorrente) para detecção de deepfakes de voz.

Ideia: tratar a locução como uma SEQUÊNCIA de vetores de features, um por
frame (60 coeficientes LFCC + Δ + ΔΔ a cada 10 ms), e processá-la passo a
passo. O estado oculto carrega informação dos frames anteriores, então o
modelo pode capturar dependências temporais (prosódia, transições entre
fonemas, continuidade da fala) que a CNN, com seu campo receptivo local,
só enxerga de forma indireta.

Para a comparação entre famílias ser justa, o modelo é recorrente "puro":
não há camadas convolucionais antes da LSTM (isso viraria uma CRNN, que mistura
as famílias). A entrada e o protocolo de treino são os mesmos da CNN.

Arquitetura (padrão do config.yaml):
    (B, 60, 401) -> transpõe -> (B, 401, 60)
    -> LSTM bidirecional, 2 camadas, 128 unidades por direção -> (B, 401, 256)
    -> pooling temporal (attention pooling) -> (B, 256)
    -> Dropout -> Linear(256, 64) -> ReLU -> Dropout -> Linear(64, 1) -> logit

O cabeçote de classificação é o mesmo da CNN, para que a diferença fique no
extrator de características (convolucional x recorrente).

Hiperparâmetros (config.yaml -> models.rnn):
    hidden_size    unidades por direção
    num_layers     camadas recorrentes empilhadas
    bidirectional  lê a sequência nos dois sentidos
    dropout        entre camadas recorrentes e no cabeçote
    cell           "lstm" | "gru"
    pooling        "attention" | "mean" | "last"; como resumir a sequência em
                   um vetor (bom ponto de comparação no relatório)
    frame_stack    1 = usa os 401 frames. k > 1 concatena k frames vizinhos
                   em um passo (sequência k vezes menor, sem descartar
                   informação); só use se o treino ficar lento demais.

O train.py já aplica gradient clipping (max_norm=5), que evita a explosão de
gradientes típica de RNN em sequências longas.

Teste rápido do contrato:
    python -m scripts.check_model --model rnn
"""
import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    """Média ponderada dos frames com pesos aprendidos.

    Para cada frame t:  e_t = v^T tanh(W h_t),  alpha = softmax(e),
    saída = sum_t alpha_t h_t.
    Permite que o modelo dê mais peso aos trechos onde os artefatos de síntese
    aparecem, em vez de tratar todos os frames igualmente. Os pesos da última
    chamada ficam em `last_weights` (B, T), úteis para análise no relatório.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.score = nn.Linear(dim, 1, bias=False)
        self.last_weights = None

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, T, D)
        e = self.score(torch.tanh(self.proj(h))).squeeze(-1)  # (B, T)
        alpha = torch.softmax(e, dim=1)                       # (B, T)
        self.last_weights = alpha.detach()
        return torch.bmm(alpha.unsqueeze(1), h).squeeze(1)    # (B, D)


class RNNClassifier(nn.Module):
    CELLS = {"lstm": nn.LSTM, "gru": nn.GRU}
    POOLINGS = ("attention", "mean", "last")

    def __init__(self, n_features: int, n_frames: int,
                 hidden_size: int = 128, num_layers: int = 2,
                 bidirectional: bool = True, dropout: float = 0.3,
                 cell: str = "lstm", pooling: str = "attention",
                 frame_stack: int = 1):
        super().__init__()
        cell = cell.lower()
        if cell not in self.CELLS:
            raise ValueError(f"cell deve ser um de {list(self.CELLS)}, veio '{cell}'")
        if pooling not in self.POOLINGS:
            raise ValueError(f"pooling deve ser um de {self.POOLINGS}, veio '{pooling}'")
        if frame_stack < 1:
            raise ValueError("frame_stack deve ser >= 1")

        self.cell = cell
        self.pooling = pooling
        self.bidirectional = bidirectional
        self.frame_stack = frame_stack

        self.rnn = self.CELLS[cell](
            input_size=n_features * frame_stack,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            # o dropout interno do PyTorch só age ENTRE camadas (avisa se houver só 1)
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)

        self.attn = AttentionPooling(out_dim) if pooling == "attention" else None

        # mesmo cabeçote da CNN
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        if cell == "lstm":
            self._init_forget_bias(hidden_size)

    def _init_forget_bias(self, hidden_size: int) -> None:
        """Bias do forget gate = 1 (0.5 em bias_ih + 0.5 em bias_hh).

        Truque clássico (Jozefowicz et al., 2015): no início do treino a LSTM
        tende a "lembrar" em vez de esquecer, o que ajuda em sequências longas.
        Ordem dos gates no PyTorch: input, forget, cell, output.
        """
        for name, p in self.rnn.named_parameters():
            if "bias" in name:
                with torch.no_grad():
                    p.zero_()
                    p[hidden_size:2 * hidden_size].fill_(0.5)

    def _stack_frames(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, F) -> (B, T // k, F * k), descartando os frames que sobram no fim
        k = self.frame_stack
        b, t, f = x.shape
        t2 = t // k
        return x[:, :t2 * k, :].reshape(b, t2, f * k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F, T) -> (B, T, F): um vetor de features por passo de tempo
        x = x.transpose(1, 2).contiguous()
        if self.frame_stack > 1:
            x = self._stack_frames(x)

        out, state = self.rnn(x)  # out: (B, T, D)
        h_n = state[0] if isinstance(state, tuple) else state  # LSTM devolve (h_n, c_n)

        if self.pooling == "attention":
            z = self.attn(out)
        elif self.pooling == "mean":
            z = out.mean(dim=1)
        else:  # "last": estado final da última camada
            # h_n: (num_layers * num_dirs, B, H). Com bidirecional, junta o
            # final da direção direta (viu até t=T) e da reversa (viu até t=0).
            z = torch.cat([h_n[-2], h_n[-1]], dim=-1) if self.bidirectional else h_n[-1]

        return self.head(z).squeeze(-1)  # (B,) logits; positivo = spoof