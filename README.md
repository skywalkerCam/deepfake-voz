# Detecção de deepfakes de voz: CNN vs RNN vs Transformer

Comparação de três famílias de redes neurais na detecção de áudio sintético,
usando um subconjunto do **ASVspoof 2019 LA**.

| Modelo        | Família       | Arquivo                     | Responsável  | Status          |
|---------------|---------------|-----------------------------|--------------|-----------------|
| `cnn`         | Convolucional | `src/models/cnn.py`         | Camilly      | ✅ pronto        |
| `rnn`         | Recorrente    | `src/models/rnn.py`         | Victor       | ✅ pronto        |
| `transformer` | Atenção       | `src/models/transformer.py` | Integrante 3 | ⬜ a fazer       |

## Como rodar (Windows)

**1. Clone o repositório e abra a pasta no VS Code**
```powershell
git clone https://github.com/skywalkerCam/deepfake-voz.git
cd deepfake-voz
code .
```

**2. Baixe o dataset**
Baixe o **LA.zip** em https://datashare.ed.ac.uk/handle/10283/3336 (~7 GB),
extraia e coloque a pasta `LA` dentro de `datasets/`. Tem que ficar assim:
```
datasets/LA/ASVspoof2019_LA_cm_protocols/
datasets/LA/ASVspoof2019_LA_train/
datasets/LA/ASVspoof2019_LA_dev/
datasets/LA/ASVspoof2019_LA_eval/
```
⚠️ Se ficar `datasets/LA/LA/...`, mova a pasta interna um nível para cima.

**3. Instale tudo** (no terminal do VS Code, na pasta do projeto)
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
Get-ChildItem -Recurse | Unblock-File
.\setup.ps1
```
O script cria o ambiente `.venv`, instala o PyTorch (com GPU, se houver NVIDIA) e
testa a CNN. Depois: **Ctrl+Shift+P → Python: Select Interpreter → `.venv`**.

**4. Treine**
```powershell
python -m src.train --model cnn
```

Nas próximas vezes, só ative o ambiente antes: `.venv\Scripts\activate`

> ❗ **Não rode `make_subset`**: o subconjunto já está em `data/subset/` e
> precisa ser o mesmo para todos.

## Como implementar o seu modelo

1. Abra o seu arquivo em `src/models/`: o comentário no topo tem sugestões.
2. Regra: o modelo recebe `x` com shape `(batch, 60, 401)` e devolve `(batch,)`
   com logits (sem sigmoid).
3. Teste sem treinar: `python -m scripts.check_model --model rnn`
4. Treine: `python -m src.train --model rnn`
5. Compare com os outros: `python -m src.compare`

Não altere o `config.yaml` (subset, features, treino) sem combinar com a equipe.
Os hiperparâmetros do seu modelo ficam em `models.<nome>` no `config.yaml`.

## Enviando seu trabalho

```powershell
git pull
git add src/models/rnn.py results/rnn config.yaml
git commit -m "Implementa RNN"
git push
```

## Resultados (eval, 4000 áudios, 13 ataques não vistos no treino)

| Modelo | Precision | Recall | F1     | EER    |
|--------|-----------|--------|--------|--------|
| CNN    | 0.9889    | 0.7585 | 0.8585 | 10.05% |
| RNN    | 0.9837    | 0.7225 | 0.8331 | 8.60%  |
| Transformer | –    | –      | –      | –      |

## Problemas comuns

| Erro | Solução |
|---|---|
| `execução de scripts desabilitada` / `não está assinado` | Rode os 2 primeiros comandos do passo 3 |
| `No module named 'torch'` | Ative o ambiente: `.venv\Scripts\activate` |
| `No module named src` | Rode da pasta do projeto e com `python -m` |
| `Dataset não encontrado` | Confira a estrutura do passo 2 |
| Treino trava no começo | Crie `config.local.yaml` com `train: {num_workers: 0}` |
| `CUDA out of memory` | Feche outros programas; se persistir, avise a equipe |

## Protocolo

Subset aleatório (seed 42) balanceado: treino 4000, dev 2000, eval 4000 áudios.
Features LFCC + Δ + ΔΔ (60 × 401, 4 s). O limiar é escolhido pelo EER no dev e
aplicado no eval. Classe positiva = spoof. Dataset: ASVspoof 2019 (licença ODC-By).