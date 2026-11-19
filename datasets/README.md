# datasets/

Esta pasta fica **fora do git** (só este README é versionado): o ASVspoof 2019 LA
tem ~7 GB, acima do que o GitHub aceita.

Cada integrante baixa o **LA.zip** e extrai aqui. A estrutura deve ficar:

```
datasets/
  LA/
    ASVspoof2019_LA_cm_protocols/
    ASVspoof2019_LA_train/flac/
    ASVspoof2019_LA_dev/flac/
    ASVspoof2019_LA_eval/flac/
```

Atenção: alguns descompactadores criam `datasets/LA/LA/...`. Nesse caso, mova a
pasta interna um nível para cima.

Dataset: ASVspoof 2019 (Edinburgh DataShare), licença ODC-By: cite o artigo
original no relatório.
