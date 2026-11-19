# Instalação do ambiente no Windows (PowerShell).
# Uso, na pasta do projeto:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned   # só na primeira vez
#   .\setup.ps1          # detecta GPU NVIDIA automaticamente
#   .\setup.ps1 -Cpu     # força a versão sem GPU
param([switch]$Cpu)
$ErrorActionPreference = "Stop"

if (-not (Test-Path "requirements.txt")) {
    Write-Error "Rode este script dentro da pasta do projeto (onde está o requirements.txt)."
}

if (-not (Test-Path ".venv")) {
    Write-Host "Criando ambiente virtual .venv ..."
    python -m venv .venv
}
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip -q

$hasGpu = $false
if (-not $Cpu) {
    try { nvidia-smi | Out-Null; $hasGpu = $true } catch { $hasGpu = $false }
}

if ($hasGpu) {
    Write-Host "GPU NVIDIA detectada: instalando PyTorch com CUDA ..."
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
} else {
    Write-Host "Sem GPU NVIDIA: instalando PyTorch para CPU ..."
    pip install torch torchaudio
}
pip install -r requirements.txt

if (-not (Test-Path "datasets\LA\ASVspoof2019_LA_cm_protocols")) {
    Write-Host "`nAVISO: dataset não encontrado. Extraia o LA.zip em datasets\ (veja datasets\README.md)."
}

python -c "import torch; print('torch', torch.__version__, '| GPU:', torch.cuda.is_available())"
python -m scripts.check_model --model cnn
Write-Host "`nPronto. Ative o ambiente nas próximas vezes com: .venv\Scripts\activate"
