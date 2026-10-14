# 📋 Requisitos de Sistema e Guia de Instalação Universal — FaceLab

Este documento contém todas as especificações técnicas, compatibilidades de hardware e um passo a passo detalhado para instalar e executar o **FaceLab** em qualquer computador (Windows, Linux ou macOS).

---

## 💻 1. Especificações de Hardware

O FaceLab foi projetado com uma camada de abstração de hardware (`DeviceManager`) que seleciona automaticamente o melhor mecanismo disponível.

| Componente | Requisitos Mínimos (Modo CPU) | Recomendado (GPU NVIDIA CUDA) |
|---|---|---|
| **Sistema Operacional** | Windows 10/11 (64-bit), Ubuntu 20.04+, macOS 12+ | Windows 10/11 ou Linux Ubuntu |
| **Processador (CPU)** | Intel Core i5 / AMD Ryzen 5 (4 núcleos) | Intel Core i7 / AMD Ryzen 7 (6+ núcleos) |
| **Placa de Vídeo (GPU)** | Nenhuma (renderização via CPU) | NVIDIA GeForce GTX 1060 / 1070 (Pascal) ou superior (RTX 20xx / 30xx / 40xx) |
| **Memória de Vídeo (VRAM)**| — | 4 GB dedicados (6 GB a 8 GB para super-resolução GFPGAN) |
| **Memória RAM** | 8 GB RAM | 16 GB RAM ou superior |
| **Armazenamento** | 3.5 GB de espaço livre | 3.5 GB em SSD rápido |
| **Python** | Python 3.11.x (64-bit) | Python 3.11.x (64-bit) |

---

## 🛠️ 2. Softwares Prévios Necessários

Antes de configurar o código, certifique-se de que os seguintes softwares estão instalados:

1. **Python 3.11:**
   - **Windows:** Baixe no site oficial [python.org](https://www.python.org/downloads/) ou via terminal:
     ```powershell
     winget install Python.Python.3.11
     ```
   - ⚠️ *Atenção:* Na instalação do instalador oficial, marque a caixinha **"Add python.exe to PATH"**.
2. **Driver NVIDIA (apenas se tiver GPU dedicada):**
   - Mantenha os drivers da sua placa atualizados através do [NVIDIA GeForce Experience](https://www.nvidia.com/geforce/geforce-experience/) ou site da NVIDIA.
3. **FFmpeg (leitor e gravador de áudio/vídeo):**
   - **Windows (Winget):**
     ```powershell
     winget install Gyan.FFmpeg
     ```
   - **Linux (Ubuntu/Debian):**
     ```bash
     sudo apt update && sudo apt install ffmpeg
     ```
   - **macOS (Homebrew):**
     ```bash
     brew install ffmpeg
     ```

---

## 🚀 3. Passo a Passo de Instalação (Universal)

### Passo 1: Obter o Código
Abra seu terminal favorito (PowerShell no Windows, ou Bash no Linux/macOS):
```bash
git clone https://github.com/SEU_USUARIO/FaceLab.git
cd FaceLab
```

### Passo 2: Criar e Ativar o Ambiente Virtual Isolado
Recomendamos fortemente o uso de um `.venv` para evitar conflito com outras bibliotecas do sistema.

- **No Windows (PowerShell):**
  ```powershell
  py -3.11 -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
  *(Se aparecer erro de script não autorizado, execute: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` e tente novamente).*

- **No Linux / macOS:**
  ```bash
  python3.11 -m venv .venv
  source .venv/bin/activate
  ```

---

### Passo 3: Instalar o PyTorch

#### Se você possui Placa de Vídeo NVIDIA (Windows / Linux):
Instale o PyTorch compilado com suporte a CUDA 12.6:
```bash
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

#### Se você NÃO possui GPU NVIDIA (Modo CPU / macOS):
```bash
python -m pip install --upgrade pip
pip install torch torchvision
```

---

### Passo 4: Instalar as Dependências do FaceLab
```bash
pip install -r requirements.txt
```

> **Nota sobre ONNX Runtime GPU no Windows:**
> O arquivo `requirements.txt` já fixa o `onnxruntime-gpu==1.23.2`, garantindo compatibilidade total com arquiteturas NVIDIA Pascal (série GTX 10) e CUDA 12.

---

### Passo 5: Download dos Modelos Neurais
O FaceLab utiliza modelos pré-treinados de deep learning. Execute o script automatizado para baixar e descompactar os pesos necessários:
```bash
python scripts/download_models.py
```
Isso fará o download de:
- `inswapper_128.onnx` (~528 MB)
- `gfpgan_1.4.onnx` (~324 MB)
- `buffalo_l` (SCRFD + ArcFace) (~275 MB)

---

## ✅ 4. Verificação e Diagnóstico do Sistema

Para ter certeza absoluta de que tudo está configurado corretamente (Python, PyTorch, CUDA, GPU e FFmpeg), execute:

```bash
python scripts/check_env.py
```

Se todos os itens exibirem `[OK]`, seu ambiente está 100% pronto!

Você também pode rodar os testes unitários automatizados:
```bash
python -m pytest -v
```

---

## 🎯 5. Executando o FaceLab

### Opção A: Interface Gráfica (Desktop PySide6)
```bash
python app.py
```
- Você terá acesso às abas de **Vídeo Gravado** e **Webcam ao Vivo**.
- Suporta ajuste dinâmico de intensidade, suavização de bordas e restauração facial.

### Opção B: Linha de Comando (CLI)
```bash
# Processar vídeo com restauração de alta nitidez (GFPGAN)
python scripts/run_swap.py "meu_video.mp4" "meu_rosto.png" --enhance
```

### Opção C: Webcam ao Vivo Direta
```bash
python scripts/run_webcam.py "meu_rosto.png"
```
*(Funciona perfeitamente com DroidCam Client, OBS Virtual Camera e webcams USB).*

