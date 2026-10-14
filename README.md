# 🎭 FaceLab

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.14 CUDA 12.6](https://img.shields.io/badge/PyTorch-2.14%20|%20CUDA%2012.6-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX Runtime GPU](https://img.shields.io/badge/ONNX%20Runtime-GPU%201.23.2-005CED.svg)](https://onnxruntime.ai/)
[![PySide6](https://img.shields.io/badge/GUI-PySide6%20(Qt6)-41CD52.svg?logo=qt&logoColor=white)](https://doc.qt.io/qtforpython/)
[![License: Non-Commercial / Educational](https://img.shields.io/badge/License-Educational%20Only-orange.svg)](#-aviso-ético-e-licença)

**FaceLab** é um ecossistema educacional de visão computacional e inteligência artificial para **substituição facial (Face Swap) e restauração em alta definição (Super-Resolution)**. O software suporta processamento de **vídeos gravados (offline)** e **transmissão ao vivo por webcam (DroidCam, OBS Virtual Cam e câmeras USB)**.

Construído do zero com foco em estabilização temporal anti-flicker, aceleração por hardware NVIDIA CUDA e interface desktop reativa.

---

## ⚡ Destaques do Projeto

- 🎬 **Processamento de Vídeos Offline:** Substitui rostos preservando áudio original, taxa de quadros exata e metadados de identificação sintética via FFmpeg.
- 🎥 **Modo Webcam ao Vivo (Real-time Live):** Integração direta com câmeras físicas e virtuais (DroidCam / OBS) com renderização fluida na interface gráfica.
- ✨ **Restauração Facial com GFPGAN v1.4:** Super-resolução e reconstituição de textura de pele, olhos e dentes, superando as limitações de resolução de modelos padrão (128x128).
- 🛡️ **Estabilização Temporal Avançada (Anti-Jitter & Deflicker):** Filtro de Média Móvel Exponencial (EMA) nos marcos faciais e nas estatísticas de cor/iluminação para eliminar tremulações e oscilações bruscas.
- 🎛️ **Controles de Ajuste Fino na Interface:**
  - Slider de Intensidade do Face Swap (0% a 100%).
  - Suavização de Borda Anatômica (*Feathering* adaptativo).
  - Balanço de Iluminação e Cor.
- 💻 **Arquitetura Modular:** Separação estrita entre detecção (`SCRFD`), extração de identidade (`ArcFace`), inferência (`INSwapper`), restauração (`GFPGAN`) e composição em camadas.

---

## 🖥️ Requisitos de Sistema

### Mínimos (Modo CPU):
- **Sistema Operacional:** Windows 10/11 (64-bit), Linux ou macOS.
- **Processador:** Intel Core i5 / AMD Ryzen 5 (quad-core ou superior).
- **Memória RAM:** 8 GB.
- **Armazenamento:** 3 GB de espaço livre para código, dependências e pesos dos modelos.

### Recomendados (Aceleração por GPU NVIDIA):
- **GPU:** NVIDIA GeForce GTX 1060 / 1070 (Pascal `sm_61`) ou superior (RTX 20xx, 30xx, 40xx).
- **VRAM:** Mínimo de 4 GB dedicados (6 GB a 8 GB recomendados para GFPGAN).
- **Driver NVIDIA:** Versão recente com suporte a CUDA 12.x ou superior (Driver 530+).

---

## 🚀 Guia de Instalação Rápida (Windows / PowerShell)

Siga o passo a passo intuitivo abaixo para configurar o ambiente em qualquer computador:

### 1. Clonar o Repositório
```powershell
git clone https://github.com/SEU_USUARIO/FaceLab.git
cd FaceLab
```

### 2. Criar e Ativar o Ambiente Virtual (Python 3.11 recomendado)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```
> *Dica: Se o PowerShell bloquear a ativação de scripts, execute uma vez como Administrador:* `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

### 3. Instalar PyTorch com Aceleração CUDA
Instale a versão do PyTorch compilada com suporte a CUDA:
```powershell
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### 4. Instalar as Dependências do Projeto
```powershell
pip install -r requirements.txt
```

### 5. Baixar os Modelos Neurais Pré-treinados
O FaceLab possui um script que faz o download automático de todos os modelos necessários (`inswapper_128`, `gfpgan_1.4` e `buffalo_l`):
```powershell
python scripts/download_models.py
```

### 6. Instalar o FFmpeg (para áudio e vídeo)
O FaceLab busca o FFmpeg tanto no `PATH` do sistema quanto na pasta `tools/ffmpeg/bin/`.
- No Windows via Winget:
  ```powershell
  winget install Gyan.FFmpeg
  ```
*(Reinicie o terminal após a instalação).*

### 7. Validar o Ambiente
Execute o script de diagnóstico automatizado para checar se todos os componentes estão prontos:
```powershell
python scripts/check_env.py
```
Você deverá ver todos os itens marcados como `[OK]`, incluindo a detecção da GPU CUDA.

---

## 🎮 Como Usar

### 1. Interface Gráfica Completa (Recomendado)
Para abrir o aplicativo desktop com PySide6:
```powershell
python app.py
```
- **Aba "Vídeo Gravado":** Selecione um vídeo de entrada, escolha o rosto de referência, ajuste a intensidade e a nitidez (GFPGAN) e clique em **Iniciar Face Swap**.
- **Aba "Webcam ao Vivo":** Selecione a câmera (DroidCam Client, OBS Virtual Cam ou Webcam USB), selecione a foto de destino e clique em **Iniciar Câmera**.

### 2. Linha de Comando (CLI) — Processamento em Lote
Para processar arquivos de vídeo sem abrir a interface gráfica:
```powershell
# Básico (Modo rápido)
python scripts/run_swap.py "caminho/do/video.mp4" "caminho/do/rosto.png"

# Com Restauração Facial (GFPGAN) e saída customizada
python scripts/run_swap.py "caminho/do/video.mp4" "caminho/do/rosto.png" --enhance --saida "output/resultado_hd.mp4"
```

### 3. Modo Webcam via Terminal
```powershell
python scripts/run_webcam.py "caminho/do/rosto.png" --camera 0
```

---

## 🧪 Suíte de Testes Automatizados

O FaceLab inclui uma suíte completa de testes unitários para validar dispositivos, decodificadores de vídeo, pipeline de detecção e matrizes afins:

```powershell
python -m pytest -v
```

---

## 🏗️ Estrutura do Projeto

```text
FaceLab/
├── app.py                      # Ponto de entrada da Interface Gráfica (PySide6)
├── requirements.txt            # Dependências Python
├── config/
│   └── settings.py             # Configurações globais e resolução de binários
├── src/
│   ├── core/                   # Protocolos abstratos e tipos de dados (dataclasses)
│   ├── face/                   # Detecção, Alinhamento, INSwapper, GFPGAN e Suavização
│   ├── inference/              # Gerenciador de hardware (DeviceManager) e carregador ONNX
│   ├── interface/              # Janela PySide6 (MainWindow) e background workers (QThread)
│   ├── pipeline/               # Pipelines orquestradores (VideoPipeline e LivePipeline)
│   └── video/                  # Leitor de vídeo (FrameSource), gravador (VideoWriter) e Webcam
├── scripts/
│   ├── check_env.py            # Diagnóstico de hardware e drivers
│   ├── download_models.py      # Download automatizado de pesos ONNX
│   ├── run_swap.py             # Interface de linha de comando (CLI)
│   └── run_webcam.py           # Execução direta de webcam
└── tests/                      # Testes automatizados com pytest
```

---

## ⚖️ Aviso Ético e Licença

Este projeto é desenvolvido para **fins estritamente educacionais e de pesquisa científica** em visão computacional e inteligência artificial.

1. **Uso Responsável:** Utilize apenas imagens e vídeos de sua autoria ou com o consentimento explícito e informado dos envolvidos.
2. **Identificação de Conteúdo:** Todos os vídeos sintetizados pelo FaceLab recebem metadados indicando geração por inteligência artificial.
3. **Licença dos Modelos:** Os pesos neurais utilizados (`buffalo_l` e `inswapper_128`) pertencem aos respectivos autores e foram disponibilizados originalmente para uso acadêmico e não comercial.
