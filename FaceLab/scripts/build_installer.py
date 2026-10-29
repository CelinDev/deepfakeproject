"""Compila o Instalador Único `FaceLab-Setup-v1.0.0.exe` e publica no GitHub Releases.

Passos executados:
1. Garante que `7zr.exe` esteja disponível em `FaceLab/tools/7z/7zr.exe`.
2. Compila `FaceLab/scripts/installer_stub.py` em um executável GUI leve (`FaceLabSetupStub.exe`) usando PyInstaller `--onefile --windowed` com `7zr.exe` embutido.
3. Compacta toda a pasta `FaceLab/dist/FaceLab/*` em `payload.7z` usando compressão LZMA2.
4. Concatena `FaceLabSetupStub.exe` + `PAYLOAD_MARKER` + `payload.7z` gerando o arquivo único:
   `FaceLab/dist/release_assets/FaceLab-Setup-v1.0.0.exe`.
5. Faz o upload de `FaceLab-Setup-v1.0.0.exe` para a Release `v1.0.0` no GitHub e remove os arquivos divididos antigos (`.zip.001` e `.zip.002`).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

FACELAB_DIR = Path(__file__).resolve().parent.parent
DIST_DIR = FACELAB_DIR / "dist" / "FaceLab"
RELEASE_DIR = FACELAB_DIR / "dist" / "release_assets"
TOOLS_7Z_DIR = FACELAB_DIR / "tools" / "7z"
SEVEN_ZIP_EXE = TOOLS_7Z_DIR / "7zr.exe"
STUB_SCRIPT = FACELAB_DIR / "scripts" / "installer_stub.py"

REPO = "CelinDev/deepfakeproject"
TAG = "v1.0.0"
PAYLOAD_MARKER = b"FACELAB_7Z_PAYLOAD_V1\x00"


def ensure_7zr() -> Path:
    TOOLS_7Z_DIR.mkdir(parents=True, exist_ok=True)
    if SEVEN_ZIP_EXE.is_file() and SEVEN_ZIP_EXE.stat().st_size > 100_000:
        return SEVEN_ZIP_EXE
    print("Baixando utilitário 7zr.exe oficial...")
    urllib.request.urlretrieve("https://www.7-zip.org/a/7zr.exe", SEVEN_ZIP_EXE)
    return SEVEN_ZIP_EXE


def build_stub_exe(seven_zip: Path) -> Path:
    print("=== 1/4 Compilando interface gráfica do Instalador (Stub) ===")
    stub_build = FACELAB_DIR / "build" / "installer_stub"
    stub_dist = FACELAB_DIR / "dist" / "installer_stub"
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--windowed",
        "--name",
        "FaceLabSetupStub",
        "--workpath",
        str(stub_build),
        "--distpath",
        str(stub_dist),
        "--add-binary",
        f"{seven_zip};.",
        "--exclude-module",
        "torch",
        "--exclude-module",
        "cv2",
        "--exclude-module",
        "numpy",
        "--exclude-module",
        "scipy",
        "--exclude-module",
        "onnxruntime",
        "--exclude-module",
        "PySide6",
        str(STUB_SCRIPT),
    ]
    subprocess.run(cmd, cwd=str(FACELAB_DIR), check=True)
    stub_exe = stub_dist / "FaceLabSetupStub.exe"
    if not stub_exe.is_file():
        raise RuntimeError(f"Stub não encontrado em {stub_exe}")
    print(f"Stub compilado: {stub_exe} ({stub_exe.stat().st_size / (1024**2):.1f} MB)")
    return stub_exe


def compress_payload_7z(seven_zip: Path) -> Path:
    print("=== 2/4 Compactando distribuição com LZMA2 (7-Zip) ===")
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    payload_7z = RELEASE_DIR / "payload.7z"
    if payload_7z.exists():
        payload_7z.unlink()

    # Compacta o conteúdo de DIST_DIR com LZMA2 rápido e multithread
    cmd = [
        str(seven_zip),
        "a",
        "-t7z",
        "-m0=lzma2",
        "-mx=5",
        "-mmt=on",
        str(payload_7z),
        f"{DIST_DIR}\\*",
    ]
    subprocess.run(cmd, check=True)
    size_mb = payload_7z.stat().st_size / (1024**2)
    print(f"Arquivo payload.7z criado: {size_mb:.1f} MB")
    return payload_7z


def assemble_single_installer(stub_exe: Path, payload_7z: Path) -> Path:
    print("=== 3/4 Montando instalador em arquivo único (FaceLab-Setup-v1.0.0.exe) ===")
    installer_exe = RELEASE_DIR / f"FaceLab-Setup-{TAG}.exe"
    if installer_exe.exists():
        installer_exe.unlink()

    with open(installer_exe, "wb") as out_f:
        with open(stub_exe, "rb") as stub_f:
            shutil.copyfileobj(stub_f, out_f, length=8 * 1024 * 1024)
        out_f.write(PAYLOAD_MARKER)
        with open(payload_7z, "rb") as pay_f:
            shutil.copyfileobj(pay_f, out_f, length=16 * 1024 * 1024)

    payload_7z.unlink()
    total_mb = installer_exe.stat().st_size / (1024**2)
    print(f"[OK] Instalador único gerado em: {installer_exe} ({total_mb:.1f} MB)")
    return installer_exe


def get_github_token() -> str:
    p = subprocess.Popen(
        ["git", "credential", "fill"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, _ = p.communicate("protocol=https\nhost=github.com\n\n")
    creds = dict(line.split("=", 1) for line in out.strip().splitlines() if "=" in line)
    token = creds.get("password", "")
    if not token:
        raise RuntimeError("Token do GitHub não encontrado no git credential manager.")
    return token


def upload_single_installer_to_github(installer_exe: Path) -> str:
    print("=== 4/4 Atualizando Release v1.0.0 no GitHub com o Instalador Único ===")
    token = get_github_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "FaceLab-Release-Publisher",
    }

    req_get = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/releases/tags/{TAG}",
        headers=headers,
    )
    with urllib.request.urlopen(req_get) as resp:
        release = json.loads(resp.read().decode())

    # Atualiza a descrição da release para destacar o instalador único
    body_notes = (
        "## ⚡ FaceLab Studio v1.0.0 — Instalador Completo (Windows x64)\n\n"
        "### ✨ Destaques desta versão\n"
        "- **Instalador em Arquivo Único (`FaceLab-Setup-v1.0.0.exe`)**: instala o FaceLab Studio completo, cria atalhos na Área de Trabalho e no Menu Iniciar, e não requer instalação prévia de Python, CUDA ou FFmpeg.\n"
        "- **Download Obrigatório de Modelos no 1º Uso**: pacote enxuto sem a pasta pesada `models/`. "
        "Ao abrir o FaceLab Studio pela primeira vez, uma janela interativa solicita e baixa automaticamente "
        "todos os modelos necessários (`inswapper_128`, `gfpgan_1.4`, `bisenet_resnet_34` e `buffalo_l`).\n"
        "- **Presets Cinematográficos em 8K**: Zumbi Hiper-realista, Vampiro Nobre e Ciborgue Futurista.\n"
        "- **Segmentação Semântica (BiSeNet)** para preservação orgânica de cabelos e orelhas.\n"
        "- **Super-Resolução e Aceleração por GPU NVIDIA (CUDA 12 / cuDNN 9)** com fallback automático para CPU.\n\n"
        "### 🚀 Como instalar e usar\n"
        "1. Baixe e execute o arquivo único **`FaceLab-Setup-v1.0.0.exe`** abaixo.\n"
        "2. Clique em **Instalar Agora** (ele criará o atalho na sua Área de Trabalho e abrirá o programa).\n"
        "3. Na primeira execução, clique em **Baixar Modelos** na janela inicial e aguarde a conclusão."
    )
    patch_payload = json.dumps({
        "name": "FaceLab Studio v1.0.0 — Instalador (Windows x64)",
        "body": body_notes,
    }).encode("utf-8")
    req_patch = urllib.request.Request(
        release["url"],
        data=patch_payload,
        headers={**headers, "Content-Type": "application/json"},
        method="PATCH",
    )
    urllib.request.urlopen(req_patch).close()

    # Remove assets antigos (.zip.001, .zip.002 ou instalador anterior)
    for asset in release.get("assets", []):
        del_req = urllib.request.Request(asset["url"], headers=headers, method="DELETE")
        urllib.request.urlopen(del_req).close()
        print(f"  Removido asset antigo da release: {asset['name']}")

    upload_url_template = release["upload_url"].split("{")[0]
    url = f"{upload_url_template}?name={urllib.parse.quote(installer_exe.name)}"
    size = installer_exe.stat().st_size
    print(f"Enviando {installer_exe.name} ({size / (1024**2):.1f} MB) para o GitHub Releases...")

    with open(installer_exe, "rb") as f:
        req_up = urllib.request.Request(
            url,
            data=f,
            headers={
                **headers,
                "Content-Type": "application/octet-stream",
                "Content-Length": str(size),
            },
            method="POST",
        )
        with urllib.request.urlopen(req_up, timeout=1800) as resp:
            info = json.loads(resp.read().decode())
            return info["browser_download_url"]


def main() -> int:
    seven_zip = ensure_7zr()
    stub_exe = build_stub_exe(seven_zip)
    payload_7z = compress_payload_7z(seven_zip)
    installer_exe = assemble_single_installer(stub_exe, payload_7z)
    dl_url = upload_single_installer_to_github(installer_exe)
    print(f"\n[SUCESSO] Instalador único publicado em: {dl_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

