"""Instalador Gráfico Nativo do FaceLab Studio (Windows x64).

Quando executado como arquivo único (`FaceLab-Setup-v1.0.0.exe`):
1. Localiza o payload `.7z` embutido no próprio arquivo `.exe` (ou ao lado dele).
2. Permite ao usuário escolher a pasta de instalação (padrão: `%LOCALAPPDATA%\\Programs\\FaceLab`)
   e opções de atalho na Área de Trabalho e no Menu Iniciar.
3. Extrai todos os arquivos com barra de progresso em tempo real.
4. Cria os atalhos no Windows e oferece iniciar o `FaceLab.exe` imediatamente.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

PAYLOAD_MARKER = b"FACELAB_7Z_PAYLOAD_V1\x00"
APP_NAME = "FaceLab Studio"
APP_VERSION = "v1.0.0"


def get_bundle_resource(name: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / name


def find_payload_offset(exe_path: Path) -> tuple[int, int] | None:
    """Procura o marcador do arquivo .7z anexado ao final do próprio executável."""
    chunk_size = 4 * 1024 * 1024
    marker_len = len(PAYLOAD_MARKER)
    file_size = exe_path.stat().st_size

    with open(exe_path, "rb") as f:
        pos = 0
        while pos < file_size:
            f.seek(pos)
            data = f.read(chunk_size + marker_len)
            if not data:
                break
            idx = data.find(PAYLOAD_MARKER)
            if idx != -1:
                payload_start = pos + idx + marker_len
                return payload_start, file_size - payload_start
            pos += chunk_size
    return None


def create_windows_shortcut(target_exe: Path, shortcut_path: Path, description: str) -> None:
    """Cria um atalho (.lnk) usando PowerShell WScript.Shell."""
    shortcut_path.parent.mkdir(parents=True, exist_ok=True)
    ps_script = (
        "$ws = New-Object -ComObject WScript.Shell; "
        f"$s = $ws.CreateShortcut('{shortcut_path}'); "
        f"$s.TargetPath = '{target_exe}'; "
        f"$s.WorkingDirectory = '{target_exe.parent}'; "
        f"$s.Description = '{description}'; "
        "$s.Save()"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden", "-Command", ps_script],
        check=False,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )


class InstallerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(f"Instalador — {APP_NAME} {APP_VERSION}")
        self.root.geometry("560x380")
        self.root.resizable(False, False)
        self.root.configure(bg="#0f111a")

        default_dir = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Programs" / "FaceLab"
        self.install_dir_var = tk.StringVar(value=str(default_dir))
        self.desktop_shortcut_var = tk.BooleanVar(value=True)
        self.startmenu_shortcut_var = tk.BooleanVar(value=True)
        self.launch_after_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Pronto para instalar.")

        self._build_ui()

    def _build_ui(self) -> None:
        header = tk.Frame(self.root, bg="#171b26", padx=24, pady=18)
        header.pack(fill="x")

        tk.Label(
            header,
            text=f"⚡ {APP_NAME} {APP_VERSION}",
            font=("Segoe UI", 16, "bold"),
            fg="#f8fafc",
            bg="#171b26",
            anchor="w",
        ).pack(fill="x")

        tk.Label(
            header,
            text="Instalador Completo (Windows x64 — Aceleração GPU CUDA 12 & CPU)",
            font=("Segoe UI", 9),
            fg="#94a3b8",
            bg="#171b26",
            anchor="w",
        ).pack(fill="x", pady=(4, 0))

        body = tk.Frame(self.root, bg="#0f111a", padx=24, pady=16)
        body.pack(fill="both", expand=True)

        tk.Label(
            body,
            text="Pasta de instalação:",
            font=("Segoe UI", 10, "bold"),
            fg="#e2e8f0",
            bg="#0f111a",
            anchor="w",
        ).pack(fill="x")

        dir_row = tk.Frame(body, bg="#0f111a")
        dir_row.pack(fill="x", pady=(6, 14))

        self.dir_entry = tk.Entry(
            dir_row,
            textvariable=self.install_dir_var,
            font=("Segoe UI", 9),
            bg="#1e2433",
            fg="#f8fafc",
            insertbackground="#f8fafc",
            relief="flat",
        )
        self.dir_entry.pack(side="left", fill="x", expand=True, ipady=6, padx=(0, 8))

        self.browse_btn = tk.Button(
            dir_row,
            text="Procurar...",
            command=self._choose_dir,
            font=("Segoe UI", 9),
            bg="#2d3548",
            fg="#f8fafc",
            activebackground="#3b455e",
            activeforeground="#ffffff",
            relief="flat",
            padx=12,
            pady=4,
        )
        self.browse_btn.pack(side="right")

        chk_style = {
            "font": ("Segoe UI", 9),
            "fg": "#cbd5e1",
            "bg": "#0f111a",
            "selectcolor": "#1e2433",
            "activebackground": "#0f111a",
            "activeforeground": "#ffffff",
            "anchor": "w",
        }
        tk.Checkbutton(
            body,
            text="Criar atalho na Área de Trabalho",
            variable=self.desktop_shortcut_var,
            **chk_style,
        ).pack(fill="x", pady=2)

        tk.Checkbutton(
            body,
            text="Criar atalho no Menu Iniciar",
            variable=self.startmenu_shortcut_var,
            **chk_style,
        ).pack(fill="x", pady=2)

        tk.Checkbutton(
            body,
            text="Iniciar o FaceLab Studio ao concluir a instalação",
            variable=self.launch_after_var,
            **chk_style,
        ).pack(fill="x", pady=2)

        self.progress = ttk.Progressbar(body, mode="determinate", maximum=100)
        self.progress.pack(fill="x", pady=(14, 6))

        tk.Label(
            body,
            textvariable=self.status_var,
            font=("Segoe UI", 9),
            fg="#94a3b8",
            bg="#0f111a",
            anchor="w",
        ).pack(fill="x")

        footer = tk.Frame(self.root, bg="#0f111a", padx=24, pady=12)
        footer.pack(fill="x", side="bottom")

        self.install_btn = tk.Button(
            footer,
            text="Instalar Agora",
            command=self._start_install,
            font=("Segoe UI", 10, "bold"),
            bg="#6366f1",
            fg="#ffffff",
            activebackground="#4f46e5",
            activeforeground="#ffffff",
            relief="flat",
            padx=20,
            pady=6,
        )
        self.install_btn.pack(side="right")

        self.cancel_btn = tk.Button(
            footer,
            text="Cancelar",
            command=self.root.destroy,
            font=("Segoe UI", 9),
            bg="#1e2433",
            fg="#cbd5e1",
            relief="flat",
            padx=14,
            pady=6,
        )
        self.cancel_btn.pack(side="right", padx=(0, 8))

    def _choose_dir(self) -> None:
        chosen = filedialog.askdirectory(initialdir=self.install_dir_var.get())
        if chosen:
            self.install_dir_var.set(str(Path(chosen) / "FaceLab"))

    def _start_install(self) -> None:
        import threading

        self.install_btn.config(state="disabled")
        self.browse_btn.config(state="disabled")
        self.cancel_btn.config(state="disabled")
        threading.Thread(target=self._run_extraction, daemon=True).start()

    def _update_ui(self, pct: float, msg: str) -> None:
        self.root.after(0, lambda: (self.progress.configure(value=pct), self.status_var.set(msg)))

    def _run_extraction(self) -> None:
        try:
            target_dir = Path(self.install_dir_var.get()).expanduser().resolve()
            target_dir.mkdir(parents=True, exist_ok=True)

            exe_path = Path(sys.executable).resolve()
            info = find_payload_offset(exe_path)
            if info is None:
                raise RuntimeError("Pacote comprimido não encontrado dentro do instalador.")

            offset, length = info
            self._update_ui(5, "Preparando pacote de instalação (extraindo arquivo comprimido)...")

            with tempfile.TemporaryDirectory(prefix="facelab_setup_") as tmp_dir:
                archive_7z = Path(tmp_dir) / "payload.7z"
                copied = 0
                buf_size = 8 * 1024 * 1024
                with open(exe_path, "rb") as src, open(archive_7z, "wb") as dst:
                    src.seek(offset)
                    while copied < length:
                        chunk = src.read(min(buf_size, length - copied))
                        if not chunk:
                            break
                        dst.write(chunk)
                        copied += len(chunk)
                        pct = 5 + (copied / length) * 35
                        self._update_ui(pct, f"Preparando pacote ({copied // (1024**2)} / {length // (1024**2)} MB)...")

                self._update_ui(42, "Descompactando arquivos do FaceLab Studio (isso pode levar ~1 minuto)...")
                seven_zip = get_bundle_resource("7zr.exe")
                proc = subprocess.Popen(
                    [str(seven_zip), "x", "-y", str(archive_7z), f"-o{target_dir}"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
                )
                if proc.stdout:
                    for line in proc.stdout:
                        pass
                ret = proc.wait()
                if ret != 0:
                    raise RuntimeError(f"Erro ao descompactar arquivos (código {ret}).")

            facelab_exe = target_dir / "FaceLab.exe"
            if not facelab_exe.is_file() and (target_dir / "FaceLab" / "FaceLab.exe").is_file():
                facelab_exe = target_dir / "FaceLab" / "FaceLab.exe"

            self._update_ui(92, "Criando atalhos no sistema...")
            if self.desktop_shortcut_var.get():
                desktop = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Desktop"
                create_windows_shortcut(facelab_exe, desktop / "FaceLab Studio.lnk", "FaceLab Studio")

            if self.startmenu_shortcut_var.get():
                start_menu = (
                    Path(os.environ.get("APPDATA", str(Path.home())))
                    / "Microsoft"
                    / "Windows"
                    / "Start Menu"
                    / "Programs"
                )
                create_windows_shortcut(facelab_exe, start_menu / "FaceLab Studio.lnk", "FaceLab Studio")

            self._update_ui(100, "Instalação concluída com sucesso!")

            def on_done() -> None:
                if self.launch_after_var.get() and facelab_exe.is_file():
                    subprocess.Popen([str(facelab_exe)], cwd=str(facelab_exe.parent))
                else:
                    messagebox.showinfo(
                        "Instalação Concluída",
                        f"O {APP_NAME} foi instalado com sucesso em:\n{facelab_exe.parent}",
                    )
                self.root.destroy()

            self.root.after(400, on_done)
        except Exception as exc:
            self.root.after(
                0,
                lambda: (
                    messagebox.showerror("Erro na Instalação", str(exc)),
                    self.install_btn.config(state="normal"),
                    self.browse_btn.config(state="normal"),
                    self.cancel_btn.config(state="normal"),
                ),
            )


def main() -> int:
    root = tk.Tk()
    InstallerApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

