"""Janela Principal do FaceLab (PySide6 Desktop Application).

Recursos implementados:
1. Aba "Vídeo Gravado":
   - Seleção e pré-visualização de Vídeo e Imagem Facial de Origem.
   - Seleção de Dispositivo (Auto, GPU CUDA, CPU).
   - Ativação/Desativação de Restauração Facial (GFPGAN).
   - Sliders de Ajuste Fino: Intensidade da Troca (0% a 100%), Suavização de Bordas (Feathering), e Correção de Cor/Iluminação.
   - Barra de progresso com FPS, tempo decorrido e estimativa de término (ETA).
2. Aba "Webcam ao Vivo (Live)":
   - Visualização ao vivo diretamente na interface gráfica.
   - Seleção de Câmera (Índice / DroidCam / OBS Virtual Cam).
   - Aplicação em tempo real dos sliders de intensidade, bordas e cor.
   - Controles de Iniciar e Parar o stream com contador de FPS ao vivo.
3. Avisos éticos de identificação de conteúdo sintético gerado por IA.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from config.settings import OUTPUT_DIR, SUPPORTED_IMAGE_EXTENSIONS, SUPPORTED_VIDEO_EXTENSIONS
from src.inference.device_manager import DevicePreference
from src.interface.worker import LiveWebcamWorker, VideoWorker
from src.pipeline.video_pipeline import PipelineProgress

logger = logging.getLogger(__name__)


def cv2_to_pixmap(cv_img: np.ndarray | None, max_size=(480, 270)) -> QPixmap:
    """Converte imagem BGR do OpenCV para QPixmap redimensionado mantendo proporção."""
    if cv_img is None:
        return QPixmap()
    h, w, ch = cv_img.shape
    bytes_per_line = ch * w
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    q_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    pix = QPixmap.fromImage(q_img)
    return pix.scaled(
        max_size[0], max_size[1], Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
    )


class MainWindow(QWidget):
    """Janela principal da aplicação FaceLab com abas de Vídeo Gravado e Webcam ao Vivo."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("FaceLab — Software Educacional de Face Swap")
        self.resize(1020, 820)

        # Estados de mídia
        self.video_path: Path | None = None
        self.image_path: Path | None = None

        # Threads e Workers
        self.video_thread: QThread | None = None
        self.video_worker: VideoWorker | None = None

        self.webcam_thread: QThread | None = None
        self.webcam_worker: LiveWebcamWorker | None = None

        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        # Cabeçalho / Banner Ético
        header_box = QGroupBox("Aviso Ético e Educacional")
        header_layout = QVBoxLayout(header_box)
        ethical_label = QLabel(
            "<b>Projeto Educacional:</b> O FaceLab destina-se exclusivamente ao estudo de visão computacional e redes neurais.<br>"
            "Os vídeos exportados incluem metadados de identificação como conteúdo gerado/alterado por IA."
        )
        ethical_label.setWordWrap(True)
        header_layout.addWidget(ethical_label)
        main_layout.addWidget(header_box)

        # Painel Compartilhado: Rosto de Referência (Source)
        source_box = QGroupBox("Rosto de Referência (Source Face)")
        s_layout = QHBoxLayout(source_box)

        self.img_preview = QLabel("Nenhuma imagem selecionada")
        self.img_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_preview.setStyleSheet(
            "border: 1px dashed #666; background-color: #1e1e1e; min-width: 140px; min-height: 140px; max-width: 160px; max-height: 160px;"
        )
        s_layout.addWidget(self.img_preview)

        s_info_layout = QVBoxLayout()
        self.lbl_img_info = QLabel("Dimensões: Selecione uma foto nítida de rosto frontal.")
        s_info_layout.addWidget(self.lbl_img_info)

        self.btn_select_img = QPushButton("Selecionar Imagem do Rosto...")
        self.btn_select_img.setStyleSheet("font-weight: bold; padding: 6px;")
        self.btn_select_img.clicked.connect(self._select_image)
        s_info_layout.addWidget(self.btn_select_img)
        s_layout.addLayout(s_info_layout)

        main_layout.addWidget(source_box)

        # Painel Compartilhado: Controles de Ajuste Fino
        controls_box = QGroupBox("Ajustes Finais de Composição Facial")
        c_layout = QHBoxLayout(controls_box)

        # 1. Intensidade do Swap (Slider 0% a 100%)
        vbox_blend = QVBoxLayout()
        self.lbl_blend = QLabel("Intensidade da Troca: 100%")
        self.slider_blend = QSlider(Qt.Orientation.Horizontal)
        self.slider_blend.setRange(0, 100)
        self.slider_blend.setValue(100)
        self.slider_blend.valueChanged.connect(self._on_blend_changed)
        vbox_blend.addWidget(self.lbl_blend)
        vbox_blend.addWidget(self.slider_blend)
        c_layout.addLayout(vbox_blend)

        # 2. Suavização de Bordas / Feathering (Slider 5 a 45)
        vbox_feather = QVBoxLayout()
        self.lbl_feather = QLabel("Suavização de Borda: 15 px")
        self.slider_feather = QSlider(Qt.Orientation.Horizontal)
        self.slider_feather.setRange(3, 45)
        self.slider_feather.setSingleStep(2)
        self.slider_feather.setValue(15)
        self.slider_feather.valueChanged.connect(self._on_feather_changed)
        vbox_feather.addWidget(self.lbl_feather)
        vbox_feather.addWidget(self.slider_feather)
        c_layout.addLayout(vbox_feather)

        # 3. Correção de Cor / Iluminação
        vbox_color = QVBoxLayout()
        self.chk_color_correct = QCheckBox("Balanço de Cor e Iluminação")
        self.chk_color_correct.setChecked(True)
        self.chk_color_correct.setToolTip("Ajusta a luminância e os canais RGB do novo rosto para combinar com o tom de pele do alvo.")
        vbox_color.addWidget(QLabel("Harmonização:"))
        vbox_color.addWidget(self.chk_color_correct)
        c_layout.addLayout(vbox_color)

        main_layout.addWidget(controls_box)

        # Sistema de Abas (Tabs): Vídeo Gravado vs Webcam ao Vivo
        self.tabs = QTabWidget()
        self.tab_video = QWidget()
        self.tab_webcam = QWidget()

        self._setup_video_tab()
        self._setup_webcam_tab()

        self.tabs.addTab(self.tab_video, "🎬 Vídeo Gravado (Arquivo MP4)")
        self.tabs.addTab(self.tab_webcam, "🎥 Webcam ao Vivo (DroidCam / USB)")
        main_layout.addWidget(self.tabs)

    def _setup_video_tab(self) -> None:
        """Monta os controles da aba de Vídeo Gravado."""
        layout = QVBoxLayout(self.tab_video)

        # Área de Pré-visualização do Vídeo
        v_box = QGroupBox("Vídeo de Entrada")
        v_layout = QHBoxLayout(v_box)

        self.video_preview = QLabel("Nenhum vídeo selecionado")
        self.video_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_preview.setStyleSheet("border: 1px dashed #666; background-color: #1e1e1e; min-height: 160px; min-width: 240px;")
        v_layout.addWidget(self.video_preview)

        v_side = QVBoxLayout()
        self.btn_select_video = QPushButton("Selecionar Arquivo de Vídeo...")
        self.btn_select_video.clicked.connect(self._select_video)
        v_side.addWidget(self.btn_select_video)

        self.lbl_video_info = QLabel("Resolução: - | Duração: - | FPS: -")
        v_side.addWidget(self.lbl_video_info)
        v_layout.addLayout(v_side)
        layout.addWidget(v_box)

        # Configurações de Hardware e Destino
        config_box = QGroupBox("Parâmetros do Vídeo")
        config_layout = QHBoxLayout(config_box)

        config_layout.addWidget(QLabel("Dispositivo:"))
        self.combo_device = QComboBox()
        self.combo_device.addItems(["Automático (Recomendado)", "GPU (NVIDIA CUDA)", "CPU"])
        config_layout.addWidget(self.combo_device)

        self.chk_enhance = QCheckBox("Restaurar Nitidez (GFPGAN)")
        self.chk_enhance.setChecked(True)
        self.chk_enhance.setToolTip("Aplica super-resolução e textura facial em alta fidelidade com GFPGAN v1.4.")
        config_layout.addWidget(self.chk_enhance)

        config_layout.addWidget(QLabel("Destino:"))
        self.edit_output = QLineEdit(str(OUTPUT_DIR / "saida_swap.mp4"))
        config_layout.addWidget(self.edit_output)

        self.btn_browse_output = QPushButton("Procurar...")
        self.btn_browse_output.clicked.connect(self._browse_output)
        config_layout.addWidget(self.btn_browse_output)
        layout.addWidget(config_box)

        # Barra de Progresso
        prog_box = QGroupBox("Progresso")
        prog_layout = QVBoxLayout(prog_box)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        prog_layout.addWidget(self.progress_bar)

        self.lbl_status = QLabel("Pronto para iniciar.")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        prog_layout.addWidget(self.lbl_status)
        layout.addWidget(prog_box)

        # Botões de Ação
        action_layout = QHBoxLayout()
        self.btn_start = QPushButton("Iniciar Face Swap no Vídeo")
        self.btn_start.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; padding: 10px; font-size: 13px;")
        self.btn_start.clicked.connect(self._start_video_processing)
        action_layout.addWidget(self.btn_start)

        self.btn_cancel = QPushButton("Cancelar")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setStyleSheet("background-color: #c62828; color: white; font-weight: bold; padding: 10px; font-size: 13px;")
        self.btn_cancel.clicked.connect(self._cancel_video_processing)
        action_layout.addWidget(self.btn_cancel)
        layout.addLayout(action_layout)

    def _setup_webcam_tab(self) -> None:
        """Monta os controles e tela de exibição da aba de Webcam."""
        layout = QVBoxLayout(self.tab_webcam)

        # Configurações de Câmera
        cam_cfg_box = QGroupBox("Dispositivo de Captura")
        cam_cfg_layout = QHBoxLayout(cam_cfg_box)

        cam_cfg_layout.addWidget(QLabel("Câmera:"))
        self.combo_cam = QComboBox()
        self.combo_cam.addItem("Câmera Padrão / DroidCam (Índice 0)", 0)
        self.combo_cam.addItem("Câmera Secundária (Índice 1)", 1)
        self.combo_cam.addItem("Câmera Terciária (Índice 2)", 2)
        cam_cfg_layout.addWidget(self.combo_cam)

        self.lbl_webcam_fps = QLabel("FPS ao Vivo: 0.0")
        self.lbl_webcam_fps.setStyleSheet("font-weight: bold; color: #4caf50; font-size: 13px;")
        cam_cfg_layout.addWidget(self.lbl_webcam_fps)

        cam_cfg_layout.addStretch()

        self.btn_start_webcam = QPushButton("▶ Iniciar Câmera")
        self.btn_start_webcam.setStyleSheet("background-color: #1565c0; color: white; font-weight: bold; padding: 8px 16px;")
        self.btn_start_webcam.clicked.connect(self._toggle_webcam)
        cam_cfg_layout.addWidget(self.btn_start_webcam)

        layout.addWidget(cam_cfg_box)

        # Monitor de Exibição da Webcam
        cam_view_box = QGroupBox("Transmissão em Tempo Real")
        cam_view_layout = QVBoxLayout(cam_view_box)

        self.webcam_display = QLabel("Câmera desativada. Selecione um rosto de referência e clique em 'Iniciar Câmera'.")
        self.webcam_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.webcam_display.setStyleSheet("border: 2px solid #444; background-color: #121212; min-height: 320px;")
        cam_view_layout.addWidget(self.webcam_display)
        layout.addWidget(cam_view_box)

    # --- Callbacks de Sliders de Ajuste Fino ---
    def _on_blend_changed(self, value: int) -> None:
        self.lbl_blend.setText(f"Intensidade da Troca: {value}%")
        if self.webcam_worker:
            self.webcam_worker.blend_intensity = value / 100.0

    def _on_feather_changed(self, value: int) -> None:
        # Garante ímpar
        val = value if value % 2 != 0 else value + 1
        self.lbl_feather.setText(f"Suavização de Borda: {val} px")
        if self.webcam_worker and self.webcam_worker.pipeline:
            self.webcam_worker.feather_amount = val
            self.webcam_worker.pipeline.swapper.compositor.blur_amount = val

    # --- Seleção de Arquivos ---
    def _select_video(self) -> None:
        ext_filter = "Vídeos (*" + " *".join(SUPPORTED_VIDEO_EXTENSIONS) + ")"
        path, _ = QFileDialog.getOpenFileName(self, "Selecionar Vídeo de Origem", str(Path.cwd()), ext_filter)
        if path:
            self.video_path = Path(path)
            cap = cv2.VideoCapture(str(self.video_path))
            ret, frame = cap.read()
            if ret and frame is not None:
                self.video_preview.setPixmap(cv2_to_pixmap(frame, max_size=(240, 160)))
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                dur = frames / fps if fps > 0 else 0
                self.lbl_video_info.setText(f"{w}x{h} | {dur:.1f}s | {fps:.1f} FPS")
            cap.release()
            self.edit_output.setText(str(OUTPUT_DIR / f"{self.video_path.stem}_swapped.mp4"))

    def _select_image(self) -> None:
        ext_filter = "Imagens (*" + " *".join(SUPPORTED_IMAGE_EXTENSIONS) + ")"
        path, _ = QFileDialog.getOpenFileName(self, "Selecionar Rosto de Referência", str(Path.cwd()), ext_filter)
        if path:
            self.image_path = Path(path)
            img = cv2.imread(str(self.image_path))
            if img is not None:
                self.img_preview.setPixmap(cv2_to_pixmap(img, max_size=(160, 160)))
                h, w = img.shape[:2]
                self.lbl_img_info.setText(f"Dimensões: {w}x{h} px ({self.image_path.name})")

    def _browse_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Salvar Vídeo Como", self.edit_output.text(), "Vídeo MP4 (*.mp4)")
        if path:
            self.edit_output.setText(path)

    # --- Processamento de Vídeo Gravado ---
    def _start_video_processing(self) -> None:
        if not self.video_path or not self.video_path.is_file():
            QMessageBox.warning(self, "Atenção", "Por favor, selecione um vídeo de origem válido.")
            return
        if not self.image_path or not self.image_path.is_file():
            QMessageBox.warning(self, "Atenção", "Por favor, selecione uma imagem de rosto de referência válida.")
            return

        out_path = Path(self.edit_output.text())
        pref_idx = self.combo_device.currentIndex()
        pref = [DevicePreference.AUTO, DevicePreference.GPU, DevicePreference.CPU][pref_idx]

        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_select_video.setEnabled(False)
        self.btn_select_img.setEnabled(False)
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Carregando modelos neurais...")

        self.video_thread = QThread()
        self.video_worker = VideoWorker(
            video_path=self.video_path,
            image_path=self.image_path,
            output_path=out_path,
            device_preference=pref,
            enable_enhancer=self.chk_enhance.isChecked(),
            blend_intensity=self.slider_blend.value() / 100.0,
            color_correct=self.chk_color_correct.isChecked(),
            feather_amount=self.slider_feather.value(),
        )
        self.video_worker.moveToThread(self.video_thread)

        self.video_thread.started.connect(self.video_worker.run)
        self.video_worker.progress_updated.connect(self._on_video_progress)
        self.video_worker.finished.connect(self._on_video_finished)
        self.video_worker.error.connect(self._on_video_error)

        self.video_worker.finished.connect(self.video_thread.quit)
        self.video_worker.error.connect(self.video_thread.quit)
        self.video_thread.finished.connect(self._on_video_cleanup)

        self.video_thread.start()

    def _cancel_video_processing(self) -> None:
        if self.video_worker:
            self.lbl_status.setText("Cancelando...")
            self.video_worker.cancel()
            self.btn_cancel.setEnabled(False)

    def _on_video_progress(self, p: PipelineProgress) -> None:
        rem_seconds = (p.total_frames - p.current_frame) / p.fps if (p.total_frames and p.fps > 0) else 0.0
        eta_str = f" | Restante: {rem_seconds:.0f}s" if (p.total_frames and rem_seconds > 0) else ""

        if p.total_frames and p.total_frames > 0:
            val = int((p.current_frame / p.total_frames) * 100)
            self.progress_bar.setValue(val)
            txt = f"Quadro {p.current_frame}/{p.total_frames} ({val}%) | {p.fps:.1f} fps | {p.elapsed_seconds:.1f}s decorridos{eta_str}"
        else:
            txt = f"Quadro {p.current_frame} | {p.fps:.1f} fps | {p.elapsed_seconds:.1f}s decorridos"
        self.lbl_status.setText(txt)

    def _on_video_finished(self, output_file: str) -> None:
        self.progress_bar.setValue(100)
        self.lbl_status.setText("Processamento concluído com sucesso!")
        QMessageBox.information(self, "Sucesso", f"Vídeo gerado com sucesso em:\n{output_file}")

    def _on_video_error(self, err_msg: str) -> None:
        self.lbl_status.setText(f"Interrompido: {err_msg}")
        QMessageBox.warning(self, "Aviso", f"{err_msg}")

    def _on_video_cleanup(self) -> None:
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.btn_select_video.setEnabled(True)
        self.btn_select_img.setEnabled(True)
        self.video_thread = None
        self.video_worker = None

    # --- Processamento de Webcam ao Vivo ---
    def _toggle_webcam(self) -> None:
        if self.webcam_thread and self.webcam_thread.isRunning():
            self._stop_webcam()
        else:
            self._start_webcam()

    def _start_webcam(self) -> None:
        if not self.image_path or not self.image_path.is_file():
            QMessageBox.warning(self, "Atenção", "Por favor, selecione primeiro a foto do rosto de referência.")
            return

        cam_idx = int(self.combo_cam.currentData())
        self.btn_start_webcam.setText("⏹ Parar Câmera")
        self.btn_start_webcam.setStyleSheet("background-color: #c62828; color: white; font-weight: bold; padding: 8px 16px;")
        self.combo_cam.setEnabled(False)

        pref_idx = self.combo_device.currentIndex()
        pref = [DevicePreference.AUTO, DevicePreference.GPU, DevicePreference.CPU][pref_idx]

        self.webcam_thread = QThread()
        self.webcam_worker = LiveWebcamWorker(
            image_path=self.image_path,
            camera_index=cam_idx,
            device_preference=pref,
            blend_intensity=self.slider_blend.value() / 100.0,
            color_correct=self.chk_color_correct.isChecked(),
            feather_amount=self.slider_feather.value(),
        )
        self.webcam_worker.moveToThread(self.webcam_thread)

        self.webcam_thread.started.connect(self.webcam_worker.run)
        self.webcam_worker.frame_ready.connect(self._on_webcam_frame)
        self.webcam_worker.error.connect(self._on_webcam_error)
        self.webcam_worker.stopped.connect(self.webcam_thread.quit)
        self.webcam_thread.finished.connect(self._on_webcam_cleanup)

        self.webcam_thread.start()

    def _stop_webcam(self) -> None:
        if self.webcam_worker:
            self.webcam_worker.stop()
            self.btn_start_webcam.setEnabled(False)

    def _on_webcam_frame(self, frame_bgr: np.ndarray, fps: float) -> None:
        self.lbl_webcam_fps.setText(f"FPS ao Vivo: {fps:.1f}")
        # Converte para Pixmap e renderiza no QLabel de monitor
        pix = cv2_to_pixmap(frame_bgr, max_size=(self.webcam_display.width() - 10, self.webcam_display.height() - 10))
        self.webcam_display.setPixmap(pix)

    def _on_webcam_error(self, err_msg: str) -> None:
        QMessageBox.warning(self, "Erro na Câmera", f"Não foi possível acessar a câmera:\n{err_msg}")
        self._stop_webcam()

    def _on_webcam_cleanup(self) -> None:
        self.btn_start_webcam.setText("▶ Iniciar Câmera")
        self.btn_start_webcam.setEnabled(True)
        self.btn_start_webcam.setStyleSheet("background-color: #1565c0; color: white; font-weight: bold; padding: 8px 16px;")
        self.combo_cam.setEnabled(True)
        self.webcam_thread = None
        self.webcam_worker = None
        self.webcam_display.setText("Câmera encerrada.")
        self.lbl_webcam_fps.setText("FPS ao Vivo: 0.0")

    def closeEvent(self, event) -> None:
        """Garante encerramento seguro de threads ativas ao fechar a janela."""
        if self.webcam_worker:
            self.webcam_worker.stop()
        if self.video_worker:
            self.video_worker.cancel()
        if self.webcam_thread and self.webcam_thread.isRunning():
            self.webcam_thread.quit()
            self.webcam_thread.wait(2000)
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.quit()
            self.video_thread.wait(2000)
        event.accept()
