"""Janela Principal do FaceLab (PySide6 Desktop Application).

Design Moderno & Responsivo:
- Interface contida em QScrollArea para acomodar qualquer resolução de tela (1080p, 720p ou notebooks) com rolagem vertical suave.
- Dimensões proporcionais e elásticas para evitar cortes na visualização da webcam e dos painéis.
- Tema Cyberpunk / Sci-Fi Studio Dark & Neon Blue.
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
    QScrollArea,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from config.settings import (
    CYBORG_REFERENCE_IMAGE,
    OUTPUT_DIR,
    SUPPORTED_IMAGE_EXTENSIONS,
    SUPPORTED_VIDEO_EXTENSIONS,
    VAMPIRE_REFERENCE_IMAGE,
    ZOMBIE_REFERENCE_IMAGE,
)
from src.inference.device_manager import DevicePreference
from src.interface.theme import DARK_BLUE_THEME
from src.interface.worker import LiveWebcamWorker, VideoWorker
from src.pipeline.video_pipeline import PipelineProgress

logger = logging.getLogger(__name__)


def cv2_to_pixmap(cv_img: np.ndarray | None, max_size=(540, 340)) -> QPixmap:
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
    """Janela principal da aplicação FaceLab com interface responsiva e rolável."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("⚡ FaceLab Studio — Neural Face Swap & Restoration")
        self.resize(1060, 780)
        self.setMinimumSize(850, 600)

        # Estados de mídia
        self.video_path: Path | None = None
        self.image_path: Path | None = None

        # Threads e Workers
        self.video_thread: QThread | None = None
        self.video_worker: VideoWorker | None = None

        self.webcam_thread: QThread | None = None
        self.webcam_worker: LiveWebcamWorker | None = None

        # Aplica o tema Dark / Neon Blue
        self.setStyleSheet(DARK_BLUE_THEME)

        self._init_ui()

    def _init_ui(self) -> None:
        # Layout raiz da janela
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # QScrollArea envolvendo todo o conteúdo
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Widget interno que recebe todos os painéis
        content_widget = QWidget()
        content_widget.setObjectName("centralWidget")
        main_layout = QVBoxLayout(content_widget)
        main_layout.setContentsMargins(18, 14, 18, 18)
        main_layout.setSpacing(12)

        # 1. Cabeçalho / Banner Ético Sofisticado
        header_box = QGroupBox("DIRETRIZES & METADADOS ÉTICOS")
        header_box.setObjectName("headerBox")
        header_layout = QVBoxLayout(header_box)
        header_layout.setContentsMargins(12, 12, 12, 10)
        ethical_label = QLabel(
            "🛡️ <b>FaceLab AI Studio:</b> Ambiente educacional focado em visão computacional, redes neurais e aceleração gráfica.<br>"
            "Os arquivos exportados contêm assinaturas digitais e metadados identificando conteúdo sintético gerado por IA."
        )
        ethical_label.setStyleSheet("color: #94a3b8; font-size: 11.5px; line-height: 1.3;")
        ethical_label.setWordWrap(True)
        header_layout.addWidget(ethical_label)
        main_layout.addWidget(header_box)

        # 2. Painel: Rosto de Referência (Source Face)
        source_box = QGroupBox("1. ROSTO DE ORIGEM (SOURCE IDENTITY)")
        s_layout = QHBoxLayout(source_box)
        s_layout.setContentsMargins(14, 14, 14, 12)
        s_layout.setSpacing(16)

        self.img_preview = QLabel("Nenhuma imagem selecionada")
        self.img_preview.setObjectName("previewLabel")
        self.img_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_preview.setStyleSheet(
            "min-width: 120px; min-height: 120px; max-width: 130px; max-height: 130px;"
        )
        s_layout.addWidget(self.img_preview)

        s_info_layout = QVBoxLayout()
        s_info_layout.setSpacing(6)

        lbl_title = QLabel("<b>Identidade Alvo:</b> Carregue uma imagem frontal nítida (JPG/PNG).")
        lbl_title.setStyleSheet("color: #e2e8f0; font-size: 12.5px;")
        s_info_layout.addWidget(lbl_title)

        self.lbl_img_info = QLabel("Status: Aguardando seleção de foto.")
        self.lbl_img_info.setStyleSheet("color: #64748b; font-size: 11.5px;")
        s_info_layout.addWidget(self.lbl_img_info)

        s_info_layout.addStretch()

        self.btn_select_img = QPushButton("📁 Carregar Imagem Facial...")
        self.btn_select_img.setObjectName("secondaryBtn")
        self.btn_select_img.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_select_img.clicked.connect(self._select_image)
        s_info_layout.addWidget(self.btn_select_img)

        s_layout.addLayout(s_info_layout)
        main_layout.addWidget(source_box)

        # 3. Painel de Ajustes Finos (Sliders com gradiente neon)
        controls_box = QGroupBox("2. PARÂMETROS NEURAIS DE COMPOSIÇÃO")
        c_layout = QHBoxLayout(controls_box)
        c_layout.setContentsMargins(14, 14, 14, 12)
        c_layout.setSpacing(20)

        # Slider: Intensidade do Face Swap (0 a 100%)
        vbox_blend = QVBoxLayout()
        vbox_blend.setSpacing(4)
        self.lbl_blend = QLabel("Intensidade do Swap: <b>100%</b>")
        self.lbl_blend.setStyleSheet("color: #38bdf8;")
        self.slider_blend = QSlider(Qt.Orientation.Horizontal)
        self.slider_blend.setRange(0, 100)
        self.slider_blend.setValue(100)
        self.slider_blend.valueChanged.connect(self._on_blend_changed)
        vbox_blend.addWidget(self.lbl_blend)
        vbox_blend.addWidget(self.slider_blend)
        c_layout.addLayout(vbox_blend)

        # Slider: Suavização de Bordas / Feathering (3 a 45px)
        vbox_feather = QVBoxLayout()
        vbox_feather.setSpacing(4)
        self.lbl_feather = QLabel("Suavização Anatômica: <b>15 px</b>")
        self.lbl_feather.setStyleSheet("color: #38bdf8;")
        self.slider_feather = QSlider(Qt.Orientation.Horizontal)
        self.slider_feather.setRange(3, 45)
        self.slider_feather.setSingleStep(2)
        self.slider_feather.setValue(15)
        self.slider_feather.valueChanged.connect(self._on_feather_changed)
        vbox_feather.addWidget(self.lbl_feather)
        vbox_feather.addWidget(self.slider_feather)
        c_layout.addLayout(vbox_feather)

        # Checkbox: Correspondência de Iluminação & Cor
        vbox_color = QVBoxLayout()
        vbox_color.setSpacing(4)
        vbox_color.addWidget(QLabel("Harmonização de Pele:"))
        self.chk_color_correct = QCheckBox("Correção de Iluminação")
        self.chk_color_correct.setChecked(True)
        self.chk_color_correct.setToolTip("Harmoniza temperatura de cor e brilho do novo rosto com a cena original.")
        self.chk_color_correct.toggled.connect(self._on_color_correct_toggled)
        vbox_color.addWidget(self.chk_color_correct)
        c_layout.addLayout(vbox_color)

        # Dropdown de Presets Cinematográficos
        vbox_presets = QVBoxLayout()
        vbox_presets.setSpacing(4)
        vbox_presets.addWidget(QLabel("🎭 Presets Cinematográficos:"))
        self.combo_presets = QComboBox()
        self.combo_presets.addItem("👤 Personalizado (Foto Carregada)", None)
        self.combo_presets.addItem("🧟 Zumbi Hiper-realista", str(ZOMBIE_REFERENCE_IMAGE))
        self.combo_presets.addItem("🧛 Vampiro Nobre", str(VAMPIRE_REFERENCE_IMAGE))
        self.combo_presets.addItem("🤖 Ciborgue Futurista", str(CYBORG_REFERENCE_IMAGE))
        self.combo_presets.currentIndexChanged.connect(self._on_preset_changed)
        vbox_presets.addWidget(self.combo_presets)
        c_layout.addLayout(vbox_presets)

        # Botão Especial de Halloween: Modo Zumbi
        vbox_zombie = QVBoxLayout()
        vbox_zombie.setSpacing(4)
        vbox_zombie.addWidget(QLabel("🎃 Halloween Especial:"))
        self.btn_zombie = QPushButton("🧟 MODO ZUMBI: OFF")
        self.btn_zombie.setObjectName("zombieBtn")
        self.btn_zombie.setCheckable(True)
        self.btn_zombie.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_zombie.setToolTip("Transforma o rosto em um zumbi realista com necrose, cortes profundos e sangue!")
        self.btn_zombie.toggled.connect(self._on_zombie_toggled)
        vbox_zombie.addWidget(self.btn_zombie)
        c_layout.addLayout(vbox_zombie)

        main_layout.addWidget(controls_box)

        # 4. Sistema de Abas (Tabs) Modernas
        self.tabs = QTabWidget()
        self.tab_video = QWidget()
        self.tab_webcam = QWidget()

        self._setup_video_tab()
        self._setup_webcam_tab()

        self.tabs.addTab(self.tab_video, "🎬 VÍDEO OFFLINE (MP4 / MKV)")
        self.tabs.addTab(self.tab_webcam, "🎥 TRANSMISSÃO AO VIVO (WEBCAM / DROIDCAM)")
        main_layout.addWidget(self.tabs)

        scroll_area.setWidget(content_widget)
        root_layout.addWidget(scroll_area)

    def _setup_video_tab(self) -> None:
        """Monta os controles e monitor da aba de Vídeo Gravado."""
        layout = QVBoxLayout(self.tab_video)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        # Painel do Vídeo de Entrada
        v_box = QGroupBox("VÍDEO DE ENTRADA")
        v_layout = QHBoxLayout(v_box)
        v_layout.setSpacing(14)

        self.video_preview = QLabel("Nenhum vídeo carregado")
        self.video_preview.setObjectName("previewLabel")
        self.video_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_preview.setStyleSheet("min-height: 120px; min-width: 200px; max-width: 240px;")
        v_layout.addWidget(self.video_preview)

        v_side = QVBoxLayout()
        v_side.setSpacing(8)
        self.lbl_video_info = QLabel("Resolução: - | Duração: - | FPS: -")
        self.lbl_video_info.setStyleSheet("color: #64748b; font-size: 12px;")
        v_side.addWidget(self.lbl_video_info)
        v_side.addStretch()

        self.btn_select_video = QPushButton("📂 Selecionar Arquivo de Vídeo...")
        self.btn_select_video.setObjectName("secondaryBtn")
        self.btn_select_video.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_select_video.clicked.connect(self._select_video)
        v_side.addWidget(self.btn_select_video)

        v_layout.addLayout(v_side)
        layout.addWidget(v_box)

        # Parâmetros de Inferência e Arquivo de Saída
        config_box = QGroupBox("CONFIGURAÇÃO DE INFERÊNCIA & EXPORTAÇÃO")
        config_layout = QHBoxLayout(config_box)
        config_layout.setSpacing(12)

        config_layout.addWidget(QLabel("Processamento:"))
        self.combo_device = QComboBox()
        self.combo_device.addItems(["Automático (GPU)", "NVIDIA CUDA (GPU)", "CPU Fallback"])
        config_layout.addWidget(self.combo_device)

        self.chk_enhance = QCheckBox("Restaurar com GFPGAN v1.4")
        self.chk_enhance.setChecked(True)
        self.chk_enhance.setToolTip("Aplica super-resolução e reconstrução de textura e olhos em alta definição.")
        config_layout.addWidget(self.chk_enhance)

        config_layout.addWidget(QLabel("Saída:"))
        self.edit_output = QLineEdit(str(OUTPUT_DIR / "saida_swap.mp4"))
        config_layout.addWidget(self.edit_output)

        self.btn_browse_output = QPushButton("Procurar...")
        self.btn_browse_output.setObjectName("secondaryBtn")
        self.btn_browse_output.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_browse_output.clicked.connect(self._browse_output)
        config_layout.addWidget(self.btn_browse_output)
        layout.addWidget(config_box)

        # Barra de Progresso Futurista
        prog_box = QGroupBox("MONITOR DE RENDERIZAÇÃO")
        prog_layout = QVBoxLayout(prog_box)
        prog_layout.setSpacing(8)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        prog_layout.addWidget(self.progress_bar)

        self.lbl_status = QLabel("Pronto para iniciar renderização.")
        self.lbl_status.setObjectName("statusLabel")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        prog_layout.addWidget(self.lbl_status)
        layout.addWidget(prog_box)

        # Botões de Ação
        action_layout = QHBoxLayout()
        action_layout.setSpacing(12)

        self.btn_start = QPushButton("🚀 INICIAR PROCESSAMENTO")
        self.btn_start.setObjectName("primaryActionBtn")
        self.btn_start.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_start.clicked.connect(self._start_video_processing)
        action_layout.addWidget(self.btn_start)

        self.btn_cancel = QPushButton("CANCELAR")
        self.btn_cancel.setObjectName("cancelBtn")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_cancel.clicked.connect(self._cancel_video_processing)
        action_layout.addWidget(self.btn_cancel)
        layout.addLayout(action_layout)

    def _setup_webcam_tab(self) -> None:
        """Monta os controles e monitor da aba de Webcam de forma elástica e totalmente visível."""
        layout = QVBoxLayout(self.tab_webcam)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        # Painel Superior de Configuração da Câmera
        cam_cfg_box = QGroupBox("DISPOSITIVO DE CAPTURA AO VIVO")
        cam_cfg_layout = QHBoxLayout(cam_cfg_box)
        cam_cfg_layout.setSpacing(14)

        cam_cfg_layout.addWidget(QLabel("Fonte de Vídeo:"))
        self.combo_cam = QComboBox()
        self.combo_cam.addItem("Câmera Principal / DroidCam (Índice 0)", 0)
        self.combo_cam.addItem("Câmera Secundária / OBS (Índice 1)", 1)
        self.combo_cam.addItem("Câmera Terciária (Índice 2)", 2)
        cam_cfg_layout.addWidget(self.combo_cam)

        self.lbl_webcam_fps = QLabel("LIVE FPS: 0.0")
        self.lbl_webcam_fps.setObjectName("liveFpsLabel")
        cam_cfg_layout.addWidget(self.lbl_webcam_fps)

        cam_cfg_layout.addStretch()

        self.btn_start_webcam = QPushButton("▶ INICIAR STREAM")
        self.btn_start_webcam.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_start_webcam.clicked.connect(self._toggle_webcam)
        cam_cfg_layout.addWidget(self.btn_start_webcam)

        layout.addWidget(cam_cfg_box)

        # Monitor de Exibição da Webcam em Tempo Real com proporção adaptativa
        cam_view_box = QGroupBox("DISPLAY EM TEMPO REAL")
        cam_view_layout = QVBoxLayout(cam_view_box)

        self.webcam_display = QLabel("Feed de vídeo desativado.<br>Selecione um rosto de referência acima e clique em <b>Iniciar Stream</b>.")
        self.webcam_display.setObjectName("previewLabel")
        self.webcam_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.webcam_display.setMinimumHeight(320)
        self.webcam_display.setScaledContents(False)
        cam_view_layout.addWidget(self.webcam_display)
        layout.addWidget(cam_view_box)

    # --- Callbacks dos Sliders Dinâmicos & Presets ---
    def _on_preset_changed(self, index: int) -> None:
        """Carrega instantaneamente o preset selecionado (Zumbi, Vampiro, Ciborgue)."""
        preset_path = self.combo_presets.currentData()
        if preset_path:
            p = Path(preset_path)
            if p.is_file():
                self.image_path = p
                img = cv2.imread(str(self.image_path))
                if img is not None:
                    h, w = img.shape[:2]
                    self.img_preview.setPixmap(cv2_to_pixmap(img, max_size=(160, 160)))
                    self.lbl_img_info.setText(
                        f"<b>Preset Ativo:</b> {self.combo_presets.currentText()}<br><b>Resolução:</b> {w}x{h} px"
                    )

                # Se o usuário escolheu o preset Zumbi, sincroniza o botão especial
                if "zombie" in p.name.lower():
                    self.btn_zombie.setChecked(True)
                else:
                    self.btn_zombie.setChecked(False)

                # Atualiza a webcam ao vivo se estiver rodando
                if self.webcam_worker:
                    self.webcam_worker.image_path = self.image_path
                    if self.webcam_worker.pipeline:
                        try:
                            new_face = self.webcam_worker.pipeline.load_source_face(self.image_path, det_thresh=0.2)
                            # Atualiza dinamicamente a referência de swap na thread
                            # O pipeline recarregará no próximo frame
                        except Exception as e:
                            logger.warning("Erro ao recarregar face do preset: %s", e)

    def _on_zombie_toggled(self, checked: bool) -> None:
        if checked:
            self.btn_zombie.setText("🧟 MODO ZUMBI: ATIVO")
        else:
            self.btn_zombie.setText("🧟 MODO ZUMBI: OFF")
        if self.webcam_worker:
            self.webcam_worker.update_params(enable_zombie=checked)

    def _on_blend_changed(self, value: int) -> None:
        self.lbl_blend.setText(f"Intensidade do Swap: <b>{value}%</b>")
        if self.webcam_worker:
            self.webcam_worker.update_params(blend_intensity=value / 100.0)

    def _on_feather_changed(self, value: int) -> None:
        val = value if value % 2 != 0 else value + 1
        self.lbl_feather.setText(f"Suavização Anatômica: <b>{val} px</b>")
        if self.webcam_worker:
            self.webcam_worker.update_params(feather_amount=val)

    def _on_color_correct_toggled(self, checked: bool) -> None:
        if self.webcam_worker:
            self.webcam_worker.update_params(color_correct=checked)

    # --- Seleção de Arquivos de Mídia ---
    def _select_video(self) -> None:
        ext_filter = "Vídeos (*" + " *".join(SUPPORTED_VIDEO_EXTENSIONS) + ")"
        path, _ = QFileDialog.getOpenFileName(self, "Selecionar Vídeo de Origem", str(Path.cwd()), ext_filter)
        if path:
            self.video_path = Path(path)
            cap = cv2.VideoCapture(str(self.video_path))
            ret, frame = cap.read()
            if ret and frame is not None:
                self.video_preview.setPixmap(cv2_to_pixmap(frame, max_size=(240, 140)))
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                dur = frames / fps if fps > 0 else 0
                self.lbl_video_info.setText(f"<b>Dimensões:</b> {w}x{h} px | <b>Duração:</b> {dur:.1f}s | <b>FPS:</b> {fps:.1f}")
            cap.release()
            self.edit_output.setText(str(OUTPUT_DIR / f"{self.video_path.stem}_swapped.mp4"))

    def _select_image(self) -> None:
        ext_filter = "Imagens (*" + " *".join(SUPPORTED_IMAGE_EXTENSIONS) + ")"
        path, _ = QFileDialog.getOpenFileName(self, "Selecionar Imagem Facial", str(Path.cwd()), ext_filter)
        if path:
            self.image_path = Path(path)
            img = cv2.imread(str(self.image_path))
            if img is not None:
                self.img_preview.setPixmap(cv2_to_pixmap(img, max_size=(130, 130)))
                h, w = img.shape[:2]
                self.lbl_img_info.setText(f"<b>Rosto:</b> {self.image_path.name} ({w}x{h} px)")

    def _browse_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Salvar Vídeo Como", self.edit_output.text(), "Vídeo MP4 (*.mp4)")
        if path:
            self.edit_output.setText(path)

    # --- Execução de Vídeo Offline ---
    def _start_video_processing(self) -> None:
        if not self.video_path or not self.video_path.is_file():
            QMessageBox.warning(self, "Atenção", "Por favor, selecione um arquivo de vídeo de origem válido.")
            return

        is_zombie_mode = self.btn_zombie.isChecked()
        has_source_img = self.image_path and self.image_path.is_file()

        # Se não estiver no modo zumbi, precisa obrigatoriamente de um rosto de referência para o swap
        if not is_zombie_mode and not has_source_img:
            QMessageBox.warning(
                self,
                "Atenção",
                "Por favor, selecione uma imagem de rosto de referência (ou ative o Modo Zumbi para transformar direto o vídeo original).",
            )
            return

        out_path = Path(self.edit_output.text())
        pref_idx = self.combo_device.currentIndex()
        pref = [DevicePreference.AUTO, DevicePreference.GPU, DevicePreference.CPU][pref_idx]

        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_select_video.setEnabled(False)
        self.btn_select_img.setEnabled(False)
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Carregando redes neurais e inicializando processamento...")

        self.video_thread = QThread()
        self.video_worker = VideoWorker(
            video_path=self.video_path,
            image_path=self.image_path if has_source_img else None,
            output_path=out_path,
            device_preference=pref,
            enable_enhancer=self.chk_enhance.isChecked(),
            blend_intensity=self.slider_blend.value() / 100.0,
            color_correct=self.chk_color_correct.isChecked(),
            feather_amount=self.slider_feather.value(),
            enable_zombie=is_zombie_mode,
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
            self.lbl_status.setText("Interrompendo pipeline...")
            self.video_worker.cancel()
            self.btn_cancel.setEnabled(False)

    def _on_video_progress(self, p: PipelineProgress) -> None:
        rem_seconds = (p.total_frames - p.current_frame) / p.fps if (p.total_frames and p.fps > 0) else 0.0
        eta_str = f" | Restante estimado: {rem_seconds:.0f}s" if (p.total_frames and rem_seconds > 0) else ""

        if p.total_frames and p.total_frames > 0:
            val = int((p.current_frame / p.total_frames) * 100)
            self.progress_bar.setValue(val)
            txt = f"Quadro {p.current_frame}/{p.total_frames} ({val}%) • {p.fps:.1f} fps • {p.elapsed_seconds:.1f}s decorridos{eta_str}"
        else:
            txt = f"Quadro {p.current_frame} • {p.fps:.1f} fps • {p.elapsed_seconds:.1f}s decorridos"
        self.lbl_status.setText(txt)

    def _on_video_finished(self, output_file: str) -> None:
        self.progress_bar.setValue(100)
        self.lbl_status.setText("✔ Renderização concluída com sucesso!")
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

    # --- Execução da Webcam ao Vivo ---
    def _toggle_webcam(self) -> None:
        if self.webcam_thread and self.webcam_thread.isRunning():
            self._stop_webcam()
        else:
            self._start_webcam()

    def _start_webcam(self) -> None:
        is_zombie_mode = self.btn_zombie.isChecked()
        has_source_img = self.image_path and self.image_path.is_file()

        if not is_zombie_mode and not has_source_img:
            QMessageBox.warning(
                self,
                "Atenção",
                "Por favor, selecione primeiro a foto do rosto de referência (ou ative o Modo Zumbi para transformar direto a sua webcam).",
            )
            return

        cam_idx = int(self.combo_cam.currentData())
        self.btn_start_webcam.setText("⏹ PARAR STREAM")
        self.btn_start_webcam.setObjectName("cancelBtn")
        self.btn_start_webcam.setStyle(self.style())
        self.combo_cam.setEnabled(False)

        pref_idx = self.combo_device.currentIndex()
        pref = [DevicePreference.AUTO, DevicePreference.GPU, DevicePreference.CPU][pref_idx]

        self.webcam_thread = QThread()
        self.webcam_worker = LiveWebcamWorker(
            image_path=self.image_path if has_source_img else None,
            camera_index=cam_idx,
            device_preference=pref,
            blend_intensity=self.slider_blend.value() / 100.0,
            color_correct=self.chk_color_correct.isChecked(),
            feather_amount=self.slider_feather.value(),
            enable_zombie=is_zombie_mode,
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
        self.lbl_webcam_fps.setText(f"LIVE FPS: {fps:.1f}")
        # Calcula tamanho máximo proporcional ao container
        target_w = max(320, self.webcam_display.width() - 16)
        target_h = max(240, self.webcam_display.height() - 16)
        pix = cv2_to_pixmap(frame_bgr, max_size=(target_w, target_h))
        self.webcam_display.setPixmap(pix)

    def _on_webcam_error(self, err_msg: str) -> None:
        QMessageBox.warning(self, "Erro na Câmera", f"Não foi possível acessar a câmera:\n{err_msg}")
        self._stop_webcam()

    def _on_webcam_cleanup(self) -> None:
        self.btn_start_webcam.setText("▶ INICIAR STREAM")
        self.btn_start_webcam.setEnabled(True)
        self.btn_start_webcam.setObjectName("")
        self.btn_start_webcam.setStyle(self.style())
        self.combo_cam.setEnabled(True)
        self.webcam_thread = None
        self.webcam_worker = None
        self.webcam_display.setText("Stream de vídeo finalizado.")
        self.lbl_webcam_fps.setText("LIVE FPS: 0.0")

    def closeEvent(self, event) -> None:
        """Garante encerramento gracioso ao fechar o app."""
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
