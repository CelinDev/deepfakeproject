"""Tema Dark Sci-Fi / Cyberpunk sofisticado em tons de preto e azul neon para FaceLab."""

DARK_BLUE_THEME = """
/* ==========================================================================
   FaceLab — Dark Sophisticated Theme (Deep Black & Electric Cyan/Blue)
   ========================================================================== */

QWidget {
    background-color: #0b0f19;
    color: #e2e8f0;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    font-size: 13px;
    selection-background-color: #0284c7;
    selection-color: #ffffff;
}

/* Janela Principal */
QMainWindow, QWidget#centralWidget {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #070a12, stop:1 #0d1527);
}

/* GroupBox Sofisticados com Borda Sutil e Brilho Azul */
QGroupBox {
    background-color: rgba(15, 23, 42, 0.75);
    border: 1px solid #1e293b;
    border-radius: 10px;
    margin-top: 14px;
    padding-top: 18px;
    padding-left: 14px;
    padding-right: 14px;
    padding-bottom: 14px;
    font-weight: 600;
    font-size: 13px;
    color: #38bdf8;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 16px;
    padding: 0 8px;
    background-color: #070a12;
    border-radius: 4px;
    color: #0ea5e9;
    letter-spacing: 0.5px;
}

/* Painéis de Preview / Imagem / Câmera */
QLabel#previewLabel {
    background-color: #05070d;
    border: 1.5px dashed #1e3a5f;
    border-radius: 8px;
    color: #64748b;
    font-size: 12px;
}
QLabel#previewLabel:hover {
    border-color: #0ea5e9;
}

/* Banner de Cabeçalho / Ético */
QGroupBox#headerBox {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0f172a, stop:0.5 #0c213d, stop:1 #0f172a);
    border: 1px solid #0284c7;
    border-radius: 8px;
}

/* Botões Modernos com Gradiente Azul e Efeito Hover */
QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0284c7, stop:1 #0369a1);
    color: #ffffff;
    border: 1px solid #38bdf8;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 600;
    font-size: 13px;
    letter-spacing: 0.3px;
}

QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #38bdf8, stop:1 #0284c7);
    border: 1px solid #7dd3fc;
    color: #ffffff;
}

QPushButton:pressed {
    background-color: #075985;
    border-color: #0284c7;
}

QPushButton:disabled {
    background-color: #1e293b;
    border: 1px solid #334155;
    color: #64748b;
}

/* Botão Secundário / Outline */
QPushButton#secondaryBtn {
    background: transparent;
    color: #38bdf8;
    border: 1px solid #0284c7;
}
QPushButton#secondaryBtn:hover {
    background-color: rgba(14, 165, 233, 0.15);
    border-color: #38bdf8;
}

/* Botão de Cancelar / Alerta */
QPushButton#cancelBtn {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #b91c1c, stop:1 #991b1b);
    border: 1px solid #f87171;
    color: #ffffff;
}
QPushButton#cancelBtn:hover {
    background: #dc2626;
    border-color: #fca5a5;
}

/* Botão de Ação Primária (Verde / Cyan vibrante) */
QPushButton#primaryActionBtn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #059669, stop:1 #0284c7);
    border: 1px solid #34d399;
    color: #ffffff;
    font-size: 14px;
    padding: 10px 20px;
}
QPushButton#primaryActionBtn:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #10b981, stop:1 #38bdf8);
    border: 1px solid #6ee7b7;
}

/* Sistema de Abas (Tabs) Modernas */
QTabWidget::pane {
    border: 1px solid #1e293b;
    background-color: rgba(15, 23, 42, 0.7);
    border-radius: 8px;
    top: -1px;
}

QTabBar::tab {
    background-color: #070d1a;
    border: 1px solid #1e293b;
    border-bottom: none;
    color: #94a3b8;
    padding: 10px 22px;
    font-weight: 600;
    font-size: 13px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    margin-right: 4px;
}

QTabBar::tab:selected {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0c2847, stop:1 #0f1e36);
    color: #38bdf8;
    border: 1px solid #0284c7;
    border-bottom: 2px solid #38bdf8;
}

QTabBar::tab:hover:!selected {
    background-color: #0f172a;
    color: #e2e8f0;
}

/* Campos de Texto (QLineEdit) */
QLineEdit {
    background-color: #070a12;
    border: 1px solid #1e293b;
    border-radius: 6px;
    padding: 6px 12px;
    color: #f8fafc;
    selection-background-color: #0284c7;
}

QLineEdit:focus {
    border: 1px solid #38bdf8;
    background-color: #0a0f1d;
}

/* Dropdowns (QComboBox) */
QComboBox {
    background-color: #070a12;
    border: 1px solid #1e293b;
    border-radius: 6px;
    padding: 6px 12px;
    color: #f8fafc;
    font-weight: 500;
    min-width: 140px;
}

QComboBox:hover {
    border: 1px solid #0ea5e9;
}

QComboBox:focus {
    border: 1px solid #38bdf8;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 25px;
    border-left: 1px solid #1e293b;
}

QComboBox QAbstractItemView {
    background-color: #090e1a;
    border: 1px solid #0284c7;
    selection-background-color: #0284c7;
    selection-color: #ffffff;
    color: #e2e8f0;
    padding: 4px;
}

/* Sliders de Ajuste Fino Futuristas */
QSlider::groove:horizontal {
    height: 6px;
    background: #070a12;
    border: 1px solid #1e293b;
    border-radius: 3px;
}

QSlider::sub-page:horizontal {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0284c7, stop:1 #38bdf8);
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #ffffff;
    border: 2px solid #0284c7;
    width: 16px;
    margin-top: -6px;
    margin-bottom: -6px;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    background: #38bdf8;
    border-color: #ffffff;
}

/* Checkboxes Elegantes */
QCheckBox {
    spacing: 8px;
    color: #e2e8f0;
    font-weight: 500;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border: 1px solid #334155;
    border-radius: 4px;
    background-color: #070a12;
}

QCheckBox::indicator:hover {
    border-color: #0ea5e9;
}

QCheckBox::indicator:checked {
    background-color: #0284c7;
    border: 1px solid #38bdf8;
}

/* Barra de Progresso Futurista */
QProgressBar {
    background-color: #070a12;
    border: 1px solid #1e293b;
    border-radius: 8px;
    height: 20px;
    text-align: center;
    color: #f8fafc;
    font-weight: 600;
    font-size: 11px;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0284c7, stop:0.7 #38bdf8, stop:1 #00f0ff);
    border-radius: 7px;
}

/* Rótulos e Textos de Status */
QLabel {
    color: #cbd5e1;
}

QLabel#statusLabel {
    color: #38bdf8;
    font-weight: 600;
    font-size: 12px;
}

QLabel#liveFpsLabel {
    color: #00f0ff;
    font-weight: bold;
    font-size: 14px;
    font-family: 'Consolas', 'Courier New', monospace;
    background-color: rgba(2, 132, 199, 0.1);
    border: 1px solid rgba(14, 165, 233, 0.3);
    border-radius: 4px;
    padding: 3px 8px;
}

/* Scrollbars */
QScrollBar:vertical {
    border: none;
    background: #070a12;
    width: 8px;
    margin: 0px;
}
QScrollBar::handle:vertical {
    background: #1e293b;
    border-radius: 4px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover {
    background: #0284c7;
}
"""

