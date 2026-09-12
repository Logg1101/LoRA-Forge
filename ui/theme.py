"""
Futuristic Dark Workstation Theme (QSS) for LoRA Forge.
Professional AI research software aesthetic: dark glass/metal, restrained cyan/blue/emerald accents,
compact typography, crisp borders, subtle interactive states.
"""

DARK_THEME_QSS = """
/* === GLOBAL RESET & BASE STYLING === */
QWidget {
    background-color: #0b0e14;
    color: #c9d1d9;
    font-family: 'Segoe UI', 'Inter', -apple-system, BlinkMacSystemFont, 'Roboto', sans-serif;
    font-size: 12px;
    selection-background-color: #1f6feb;
    selection-color: #ffffff;
    outline: none;
}

/* === MAIN WINDOW & CENTRAL CONTAINER === */
QMainWindow {
    background-color: #07090e;
}

QScrollArea {
    background: transparent;
    border: none;
}

QScrollArea > QWidget > QWidget {
    background: transparent;
}

/* === TOP BAR === */
#topBar {
    background-color: #0d1117;
    border-bottom: 1px solid #21262d;
    min-height: 52px;
    max-height: 52px;
    padding: 0 16px;
}

#appNameLabel {
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 1.5px;
    color: #58a6ff;
    text-transform: uppercase;
}

#appVersionLabel {
    font-size: 10px;
    font-weight: 600;
    color: #484f58;
    padding-left: 4px;
}

#archBadge {
    background-color: #161b22;
    color: #79c0ff;
    border: 1px solid #30363d;
    border-radius: 4px;
    padding: 3px 8px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
}

#statusIndicator {
    font-size: 12px;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 4px;
    border: 1px solid #30363d;
    background-color: #161b22;
}

#statusIndicator[status="IDLE"] { color: #8b949e; border-color: #30363d; }
#statusIndicator[status="PREPARING"] { color: #d29922; border-color: #bb8009; }
#statusIndicator[status="TRAINING"] { color: #3fb950; border-color: #238636; background-color: #04260f; }
#statusIndicator[status="PAUSED"] { color: #e3b341; border-color: #9e6a03; background-color: #2b1d03; }
#statusIndicator[status="SAVING"] { color: #a371f7; border-color: #8957e5; }
#statusIndicator[status="COMPLETED"] { color: #58a6ff; border-color: #1f6feb; background-color: #071e3d; }
#statusIndicator[status="ERROR"] { color: #f85149; border-color: #da3633; background-color: #310d0d; }
#statusIndicator[status="STOPPED"] { color: #f85149; border-color: #8e1519; }

/* Top bar action buttons */
QPushButton#btnStart {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #238636, stop:1 #2ea043);
    color: #ffffff;
    border: 1px solid #3fb950;
    border-radius: 4px;
    font-weight: 700;
    font-size: 12px;
    padding: 6px 16px;
    letter-spacing: 0.5px;
}
QPushButton#btnStart:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2ea043, stop:1 #3fb950);
}
QPushButton#btnStart:pressed {
    background-color: #196c2e;
}
QPushButton#btnStart:disabled {
    background-color: #21262d;
    border-color: #30363d;
    color: #484f58;
}

QPushButton#btnPause {
    background-color: #161b22;
    color: #e3b341;
    border: 1px solid #9e6a03;
    border-radius: 4px;
    font-weight: 600;
    padding: 6px 12px;
}
QPushButton#btnPause:hover { background-color: #271d05; }
QPushButton#btnPause:disabled { background-color: #161b22; border-color: #21262d; color: #484f58; }

QPushButton#btnResume {
    background-color: #161b22;
    color: #3fb950;
    border: 1px solid #238636;
    border-radius: 4px;
    font-weight: 600;
    padding: 6px 12px;
}
QPushButton#btnResume:hover { background-color: #0d2818; }
QPushButton#btnResume:disabled { background-color: #161b22; border-color: #21262d; color: #484f58; }

QPushButton#btnStop {
    background-color: #161b22;
    color: #f85149;
    border: 1px solid #da3633;
    border-radius: 4px;
    font-weight: 600;
    padding: 6px 12px;
}
QPushButton#btnStop:hover { background-color: #310d0d; }
QPushButton#btnStop:disabled { background-color: #161b22; border-color: #21262d; color: #484f58; }

/* === SIDEBAR === */
#sidebar {
    background-color: #0d1117;
    border-right: 1px solid #21262d;
    min-width: 180px;
    max-width: 180px;
    padding: 10px 0;
}

QPushButton.nav-btn {
    background-color: transparent;
    color: #8b949e;
    border: none;
    border-left: 3px solid transparent;
    text-align: left;
    padding: 10px 16px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
}

QPushButton.nav-btn:hover {
    background-color: #161b22;
    color: #c9d1d9;
}

QPushButton.nav-btn:checked {
    background-color: #161b22;
    color: #58a6ff;
    border-left: 3px solid #58a6ff;
    font-weight: 700;
}

/* === STATUS / TELEMETRY BAR === */
#statusBar {
    background-color: #0d1117;
    border-top: 1px solid #21262d;
    min-height: 34px;
    max-height: 34px;
    padding: 0 16px;
}

#statusMetricLabel {
    font-size: 11px;
    color: #8b949e;
    font-family: 'Consolas', 'Cascadia Code', monospace;
}

#statusMetricValue {
    font-size: 11px;
    font-weight: 600;
    color: #79c0ff;
    font-family: 'Consolas', 'Cascadia Code', monospace;
}

/* === FORM CONTROLS & INPUTS === */
QLineEdit, QTextEdit, QPlainTextEdit {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 4px;
    color: #f0f6fc;
    padding: 5px 8px;
    selection-background-color: #1f6feb;
}

QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
    border: 1px solid #58a6ff;
    background-color: #0d1117;
}

QLineEdit:disabled, QTextEdit:disabled, QPlainTextEdit:disabled {
    background-color: #0d1117;
    color: #484f58;
    border-color: #21262d;
}

QSpinBox, QDoubleSpinBox {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 4px;
    color: #f0f6fc;
    padding: 4px 8px;
}

QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #58a6ff;
}

QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {
    background-color: #21262d;
    border: none;
    width: 16px;
}

QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background-color: #30363d;
}

QComboBox {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 4px;
    color: #f0f6fc;
    padding: 4px 10px;
    min-height: 20px;
}

QComboBox:focus, QComboBox:hover {
    border: 1px solid #58a6ff;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QComboBox QAbstractItemView {
    background-color: #161b22;
    border: 1px solid #30363d;
    color: #f0f6fc;
    selection-background-color: #1f6feb;
    selection-color: #ffffff;
    padding: 4px;
}

QCheckBox {
    color: #c9d1d9;
    spacing: 8px;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 3px;
}

QCheckBox::indicator:hover {
    border-color: #58a6ff;
}

QCheckBox::indicator:checked {
    background-color: #1f6feb;
    border-color: #58a6ff;
    image: none;
}

QSlider::groove:horizontal {
    height: 4px;
    background: #21262d;
    border-radius: 2px;
}

QSlider::sub-page:horizontal {
    background: #1f6feb;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: #58a6ff;
    border: 1px solid #79c0ff;
    width: 14px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 7px;
}

QSlider::handle:horizontal:hover {
    background: #79c0ff;
}

/* === BUTTONS === */
QPushButton {
    background-color: #21262d;
    color: #c9d1d9;
    border: 1px solid #30363d;
    border-radius: 4px;
    padding: 6px 14px;
    font-weight: 600;
}

QPushButton:hover {
    background-color: #30363d;
    border-color: #8b949e;
    color: #f0f6fc;
}

QPushButton:pressed {
    background-color: #161b22;
}

QPushButton:disabled {
    background-color: #0d1117;
    border-color: #21262d;
    color: #484f58;
}

QPushButton.primary-btn {
    background-color: #1f6feb;
    color: #ffffff;
    border-color: #58a6ff;
}

QPushButton.primary-btn:hover {
    background-color: #388bfd;
}

/* Browse Button Mini */
QPushButton.browse-btn {
    background-color: #161b22;
    border: 1px solid #30363d;
    padding: 5px 12px;
    font-size: 11px;
}

/* === SECTION & CARD HEADERS === */
QLabel.section-title {
    color: #58a6ff;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    padding-bottom: 2px;
}

QLabel.section-desc {
    color: #8b949e;
    font-size: 11px;
    padding-bottom: 8px;
}

QLabel.field-label {
    color: #c9d1d9;
    font-size: 11px;
    font-weight: 600;
}

QLabel.meta-badge {
    background-color: #161b22;
    border: 1px solid #21262d;
    border-radius: 3px;
    color: #8b949e;
    padding: 2px 6px;
    font-size: 10px;
    font-family: 'Consolas', 'Cascadia Code', monospace;
}

/* Section divider */
QFrame.separator {
    background-color: #21262d;
    border: none;
    max-height: 1px;
    margin: 12px 0;
}

/* Compact Group Container */
QFrame.group-box {
    background-color: #0d1117;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 12px;
}

/* Mini Progress Bar */
QProgressBar {
    background-color: #161b22;
    border: 1px solid #21262d;
    border-radius: 3px;
    height: 12px;
    text-align: center;
    font-size: 9px;
    color: #c9d1d9;
    font-weight: 600;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1f6feb, stop:1 #58a6ff);
    border-radius: 2px;
}

/* Table / Tree / List */
QTableWidget, QTreeWidget, QListWidget {
    background-color: #0d1117;
    border: 1px solid #21262d;
    border-radius: 4px;
    gridline-color: #161b22;
    color: #c9d1d9;
    font-size: 11px;
}

QTableWidget::item, QTreeWidget::item, QListWidget::item {
    padding: 4px 8px;
}

QTableWidget::item:selected, QTreeWidget::item:selected, QListWidget::item:selected {
    background-color: #1f6feb;
    color: #ffffff;
}

QHeaderView::section {
    background-color: #161b22;
    color: #8b949e;
    border: none;
    border-bottom: 1px solid #21262d;
    border-right: 1px solid #21262d;
    padding: 4px 8px;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
}

/* Scrollbars */
QScrollBar:vertical {
    background-color: #07090e;
    width: 8px;
    margin: 0;
}

QScrollBar::handle:vertical {
    background-color: #21262d;
    min-height: 20px;
    border-radius: 4px;
}

QScrollBar::handle:vertical:hover {
    background-color: #30363d;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    background-color: #07090e;
    height: 8px;
    margin: 0;
}

QScrollBar::handle:horizontal {
    background-color: #21262d;
    min-width: 20px;
    border-radius: 4px;
}

QScrollBar::handle:horizontal:hover {
    background-color: #30363d;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

/* Tooltips */
QToolTip {
    background-color: #161b22;
    color: #f0f6fc;
    border: 1px solid #30363d;
    border-radius: 4px;
    padding: 6px;
    font-size: 11px;
}

/* Tab Widget */
QTabWidget::pane {
    border: 1px solid #21262d;
    background-color: #0d1117;
    border-radius: 4px;
}

QTabBar::tab {
    background-color: #161b22;
    color: #8b949e;
    border: 1px solid #21262d;
    border-bottom: none;
    padding: 6px 14px;
    font-size: 11px;
    font-weight: 600;
    margin-right: 2px;
}

QTabBar::tab:selected {
    background-color: #0d1117;
    color: #58a6ff;
    border-top: 2px solid #58a6ff;
}

QTabBar::tab:hover:!selected {
    background-color: #21262d;
    color: #c9d1d9;
}
"""
