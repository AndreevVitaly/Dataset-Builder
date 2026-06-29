from __future__ import annotations

import os
from pathlib import Path

from .settings import BuilderSettings, SourceType, StepMode
from .worker import DatasetBuildPipeline, StopRequested

try:
    from PySide6.QtCore import QObject, QThread, Signal, Slot
    from PySide6.QtWidgets import (
        QApplication,
        QButtonGroup,
        QCheckBox,
        QFileDialog,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QProgressBar,
        QRadioButton,
        QSpinBox,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Для GUI установите PySide6.") from exc


class BuildWorker(QObject):
    log = Signal(str)
    progress = Signal(int)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, settings: BuilderSettings) -> None:
        super().__init__()
        self.settings = settings
        self._stop = False

    @Slot()
    def run(self) -> None:
        try:
            pipeline = DatasetBuildPipeline(
                self.settings,
                log=self.log.emit,
                progress=lambda current, total: self.progress.emit(int(current / max(total, 1) * 100)),
                should_stop=lambda: self._stop,
            )
            self.finished.emit(pipeline.run())
        except StopRequested as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            self.failed.emit(str(exc))

    def stop(self) -> None:
        self._stop = True


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Dataset Builder")
        self.resize(840, 640)
        self.thread: QThread | None = None
        self.worker: BuildWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)

        source_box = QGroupBox("Источник")
        source_layout = QGridLayout(source_box)
        self.file_radio = QRadioButton("Видеофайл")
        self.youtube_radio = QRadioButton("YouTube")
        self.file_radio.setChecked(True)
        self.source_group = QButtonGroup(self)
        self.source_group.addButton(self.file_radio)
        self.source_group.addButton(self.youtube_radio)
        self.file_path = QLineEdit()
        self.file_path.setPlaceholderText("*.mp4, *.avi, *.mov, *.mkv")
        browse = QPushButton("Выбрать файл...")
        browse.clicked.connect(self._choose_file)
        self.url = QLineEdit()
        self.url.setPlaceholderText("https://youtube...")
        source_layout.addWidget(self.file_radio, 0, 0)
        source_layout.addWidget(self.file_path, 0, 1)
        source_layout.addWidget(browse, 0, 2)
        source_layout.addWidget(self.youtube_radio, 1, 0)
        source_layout.addWidget(self.url, 1, 1, 1, 2)
        layout.addWidget(source_box)

        settings_box = QGroupBox("Настройки")
        settings_layout = QFormLayout(settings_box)
        self.frames_radio = QRadioButton("каждые N кадров")
        self.seconds_radio = QRadioButton("каждые N секунд")
        self.seconds_radio.setChecked(True)
        self.step_group = QButtonGroup(self)
        self.step_group.addButton(self.frames_radio)
        self.step_group.addButton(self.seconds_radio)
        self.step_value = QSpinBox()
        self.step_value.setRange(1, 10000)
        self.step_value.setValue(1)
        step_row = QHBoxLayout()
        step_row.addWidget(self.frames_radio)
        step_row.addWidget(self.seconds_radio)
        step_row.addWidget(QLabel("N:"))
        step_row.addWidget(self.step_value)
        settings_layout.addRow("Шаг кадров", step_row)
        layout.addWidget(settings_box)

        self.best_face = QCheckBox("сохранять только лучшее лицо")
        self.best_face.setChecked(True)
        self.remove_duplicates = QCheckBox("удалять одинаковые кадры")
        self.remove_duplicates.setChecked(True)
        self.create_json = QCheckBox("автоматически создавать portrait.json")
        self.create_json.setChecked(True)
        self.delete_temp = QCheckBox("удалить временное видео после анализа")
        self.delete_temp.setChecked(True)
        self.save_rejected = QCheckBox("сохранять rejected")
        self.save_rejected.setChecked(True)
        for checkbox in (self.best_face, self.remove_duplicates, self.create_json, self.delete_temp, self.save_rejected):
            layout.addWidget(checkbox)

        buttons = QHBoxLayout()
        self.start_button = QPushButton("START")
        self.stop_button = QPushButton("STOP")
        self.open_button = QPushButton("Открыть папку результата")
        self.stop_button.setEnabled(False)
        self.start_button.clicked.connect(self._start)
        self.stop_button.clicked.connect(self._stop)
        self.open_button.clicked.connect(self._open_result)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        buttons.addWidget(self.open_button)
        layout.addLayout(buttons)

        self.progress = QProgressBar()
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.progress)
        layout.addWidget(self.log, 1)

        self.setCentralWidget(root)

    def _choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Выбрать видео", "", "Video (*.mp4 *.avi *.mov *.mkv)")
        if path:
            self.file_path.setText(path)
            self.file_radio.setChecked(True)

    def _settings(self) -> BuilderSettings:
        source_type = SourceType.YOUTUBE if self.youtube_radio.isChecked() else SourceType.FILE
        source = self.url.text().strip() if source_type == SourceType.YOUTUBE else self.file_path.text().strip()
        step_mode = StepMode.FRAMES if self.frames_radio.isChecked() else StepMode.SECONDS
        return BuilderSettings(
            source=source,
            source_type=source_type,
            output_dir=Path("dataset"),
            step_mode=step_mode,
            step_value=self.step_value.value(),
            save_best_face_only=self.best_face.isChecked(),
            remove_duplicates=self.remove_duplicates.isChecked(),
            create_portrait_json=self.create_json.isChecked(),
            delete_temp_video=self.delete_temp.isChecked(),
            save_rejected=self.save_rejected.isChecked(),
        )

    def _start(self) -> None:
        settings = self._settings()
        try:
            settings.validate()
        except ValueError as exc:
            QMessageBox.warning(self, "Dataset Builder", str(exc))
            return
        self.progress.setValue(0)
        self.log.clear()
        self.thread = QThread(self)
        self.worker = BuildWorker(settings)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self._append_log)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self._finished)
        self.worker.failed.connect(self._failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.thread.start()

    def _stop(self) -> None:
        if self.worker:
            self.worker.stop()
            self._append_log("Остановка запрошена...")

    def _finished(self, summary) -> None:
        self.progress.setValue(100)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self._append_log(f"Готово: {summary.output_dir}")

    def _failed(self, message: str) -> None:
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self._append_log(message)
        QMessageBox.warning(self, "Dataset Builder", message)

    def _append_log(self, message: str) -> None:
        self.log.append(message)

    def _open_result(self) -> None:
        path = Path("dataset").resolve()
        path.mkdir(exist_ok=True)
        os.startfile(path)
