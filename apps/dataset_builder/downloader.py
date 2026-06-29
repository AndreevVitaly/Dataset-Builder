from __future__ import annotations

from pathlib import Path
from typing import Callable


ProgressCallback = Callable[[str, float | None], None]


def download_youtube_video(url: str, temp_dir: Path, progress: ProgressCallback | None = None) -> Path:
    """Download a public YouTube video with yt-dlp and return the local file path."""
    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:
        raise RuntimeError("Для YouTube-источников установите yt-dlp.") from exc

    temp_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(temp_dir / "%(id)s.%(ext)s")
    downloaded: dict[str, Path] = {}

    def hook(status: dict) -> None:
        if status.get("status") == "downloading":
            total = status.get("total_bytes") or status.get("total_bytes_estimate")
            done = status.get("downloaded_bytes", 0)
            percent = (done / total * 100.0) if total else None
            if progress:
                progress("Скачивание видео", percent)
        elif status.get("status") == "finished":
            filename = status.get("filename")
            if filename:
                downloaded["path"] = Path(filename)
            if progress:
                progress("Видео скачано", 100.0)

    options = {
        "outtmpl": output_template,
        "format": "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/best",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [hook],
    }

    with YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        if downloaded.get("path"):
            return downloaded["path"]
        path = Path(filename)
        if path.exists():
            return path
        mp4_path = path.with_suffix(".mp4")
        if mp4_path.exists():
            return mp4_path

    raise RuntimeError("Не удалось определить путь скачанного видео.")
