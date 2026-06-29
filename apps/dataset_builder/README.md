# Dataset Builder

Dataset Builder is a separate desktop and CLI application for collecting portrait datasets from local videos or public YouTube URLs.

It does not belong to `portrait_core`; `portrait_core` remains the analysis core and is used only through public callable APIs when it is available.

## GUI

```powershell
python -m apps.dataset_builder.app
```

The GUI supports:

- local video files: `*.mp4`, `*.avi`, `*.mov`, `*.mkv`
- public YouTube URLs through `yt-dlp`
- frame sampling by frames or seconds
- duplicate filtering by perceptual hash
- `passed/`, `warning/`, `rejected/`, `duplicates/` result folders
- `summary.json` and `log.txt`

## CLI

```powershell
python -m apps.dataset_builder.cli movie.mp4 --output dataset --step-seconds 1
python -m apps.dataset_builder.cli https://youtube.example/watch?v=... --output dataset
```

## Dependencies

```powershell
pip install -r requirements.txt
```

YouTube downloading is limited to publicly available videos. The app does not bypass DRM, does not use paid or private videos, and does not use user accounts.
