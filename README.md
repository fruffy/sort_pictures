# Python Script for Media File Organization

## Description

This Python script sorts image and video files based on creation date, renames dated files according to a specified format, and organizes them into a structured output directory. It includes features for handling duplicates, identifying potentially unwanted files, managing metadata, and ensuring data safety through backups and dry runs.

## Features

* **Automatic Backup:** Creates a timestamped copy of the source directory within the target directory before processing begins.
* **Recursive Processing:** Scans the source directory and its subdirectories for media files.
* **Date Extraction:** Determines file date using the following priority:
    1.  Image EXIF metadata (`DateTimeOriginal`).
    2.  Video file metadata (`creation_date`, etc., using `hachoir`).
    3.  Filename pattern (`YYYYMMDD_HHMMSS`).
* **Configurable Output Structure:** Organizes dated files into subdirectories based on the `--structure` option:
    * `Y`: Year (`YYYY/`) (Default)
    * `YM`: Year and Month (`YYYY/MM/`)
    * `YMD`: Year, Month, and Day (`YYYY/MM/DD/`)
* **Configurable File Renaming:** Renames dated files using a Python `strftime`-compatible format provided via `--filename-format` (Default: `%Y-%m-%d_%H%M%S`).
* **Video File Support:** Processes common video formats and attempts to extract creation dates.
* **EXIF Data Update:** Attempts to write the creation date to image EXIF metadata if the date was determined from the filename and was previously missing.
* **Exact Duplicate Handling:** Skips processing files identified as having identical content (SHA256 hash) to a previously processed file.
* **Near-Duplicate Image Detection:** Identifies visually similar images using perceptual hashing (`imagehash`). Configurable action (`--near-duplicates-action`):
    * `ignore`: No action (Default).
    * `log`: Records sets of similar image paths in the log file.
    * `move`: Moves sets of similar images to subfolders within a `_near_duplicates` review directory.
* **Low-Quality Image Detection:** Identifies images below specified file size (`--min-filesize`) or dimension (`--min-dimension`) thresholds. Configurable action (`--low-quality-action`):
    * `ignore`: No action (Default).
    * `log`: Records paths of flagged images in the log file.
    * `move`: Moves flagged images to a `_low_quality` review directory.
* **Unknown Date Handling:** Moves files without a determinable date to an `unknown_date` subdirectory, preserving original filenames.
* **Dry Run Mode:** The `--dry-run` option simulates all actions (copying, moving, renaming, EXIF writing) without making changes, logging the intended operations instead.
* **Logging:** Records detailed operational information, warnings, and errors to the console and optionally to a specified log file (`--log-file`, `--log-level`).
* **Progress Indication:** Displays a progress bar (`tqdm`) during the file processing stage for larger sets of files.

## Warning: File Operations

**Operation:** This script **MOVES** files from the source directory to the target directory after the initial backup step. It also **MODIFIES** image EXIF data in the source directory before moving if adding a date derived from the filename.

**Backup Recommendation:** The script attempts an automatic backup of the source directory before processing. However, **it is strongly recommended to have an independent, verified backup of your source directory before execution**, as the primary function involves moving and potentially modifying original files. Use the `--dry-run` option to review planned actions before execution.

## Requirements

* Python 3.10 or newer
* Required Python libraries:
    * `Pillow`
    * `piexif`
    * `tqdm`
    * `hachoir`
    * `ImageHash`

## Installation

1.  Confirm Python 3.10+ is installed and accessible.
2.  Install the necessary libraries using pip:

    ```bash
    pip install Pillow piexif tqdm hachoir ImageHash
    ```
    Alternatively, use a virtual environment manager like `uv`:
    ```bash
    uv pip install Pillow piexif tqdm hachoir ImageHash
    ```

## Usage

Execute the script from the command line.

**Syntax:**

```bash
sort_pictures.py --source <SOURCE_PATH> --target <TARGET_PATH> [OPTIONS]
