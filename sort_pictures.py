#!/usr/bin/env python3

import argparse
import hashlib
import logging
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Set, List, Tuple, Dict, Any

# External dependencies
try:
    from PIL import Image

    # Pillow透视变换需要此模块
    # from PIL import PerspectiveTransform # Example if specific PIL modules needed later
    Image.MAX_IMAGE_PIXELS = None  # Allow Pillow to load large images
    from PIL.ExifTags import TAGS
    import piexif
    from tqdm import tqdm
    from hachoir.parser import createParser
    from hachoir.metadata import extractMetadata
    import imagehash  # For near-duplicate detection
except ImportError as e:
    print(f"Error: Missing required libraries. Please install them.", file=sys.stderr)
    print(f"Missing: {e.name}", file=sys.stderr)
    print("Run: pip install Pillow piexif tqdm hachoir ImageHash", file=sys.stderr)
    sys.exit(1)

# --- Constants ---
UNKNOWN_DATE_FOLDER_NAME: str = "unknown_date"
NEAR_DUPLICATES_FOLDER_NAME: str = "_near_duplicates"
LOW_QUALITY_FOLDER_NAME: str = "_low_quality"

FILENAME_DATE_REGEX: re.Pattern = re.compile(r".*?(\d{8})_(\d{6}).*?")
IMAGE_EXTENSIONS: Set[str] = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".heic",
    ".webp",
    ".gif",
    ".bmp",
}
VIDEO_EXTENSIONS: Set[str] = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".mpg",
    ".mpeg",
    ".wmv",
    ".m4v",
    ".3gp",
}
SUPPORTED_EXTENSIONS: Set[str] = IMAGE_EXTENSIONS.union(VIDEO_EXTENSIONS)

# EXIF Tags
EXIF_DATETIME_ORIGINAL_TAG: int = 36867
EXIF_DATETIME_DIGITIZED_TAG: int = 36868
EXIF_DATETIME_TAG: int = 306


# --- Configuration Class ---
class Config:
    """Holds script configuration."""

    def __init__(self, args: argparse.Namespace):
        self.source_dir: Path = args.source.resolve()
        self.target_dir: Path = args.target.resolve()
        self.structure: str = args.structure
        self.filename_format: str = args.filename_format
        self.dry_run: bool = args.dry_run
        self.log_level: str = args.log_level
        self.log_file: Optional[Path] = Path(args.log_file) if args.log_file else None
        self.halt_on_error: bool = args.halt_on_error

        # New options
        self.near_duplicates_action: str = args.near_duplicates_action
        self.low_quality_action: str = args.low_quality_action
        self.min_filesize_kb: int = args.min_filesize
        self.min_dimension: int = args.min_dimension

        # Derived paths
        self.unknown_dir_path: Path = self.target_dir / UNKNOWN_DATE_FOLDER_NAME
        self.near_duplicates_path: Path = self.target_dir / NEAR_DUPLICATES_FOLDER_NAME
        self.low_quality_path: Path = self.target_dir / LOW_QUALITY_FOLDER_NAME
        self.backup_dir: Optional[Path] = None

        # Runtime state for near duplicates
        self.perceptual_hashes: Dict[str, List[Tuple[Path, bool]]] = (
            {}
        )  # hash -> List[(filepath, moved_to_review)]


# --- Logging Setup ---
def setup_logging(config: Config):
    """Configures logging to file and console."""
    handlers: List[logging.Handler] = []
    log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    numeric_log_level = getattr(logging, config.log_level.upper(), logging.INFO)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(log_format)
    console_handler.setLevel(numeric_log_level)
    handlers.append(console_handler)

    if config.log_file:
        try:
            config.log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(config.log_file, mode="a")
            file_handler.setFormatter(log_format)
            file_handler.setLevel(logging.DEBUG)
            handlers.append(file_handler)
            print(f"Logging detailed output to: {config.log_file}")
        except OSError as e:
            print(
                f"Warning: Could not create log file '{config.log_file}': {e}. Logging to console only.",
                file=sys.stderr,
            )

    logging.basicConfig(level=logging.DEBUG, handlers=handlers)


# --- Helper Functions ---


# (get_exif_datetime, parse_filename_datetime, set_exif_datetime,
#  calculate_hash, get_video_datetime remain largely the same as before,
#  but ensure they use logging instead of print for errors/warnings)
# --- Minor logging adjustments in helpers ---
def get_exif_datetime(filepath: Path) -> Optional[datetime]:
    # ... (previous code) ...
    # Replace print warnings/errors with logging.warning/logging.error
    # Example change:
    # except Exception as e:
    #     logging.warning(f"Could not read EXIF from image {filepath.name}: {type(e).__name__}", exc_info=False)
    #     return None
    # ... (rest of the function)
    # --- [Code from previous version, replace prints with logging] ---
    try:
        img = Image.open(filepath)
        try:
            exif_data = img.getexif()
        except AttributeError:
            exif_dict = getattr(img, "info", {}).get("exif")
            if exif_dict:
                try:
                    exif_data = piexif.load(exif_dict)
                except Exception:
                    exif_data = None
            else:
                exif_data = None
        if not exif_data:
            return None
        date_tags_priority = [
            EXIF_DATETIME_ORIGINAL_TAG,
            EXIF_DATETIME_DIGITIZED_TAG,
            EXIF_DATETIME_TAG,
        ]
        date_str: Optional[str] = None
        for tag_id in date_tags_priority:
            tag_val = None
            if isinstance(exif_data, dict):
                ifd_name = (
                    "Exif"
                    if tag_id
                    in [EXIF_DATETIME_ORIGINAL_TAG, EXIF_DATETIME_DIGITIZED_TAG]
                    else "0th"
                )
                ifd = exif_data.get(ifd_name, {})
                tag_val = ifd.get(tag_id)
            elif hasattr(exif_data, "get"):
                tag_val = exif_data.get(tag_id)
            if tag_val:
                if isinstance(tag_val, bytes):
                    date_str = tag_val.decode("utf-8", errors="ignore").strip()
                else:
                    date_str = str(tag_val).strip()
                if date_str:
                    break
        if not date_str:
            return None
        date_str = date_str.replace("\x00", "")
        for fmt in ("%Y:%m:%d %H:%M:%S", "%Y %m %d %H:%M:%S"):
            try:
                return datetime.strptime(date_str, fmt)
            except (ValueError, TypeError):
                continue
        logging.debug(
            f"Could not parse potential EXIF date string '{date_str}' for {filepath.name}"
        )
        return None
    except FileNotFoundError:
        logging.error(f"File not found during EXIF read: {filepath}")
        return None
    except Exception as e:
        logging.warning(
            f"Could not read EXIF from image {filepath.name}: {type(e).__name__}",
            exc_info=False,
        )
        return None


def get_video_datetime(filepath: Path) -> Optional[datetime]:
    # ... (previous code) ...
    # Replace print warnings/errors with logging.warning/logging.error
    # Example change:
    # except Exception as e:
    #     logging.warning(f"Hachoir failed to process video {filepath.name}: {type(e).__name__}", exc_info=False)
    #     return None
    # ... (rest of the function)
    # --- [Code from previous version, replace prints with logging] ---
    parser = None
    try:
        parser = createParser(str(filepath))
        if not parser:
            logging.warning(
                f"Hachoir could not create parser for video: {filepath.name}"
            )
            return None
        with parser:
            metadata = extractMetadata(parser)
        if not metadata:
            logging.warning(
                f"Hachoir could not extract metadata from video: {filepath.name}"
            )
            return None
        date_keys_priority = [
            "creation_date",
            "last_modification",
            "date_time_original",
            "media_create_date",
            "track_create_date",
        ]
        dt = None
        for key in date_keys_priority:
            if metadata.has(key):
                try:
                    val = metadata.get(key)
                    if isinstance(val, list):
                        val = val[0]
                    if isinstance(val, datetime):
                        dt = val
                        logging.debug(
                            f"Found video date '{key}': {dt} in {filepath.name}"
                        )
                        break
                except Exception as e:
                    logging.debug(
                        f"Error processing metadata key '{key}' for {filepath.name}: {e}"
                    )
                    continue
        return dt
    except FileNotFoundError:
        logging.error(f"File not found during video metadata read: {filepath}")
        return None
    except Exception as e:
        logging.warning(
            f"Hachoir failed to process video {filepath.name}: {type(e).__name__}",
            exc_info=False,
        )
        return None
    finally:
        pass


def parse_filename_datetime(filename: str) -> Optional[datetime]:
    # --- [Code from previous version, replace prints with logging] ---
    match = FILENAME_DATE_REGEX.match(filename)
    if match:
        date_str, time_str = match.group(1), match.group(2)
        try:
            return datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
        except ValueError:
            logging.debug(
                f"Filename pattern matched in '{filename}' but failed parsing."
            )
            return None
    return None


def set_exif_datetime(filepath: Path, dt: datetime, config: Config) -> bool:
    # --- [Code from previous version, uses logging and config.dry_run] ---
    if config.dry_run:
        logging.info(
            f"DRY RUN: Would add/update EXIF date ({dt.strftime('%Y:%m:%d %H:%M:%S')}) to {filepath.name}"
        )
        return True
    try:
        exif_dt_str: str = dt.strftime("%Y:%m:%d %H:%M:%S")
        exif_dict = None
        try:
            exif_dict = piexif.load(str(filepath))
        except piexif.InvalidImageDataError:
            logging.debug(f"No existing EXIF data in {filepath.name}. Creating new.")
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
        except ValueError as e:
            logging.warning(
                f"piexif could not load EXIF from {filepath.name} (ValueError: {e})."
            )
            return False
        except Exception as load_err:
            logging.warning(
                f"Could not load EXIF for writing ({filepath.name}): {type(load_err).__name__}."
            )
            return False
        if "Exif" not in exif_dict:
            exif_dict["Exif"] = {}
        exif_dict["Exif"][EXIF_DATETIME_ORIGINAL_TAG] = exif_dt_str.encode("utf-8")
        exif_dict["Exif"][EXIF_DATETIME_DIGITIZED_TAG] = exif_dt_str.encode("utf-8")
        if "0th" not in exif_dict:
            exif_dict["0th"] = {}
        exif_dict["0th"][EXIF_DATETIME_TAG] = exif_dt_str.encode("utf-8")
        try:
            exif_bytes = piexif.dump(exif_dict)
        except Exception as dump_err:
            logging.error(
                f"Failed to dump modified EXIF for {filepath.name}: {dump_err}"
            )
            return False
        piexif.insert(exif_bytes, str(filepath))
        logging.debug(f"Successfully added/updated EXIF date in {filepath.name}")
        return True
    except FileNotFoundError:
        logging.error(f"File not found writing EXIF: {filepath}")
        return False
    except PermissionError:
        logging.error(f"Permission denied writing EXIF to {filepath}")
        return False
    except Exception as e:
        logging.error(
            f"Failed to write EXIF to {filepath.name}: {type(e).__name__}",
            exc_info=False,
        )
        return False


def calculate_hash(filepath: Path) -> Optional[str]:
    # --- [Code from previous version, uses logging] ---
    hasher = hashlib.sha256()
    try:
        with filepath.open("rb") as file:
            buf = file.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = file.read(65536)
        return hasher.hexdigest()
    except OSError as e:
        logging.error(f"Could not read file for hashing {filepath.name}: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error hashing {filepath.name}: {e}")
        return None


def calculate_perceptual_hash(filepath: Path) -> Optional[str]:
    """Calculates perceptual hash (phash) for an image file."""
    try:
        img = Image.open(filepath)
        # Convert to L (grayscale) for phash consistency if needed? Docs say it handles color.
        # img = img.convert('L')
        phash = imagehash.phash(img)
        return str(phash)
    except FileNotFoundError:
        logging.error(f"File not found during perceptual hash calculation: {filepath}")
        return None
    except Exception as e:
        logging.warning(
            f"Could not calculate perceptual hash for {filepath.name}: {type(e).__name__}",
            exc_info=False,
        )
        return None


def check_low_quality(filepath: Path, config: Config) -> bool:
    """Checks if an image file meets low-quality criteria."""
    if config.min_filesize_kb <= 0 and config.min_dimension <= 0:
        return False  # Checks disabled

    try:
        # Check filesize
        if config.min_filesize_kb > 0:
            size_kb = filepath.stat().st_size / 1024
            if size_kb < config.min_filesize_kb:
                logging.debug(
                    f"Flagged {filepath.name} as low quality: Size {size_kb:.1f}KB < {config.min_filesize_kb}KB"
                )
                return True

        # Check dimensions
        if config.min_dimension > 0:
            try:
                with Image.open(filepath) as img:
                    width, height = img.size
                if width < config.min_dimension or height < config.min_dimension:
                    logging.debug(
                        f"Flagged {filepath.name} as low quality: Dimensions {width}x{height} < {config.min_dimension}px"
                    )
                    return True
            except Exception as e:
                logging.warning(
                    f"Could not read dimensions for quality check {filepath.name}: {type(e).__name__}",
                    exc_info=False,
                )
                # Treat as not low quality if dimensions can't be read? Or flag it? Let's not flag.
                return False

    except OSError as e:
        logging.error(f"Could not stat file for quality check {filepath.name}: {e}")
        return False  # Cannot determine quality

    return False  # Did not meet low quality criteria


def create_target_folder(folder_path: Path, config: Config) -> bool:
    # --- [Code from previous version, uses logging] ---
    if folder_path.exists():
        return True
    if config.dry_run:
        logging.info(f"DRY RUN: Would create directory: {folder_path}")
        return True
    try:
        folder_path.mkdir(parents=True, exist_ok=True)
        logging.debug(f"Created directory: {folder_path}")
        return True
    except OSError as e:
        logging.error(f"Could not create target folder {folder_path}: {e}")
        return False


def move_file(
    source_path: Path, target_path: Path, config: Config, action_type="Move"
) -> bool:
    """Moves the file from source to target path, handling dry run and logging."""
    relative_target_path = target_path
    try:  # Try making path relative for cleaner logging
        relative_target_path = target_path.relative_to(config.target_dir.parent)
    except ValueError:
        pass  # Keep absolute if not relative

    if config.dry_run:
        logging.info(
            f"DRY RUN: Would {action_type.upper()} '{source_path.name}' to '{relative_target_path}'"
        )
        return True
    try:
        # Ensure parent directory exists before moving (important for review folders)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_path), str(target_path))
        logging.debug(
            f"{action_type}d '{source_path.name}' to '{relative_target_path}'"
        )
        return True
    except Exception as e:
        logging.error(
            f"{action_type} failed for {source_path.name} -> {target_path}: {type(e).__name__}",
            exc_info=False,
        )
        return False


def handle_collision(
    target_filepath: Path,
    base_name: str,
    suffix: str,
    file_hash: Optional[str],
    config: Config,
) -> Optional[Path]:
    """Handles filename collisions, returns new path or None if skipped/error."""
    counter = 1
    while target_filepath.exists():
        existing_hash = calculate_hash(target_filepath)
        if file_hash and existing_hash and file_hash == existing_hash:
            relative_target_path = target_filepath
            try:
                relative_target_path = target_filepath.relative_to(
                    config.target_dir.parent
                )
            except ValueError:
                pass
            logging.debug(
                f"Skipping identical file already exists: {relative_target_path}"
            )
            return None  # Indicate skip

        new_filename = f"{base_name}_{counter}{suffix}"
        target_filepath = target_filepath.parent / new_filename
        counter += 1
        if counter > 100:
            logging.error(
                f"Too many collisions for base '{base_name}' in {target_filepath.parent.name}. Skipping."
            )
            return None  # Indicate error/skip

    return target_filepath


# --- Main Processing Function ---
def process_file(
    item: Path, config: Config, processed_hashes: Set[str]
) -> Tuple[bool, bool, bool, bool, Optional[str]]:
    """Processes a single file: quality check, dupe check, gets date, determines target, handles collisions & moves."""
    # Returns: moved_successfully, duplicate_skipped, exif_added, error_occurred, file_hash
    error_occurred = False
    exif_added = False
    is_image = item.suffix.lower() in IMAGE_EXTENSIONS
    is_video = item.suffix.lower() in VIDEO_EXTENSIONS

    logging.debug(f"Processing item: {item}")

    # === Pre-checks (Move to Review Folders) ===

    # 1. Low Quality Check (Images Only)
    if is_image and config.low_quality_action != "ignore":
        is_low_quality = check_low_quality(item, config)
        if is_low_quality:
            if config.low_quality_action == "log":
                logging.debug(f"Low quality detected (logged): {item.name}")
                # Continue processing for date sorting etc. if only logging
            elif config.low_quality_action == "move":
                if not create_target_folder(config.low_quality_path, config):
                    return (
                        False,
                        False,
                        False,
                        True,
                        None,
                    )  # Cannot proceed if folder fails
                # Move to low quality folder with collision handling
                lq_target_path = handle_collision(
                    config.low_quality_path / item.name,
                    item.stem,
                    item.suffix,
                    None,
                    config,
                )  # No hash check needed here
                if lq_target_path:
                    if move_file(
                        item, lq_target_path, config, action_type="Move (Low Quality)"
                    ):
                        return (
                            False,
                            False,
                            False,
                            False,
                            None,
                        )  # Moved to review, not sorted/error
                    else:
                        return False, False, False, True, None  # Move failed
                else:
                    # Collision handling failed or skipped identical name (unlikely without hash check)
                    return False, False, False, True, None  # Treat as error

    # 2. Near-Duplicate Check (Images Only)
    perceptual_hash: Optional[str] = None
    if is_image and config.near_duplicates_action != "ignore":
        perceptual_hash = calculate_perceptual_hash(item)
        if perceptual_hash:
            seen_files = config.perceptual_hashes.get(perceptual_hash, [])
            if seen_files:  # This hash has been seen before
                if config.near_duplicates_action == "move":
                    logging.debug(
                        f"Near-duplicate detected for {item.name} (hash: {perceptual_hash}). Moving to review."
                    )
                    if not create_target_folder(config.near_duplicates_path, config):
                        return False, False, False, True, None

                    # Create subfolder for this duplicate set, named by hash
                    dupe_set_folder = config.near_duplicates_path / perceptual_hash
                    if not create_target_folder(dupe_set_folder, config):
                        return False, False, False, True, None

                    # Move the *first* file seen for this hash if it wasn't moved already
                    first_file_path, first_file_moved = seen_files[0]
                    if not first_file_moved:
                        # Need collision handling when moving the first file too
                        first_target_path = handle_collision(
                            dupe_set_folder / first_file_path.name,
                            first_file_path.stem,
                            first_file_path.suffix,
                            None,
                            config,
                        )
                        if first_target_path:
                            if move_file(
                                first_file_path,
                                first_target_path,
                                config,
                                action_type="Move (Near Dupe Set)",
                            ):
                                seen_files[0] = (first_file_path, True)  # Mark as moved
                            else:
                                error_occurred = True  # Log error but continue to move current file if possible
                        else:
                            error_occurred = True

                    # Move the *current* file to the same folder
                    current_target_path = handle_collision(
                        dupe_set_folder / item.name,
                        item.stem,
                        item.suffix,
                        None,
                        config,
                    )
                    if current_target_path:
                        if move_file(
                            item,
                            current_target_path,
                            config,
                            action_type="Move (Near Dupe)",
                        ):
                            config.perceptual_hashes[perceptual_hash].append(
                                (item, True)
                            )  # Add current file as moved
                            return (
                                False,
                                False,
                                False,
                                error_occurred,
                                None,
                            )  # Moved to review, not sorted
                        else:
                            error_occurred = True  # Failed to move current file
                            # Add to list anyway, maybe for logging later? Mark as not moved.
                            config.perceptual_hashes.setdefault(
                                perceptual_hash, []
                            ).append((item, False))
                            return (
                                False,
                                False,
                                False,
                                True,
                                None,
                            )  # Failed move is error
                    else:
                        # Collision handling failed for current file
                        config.perceptual_hashes.setdefault(perceptual_hash, []).append(
                            (item, False)
                        )
                        return False, False, False, True, None

                # If action is 'log' or 'ignore', just record it for later summary/logging
                config.perceptual_hashes.setdefault(perceptual_hash, []).append(
                    (item, False)
                )
                # Continue processing for date sorting if not moved

            else:  # First time seeing this perceptual hash
                config.perceptual_hashes[perceptual_hash] = [
                    (item, False)
                ]  # Add as not moved yet
        else:
            error_occurred = True  # Failed to calculate perceptual hash
            if config.halt_on_error:
                return False, False, False, True, None

    # === Date Sorting (If not moved to review) ===

    # 3. Exact Duplicate Check (Content Hash)
    file_hash = calculate_hash(item)
    if file_hash:
        if file_hash in processed_hashes:
            logging.debug(f"Skipping exact duplicate content: {item.name}")
            return False, True, False, False, file_hash
    else:
        logging.warning(
            f"Could not hash file {item.name}, cannot check exact duplicates."
        )
        error_occurred = True
        if config.halt_on_error:
            return False, False, False, True, None

    # 4. Determine Date (EXIF > Video Meta > Filename)
    file_datetime: Optional[datetime] = None
    date_source: str = "Unknown"

    if is_image:
        file_datetime = get_exif_datetime(item)
        if file_datetime:
            date_source = "EXIF"
    elif is_video:
        file_datetime = get_video_datetime(item)
        if file_datetime:
            date_source = "Video Meta"

    if not file_datetime:
        file_datetime = parse_filename_datetime(item.name)
        if file_datetime:
            date_source = "Filename"
            # 5. Attempt Add EXIF if from Filename (Images Only)
            if is_image:
                logging.debug(
                    f"Date from filename ({file_datetime.strftime(config.filename_format)}) for {item.name}. Adding EXIF..."
                )
                if set_exif_datetime(item, file_datetime, config):
                    exif_added = True
                else:
                    logging.warning(
                        f"Failed add EXIF derived from filename for {item.name}."
                    )
                    error_occurred = True
                    if config.halt_on_error:
                        return False, False, exif_added, True, file_hash

    # 6. Determine Target Path & New Filename
    target_folder: Path
    target_filename_base: str
    target_filename: str

    if file_datetime:
        year_str = str(file_datetime.year)
        if config.structure == "Y":
            target_folder = config.target_dir / year_str
        elif config.structure == "YM":
            target_folder = config.target_dir / year_str / f"{file_datetime.month:02d}"
        elif config.structure == "YMD":
            target_folder = (
                config.target_dir
                / year_str
                / f"{file_datetime.month:02d}"
                / f"{file_datetime.day:02d}"
            )
        else:
            target_folder = config.target_dir / year_str

        try:
            target_filename_base = file_datetime.strftime(config.filename_format)
        except ValueError:
            logging.warning(
                f"Invalid strftime format '{config.filename_format}'. Using default."
            )
            target_filename_base = file_datetime.strftime("%Y-%m-%d_%H%M%S")  # Fallback

        target_filename = f"{target_filename_base}{item.suffix}"
        logging.debug(
            f"Date found ({date_source}). Target: {target_folder.relative_to(config.target_dir.parent)}/{target_filename}"
        )

    else:  # No date found
        target_folder = config.unknown_dir_path
        target_filename_base = item.stem
        target_filename = item.name
        logging.debug(f"No date found. Target: {target_folder.name}/{target_filename}")

    # Ensure target folder exists
    if not create_target_folder(target_folder, config):
        return False, False, exif_added, True, file_hash  # Error occurred

    # 7. Handle Filename Collisions (using exact hash)
    target_filepath = handle_collision(
        target_folder / target_filename,
        target_filename_base,
        item.suffix,
        file_hash,
        config,
    )

    if not target_filepath:
        # Collision check skipped (identical file exists) or failed (too many collisions)
        is_duplicate = (
            target_filepath is None and file_hash is not None
        )  # Check if it was skipped due to duplicate hash
        return (
            False,
            is_duplicate,
            exif_added,
            not is_duplicate,
            file_hash,
        )  # Not moved, maybe duplicate, maybe error

    # 8. Move the File
    if move_file(item, target_filepath, config, action_type="Move (Sorted)"):
        return True, False, exif_added, False, file_hash  # Moved successfully
    else:
        # Move failed, error logged in move_file
        return False, False, exif_added, True, file_hash  # Not moved, error occurred


# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(
        description="Sorts image/video files by date, detects duplicates & low quality.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Paths
    parser.add_argument(
        "-s", "--source", type=Path, required=True, help="Source directory."
    )
    parser.add_argument(
        "-t",
        "--target",
        type=Path,
        required=True,
        help="Target directory for sorted files, review folders, and backup.",
    )
    # Structure & Naming
    parser.add_argument(
        "--structure",
        choices=["Y", "YM", "YMD"],
        default="Y",
        help="Output folder structure for dated files.",
    )
    parser.add_argument(
        "--filename-format",
        default="%Y-%m-%d_%H%M%S",
        help="strftime format for renaming dated files.",
    )
    # Modes & Logging
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate actions without modifying files.",
    )
    parser.add_argument(
        "--log-file", default="photo_sorter.log", help="Path to log file ('': disable)."
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Console logging level.",
    )
    parser.add_argument(
        "--halt-on-error",
        action="store_true",
        help="Stop processing a file if a non-critical error occurs (hash, exif add, etc.).",
    )
    # New Features
    parser.add_argument(
        "--near-duplicates-action",
        choices=["ignore", "log", "move"],
        default="move",
        help="Action for visually similar images: ignore, log, or move to '_near_duplicates' folder.",
    )
    parser.add_argument(
        "--low-quality-action",
        choices=["ignore", "log", "move"],
        default="move",
        help="Action for low quality images: ignore, log, or move to '_low_quality' folder.",
    )
    parser.add_argument(
        "--min-filesize",
        type=int,
        default=0,
        help="Min filesize (KB) to avoid 'low quality' flag (0=disabled).",
    )
    parser.add_argument(
        "--min-dimension",
        type=int,
        default=0,
        help="Min width/height (px) to avoid 'low quality' flag (0=disabled).",
    )

    args = parser.parse_args()
    config = Config(args)
    setup_logging(config)

    # --- Validate Input/Output ---
    # [Previous validation code - remains the same]
    if not config.source_dir.is_dir():
        logging.critical(f"Source directory '{config.source_dir}' not found.")
        sys.exit(1)
    if config.target_dir.exists() and not config.target_dir.is_dir():
        logging.critical(
            f"Target path '{config.target_dir}' exists but is not a directory."
        )
        sys.exit(1)
    if config.source_dir == config.target_dir:
        logging.critical(f"Source and target directories cannot be the same.")
        sys.exit(1)
    try:
        if (
            config.target_dir.resolve() in config.source_dir.resolve().parents
            or config.target_dir.resolve() == config.source_dir.resolve()
        ):
            logging.critical(
                f"Target '{config.target_dir}' cannot be inside source '{config.source_dir}'."
            )
            sys.exit(1)
    except Exception as e:
        logging.warning(f"Path relationship check failed: {e}")

    # --- Setup Target Folders ---
    # [Previous setup code - remains the same]
    if not create_target_folder(config.target_dir, config):
        sys.exit(1)
    if not create_target_folder(config.unknown_dir_path, config):
        logging.warning("Proceeding without 'unknown_date' folder.")
    # Also create review folders now if needed by config, ignore errors for now
    if config.low_quality_action == "move":
        create_target_folder(config.low_quality_path, config)
    if config.near_duplicates_action == "move":
        create_target_folder(config.near_duplicates_path, config)

    # --- Automatic Backup Step ---
    # [Previous backup code - remains the same]
    backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.backup_dir = config.target_dir / f"_source_backup_{backup_timestamp}"
    logging.warning(
        "=" * 60
        + "\n>>> IMPORTANT: Automatic Backup <<<\n"
        + f" Attempting backup: '{config.source_dir}' -> '{config.backup_dir}'\n"
        + "=" * 60
    )
    if config.dry_run:
        logging.info(f"DRY RUN: Would backup source to '{config.backup_dir}'")
    else:
        try:
            logging.info(
                f"Starting backup copy at {datetime.now().strftime('%H:%M:%S')}..."
            )
            shutil.copytree(
                str(config.source_dir),
                str(config.backup_dir),
                symlinks=False,
                ignore_dangling_symlinks=True,
            )
            logging.info(
                f"Backup completed successfully at {datetime.now().strftime('%H:%M:%S')} to '{config.backup_dir}'."
            )
        except Exception as e:
            logging.critical(
                f"*** CRITICAL ERROR: Backup failed! *** Reason: {e}", exc_info=True
            )
            sys.exit(1)

    # --- Final Warning ---
    # [Previous warning code - remains the same]
    logging.warning(
        "=" * 60
        + "\n>>> WARNING: Source File Modification <<<\n Backup step completed. Script will now MOVE and MODIFY files.\n Ensure you understand these actions.\n"
        + "=" * 60
    )
    try:
        input("Press Enter to continue, or Ctrl+C to abort...")
    except KeyboardInterrupt:
        logging.info("Operation aborted by user.")
        sys.exit(0)

    # --- Collect Files ---
    # [Previous collection code - remains the same]
    logging.info(f"Collecting files from {config.source_dir}...")
    all_files: List[Path] = []
    for item in config.source_dir.rglob("*"):
        if item.is_file():
            try:  # Skip target/backup dirs
                resolved_item_parent = item.resolve().parent
                if (
                    resolved_item_parent == config.target_dir.resolve()
                    or (
                        config.backup_dir
                        and resolved_item_parent == config.backup_dir.resolve()
                    )
                    or (
                        config.backup_dir
                        and config.backup_dir.resolve() in item.resolve().parents
                    )
                    or resolved_item_parent == config.low_quality_path.resolve()
                    or resolved_item_parent == config.near_duplicates_path.resolve()
                    or resolved_item_parent == config.unknown_dir_path.resolve()
                    or (
                        resolved_item_parent.parent == config.target_dir.resolve()
                        and resolved_item_parent.name.isdigit()
                    )
                    or (
                        resolved_item_parent.parent
                        and resolved_item_parent.parent.parent
                        == config.target_dir.resolve()
                        and resolved_item_parent.parent.name.isdigit()
                    )
                    or (
                        resolved_item_parent.parent
                        and resolved_item_parent.parent.parent
                        and resolved_item_parent.parent.parent.parent
                        == config.target_dir.resolve()
                    )
                ):  # YMD check
                    continue
            except Exception:
                pass
            if item.suffix.lower() in SUPPORTED_EXTENSIONS:
                all_files.append(item)
    logging.info(f"Found {len(all_files)} potential media files to process.")

    # --- Initialize Counters & Start Processing ---
    processed_hashes: Set[str] = set()  # Exact hashes
    files_moved_sorted: int = 0
    files_moved_low_quality: int = 0  # Track separately now
    files_moved_near_duplicate: int = 0  # Track separately now
    files_skipped_duplicate_content: int = 0
    files_unknown_date: int = 0
    exif_added_count: int = 0
    errors_encountered: int = 0
    low_quality_logged: int = 0

    print("-" * 30)
    logging.info(f"Starting processing run...")

    # --- Processing Loop with TQDM ---
    for item in tqdm(
        all_files,
        desc="Processing files",
        unit="file",
        disable=(config.log_level == "DEBUG" or len(all_files) < 10),
    ):
        # Reset flags for this file
        moved, skipped, exif, error, fhash = (False, False, False, False, None)

        # process_file now handles the pre-checks (low quality, near dupe move)
        moved, skipped, exif, error, fhash = process_file(
            item, config, processed_hashes
        )

        # Update counters based on results
        if moved:
            files_moved_sorted += (
                1  # process_file returns moved=True only for sorted files now
            )
        if skipped:
            files_skipped_duplicate_content += 1
        if exif:
            exif_added_count += 1
        if error:
            errors_encountered += 1
        if fhash and (moved or skipped):
            processed_hashes.add(fhash)

    # --- Post-Processing Actions (Logging Near Duplicates) ---
    near_duplicate_sets_found = 0
    if config.near_duplicates_action == "log":
        logging.info("--- Near-Duplicate Detection Report ---")
        for phash, file_tuples in config.perceptual_hashes.items():
            if len(file_tuples) > 1:
                near_duplicate_sets_found += 1
                logging.info(f"Potential Near-Duplicate Set (phash: {phash}):")
                for fpath, moved_flag in file_tuples:
                    try:
                        relative_fpath = fpath.relative_to(config.source_dir)
                    except ValueError:
                        relative_fpath = fpath  # Keep absolute if not relative
                    logging.info(f"  - {relative_fpath}")
        logging.info(
            f"Found {near_duplicate_sets_found} sets of potential near-duplicates."
        )
        logging.info("--- End Report ---")

    # --- Final Summary ---
    # Get counts from actual folders for moved items
    final_unknown_count = (
        len(list(config.unknown_dir_path.glob("*")))
        if config.unknown_dir_path.exists()
        else 0
    )
    final_low_quality_count = (
        len(list(config.low_quality_path.glob("*")))
        if config.low_quality_path.exists()
        else 0
    )
    # Near dupe count is trickier - count files within subdirs? Let's count sets found.
    final_near_dupe_sets = (
        len([d for d in config.near_duplicates_path.iterdir() if d.is_dir()])
        if config.near_duplicates_path.exists()
        else 0
    )

    summary = f"""
----------------------------------------
Processing Summary:
----------------------------------------
Mode: {'DRY RUN' if config.dry_run else 'Execute'}
Source Directory: {config.source_dir}
Target Directory: {config.target_dir}
Source Backup: {config.backup_dir if config.backup_dir else 'N/A'}

Files Found: {len(all_files)}
Files Sorted by Date: {files_moved_sorted}
Exact Content Duplicates Skipped: {files_skipped_duplicate_content}
EXIF Dates Added/Updated: {exif_added_count}

Low Quality Files Moved to Review: {final_low_quality_count} ('{LOW_QUALITY_FOLDER_NAME}')
Near Duplicate Sets Moved to Review: {final_near_dupe_sets} (folders in '{NEAR_DUPLICATES_FOLDER_NAME}')
(Near Duplicate Sets Logged: {near_duplicate_sets_found if config.near_duplicates_action == 'log' else 'N/A'})

Files Moved to '{UNKNOWN_DATE_FOLDER_NAME}': {final_unknown_count}
Errors Encountered During Processing: {errors_encountered}
----------------------------------------"""
    logging.info(summary)
    print(summary)


# --- Script Execution ---
if __name__ == "__main__":
    main()
