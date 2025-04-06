#!/usr/bin/env python3

import argparse
import hashlib
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Set

# External dependencies (install with: pip install Pillow piexif)
try:
    from PIL import Image
    from PIL.ExifTags import TAGS
    import piexif
except ImportError:
    print("Error: Required libraries 'Pillow' and 'piexif' not found.", file=sys.stderr)
    print("Please install them: pip install Pillow piexif", file=sys.stderr)
    sys.exit(1)

# --- Constants ---
UNKNOWN_DATE_FOLDER_NAME: str = "unknown_date"
FILENAME_DATE_REGEX: re.Pattern = re.compile(r".*?(\d{8})_(\d{6}).*?")
SUPPORTED_EXTENSIONS: Set[str] = {
    '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.heic', '.webp', '.gif', '.bmp', ".mp4"
}
EXIF_DATETIME_ORIGINAL_TAG: int = 36867
EXIF_DATETIME_DIGITIZED_TAG: int = 36868
EXIF_DATETIME_TAG: int = 306


# --- Helper Functions (No changes from previous version) ---

def get_exif_datetime(filepath: Path) -> Optional[datetime]:
    """Tries to extract the original creation datetime from EXIF data."""
    try:
        img = Image.open(filepath)
        try:
            exif_data = img.getexif()
        except AttributeError:
             exif_dict = getattr(img, 'info', {}).get('exif')
             if exif_dict:
                 try:
                     exif_data = piexif.load(exif_dict)
                 except Exception:
                      exif_data = None
             else:
                 exif_data = None

        if not exif_data:
            return None

        date_tags_priority = [EXIF_DATETIME_ORIGINAL_TAG, EXIF_DATETIME_DIGITIZED_TAG, EXIF_DATETIME_TAG]
        date_str: Optional[str] = None

        for tag_id in date_tags_priority:
            if isinstance(exif_data, dict): # piexif or direct .info
                 exif_ifd = exif_data.get("Exif", {})
                 zeroth_ifd = exif_data.get("0th", {})
                 tag_val_bytes = None
                 if tag_id == EXIF_DATETIME_ORIGINAL_TAG and tag_id in exif_ifd:
                     tag_val_bytes = exif_ifd.get(tag_id)
                 elif tag_id == EXIF_DATETIME_DIGITIZED_TAG and tag_id in exif_ifd:
                      tag_val_bytes = exif_ifd.get(tag_id)
                 elif tag_id == EXIF_DATETIME_TAG and tag_id in zeroth_ifd:
                      tag_val_bytes = zeroth_ifd.get(tag_id)

                 if tag_val_bytes and isinstance(tag_val_bytes, bytes):
                      date_str = tag_val_bytes.decode('utf-8', errors='ignore').strip()
                      if date_str: break

            elif hasattr(exif_data, 'get'): # Standard PIL _Exif data
                 date_str_raw = exif_data.get(tag_id)
                 if date_str_raw:
                     if isinstance(date_str_raw, bytes):
                         date_str = date_str_raw.decode('utf-8', errors='ignore').strip()
                     else:
                         date_str = str(date_str_raw).strip()
                     if date_str: break

        if not date_str:
            return None

        date_str = date_str.replace('\x00', '')

        try:
            return datetime.strptime(date_str, '%Y:%m:%d %H:%M:%S')
        except (ValueError, TypeError):
            try:
                # Handle rare cases where date might use spaces
                return datetime.strptime(date_str, '%Y %m %d %H:%M:%S')
            except (ValueError, TypeError):
                 return None

    except FileNotFoundError:
        return None
    except Exception as e:
        # Mute most EXIF read errors unless debugging needed
        # print(f"  [Warning] Could not read EXIF from {filepath.name}: {type(e).__name__}", file=sys.stderr)
        return None


def parse_filename_datetime(filename: str) -> Optional[datetime]:
    """Tries to extract YYYYMMDD_HHMMSS datetime from the filename."""
    match = FILENAME_DATE_REGEX.match(filename)
    if match:
        date_str = match.group(1)
        time_str = match.group(2)
        try:
            return datetime.strptime(f"{date_str}{time_str}", '%Y%m%d%H%M%S')
        except ValueError:
            return None
    return None


def set_exif_datetime(filepath: Path, dt: datetime) -> bool:
    """Attempts to write the datetime (dt) into the image's EXIF data using piexif."""
    try:
        exif_dt_str: str = dt.strftime('%Y:%m:%d %H:%M:%S')
        exif_bytes: Optional[bytes] = None

        try:
            exif_dict = piexif.load(str(filepath))
        except piexif.InvalidImageDataError:
             exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
        except ValueError as e:
            # print(f"  [Warning] piexif could not load existing EXIF data from {filepath.name} (ValueError: {e}). Cannot add EXIF date.", file=sys.stderr)
            return False
        except Exception as load_err:
             # print(f"  [Warning] Could not load EXIF data using piexif for writing ({filepath.name}): {type(load_err).__name__}. Cannot add EXIF date.", file=sys.stderr)
             return False

        if "Exif" not in exif_dict:
             exif_dict["Exif"] = {}

        exif_dict['Exif'][EXIF_DATETIME_ORIGINAL_TAG] = exif_dt_str.encode('utf-8')
        exif_dict['Exif'][EXIF_DATETIME_DIGITIZED_TAG] = exif_dt_str.encode('utf-8')

        if "0th" not in exif_dict:
            exif_dict["0th"] = {}
        exif_dict['0th'][EXIF_DATETIME_TAG] = exif_dt_str.encode('utf-8')

        try:
             exif_bytes = piexif.dump(exif_dict)
        except Exception as dump_err:
             # print(f"  [Error] Failed to dump modified EXIF data for {filepath.name}: {dump_err}", file=sys.stderr)
             return False

        if exif_bytes:
             piexif.insert(exif_bytes, str(filepath))
             return True
        else:
             return False

    except FileNotFoundError:
        return False
    except PermissionError:
         return False
    except Exception as e:
        # Mute most EXIF write errors unless debugging needed
        # print(f"  [Error] Failed to write EXIF data to {filepath.name}: {type(e).__name__}", file=sys.stderr)
        return False


def calculate_hash(filepath: Path, block_size: int = 65536) -> Optional[str]:
    """Calculates the SHA256 hash of a file."""
    hasher = hashlib.sha256()
    try:
        with filepath.open('rb') as file:
            while True:
                buf = file.read(block_size)
                if not buf:
                    break
                hasher.update(buf)
        return hasher.hexdigest()
    except OSError as e:
        print(f"  [Error] Could not read file for hashing {filepath.name}: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  [Error] Unexpected error hashing {filepath.name}: {e}", file=sys.stderr)
        return None

# --- Main Function ---

def main():
    parser = argparse.ArgumentParser(
        description="Sorts image files recursively based on EXIF date or filename date."
                    " Renames dated files to 'YYYY-MM-DD_HHMMSS.ext'."
                    " Places dated files directly in the target folder (flat hierarchy)."
                    " Puts files with no date info into an 'unknown_date' subfolder."
                    " Skips exact duplicates and adds EXIF date if inferred from filename.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-s", "--source", type=Path, required=True,
        help="Source directory containing unsorted images."
    )
    parser.add_argument(
        "-t", "--target", type=Path, required=True,
        help="Target directory where sorted images will be placed."
    )
    args = parser.parse_args()

    source_dir: Path = args.source.resolve()
    target_dir: Path = args.target.resolve()

    # --- Validate Input/Output ---
    if not source_dir.is_dir():
        print(f"Error: Source directory '{source_dir}' not found or is not a directory.", file=sys.stderr)
        sys.exit(1)

    if target_dir.exists() and not target_dir.is_dir():
         print(f"Error: Target path '{target_dir}' exists but is not a directory.", file=sys.stderr)
         sys.exit(1)

    if source_dir == target_dir:
         print(f"Error: Source and target directories cannot be the same.", file=sys.stderr)
         sys.exit(1)

    try:
        # Check if target is inside source
        if target_dir.resolve() in source_dir.resolve().parents or target_dir.resolve() == source_dir.resolve():
             # More robust check using resolve() and parents
             print(f"Error: Target directory '{target_dir}' cannot be the same as or inside the source directory '{source_dir}'.", file=sys.stderr)
             sys.exit(1)
    except Exception: # Catch potential errors during path comparison on odd systems
        pass

    # --- Setup ---
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"Ensured target directory exists: {target_dir}")
    except OSError as e:
        print(f"Error: Could not create target directory '{target_dir}': {e}", file=sys.stderr)
        sys.exit(1)

    unknown_dir_path: Path = target_dir / UNKNOWN_DATE_FOLDER_NAME
    try:
        unknown_dir_path.mkdir(exist_ok=True)
        print(f"Ensured unknown date directory exists: {unknown_dir_path}")
    except OSError as e:
        print(f"Error: Could not create unknown date directory '{unknown_dir_path}': {e}", file=sys.stderr)
        sys.exit(1)

    processed_hashes: Set[str] = set()
    total_files_scanned: int = 0
    image_files_found: int = 0
    files_processed: int = 0
    files_moved: int = 0
    files_skipped_duplicate_content: int = 0
    files_unknown_date: int = 0
    exif_added_count: int = 0
    errors_encountered: int = 0

    print(f"\nStarting scan in: {source_dir}")
    print(f"Outputting sorted files to: {target_dir} (flat hierarchy)")
    print(f"Dated files renamed to: YYYY-MM-DD_HHMMSS.ext")
    print(f"Undated files will go to: {unknown_dir_path}")
    print("-" * 30)

    # --- Processing Loop ---
    for item in source_dir.rglob('*'):
        total_files_scanned += 1
        if item.is_file():
            # Skip files already in the target or unknown directory to prevent re-processing
            try:
                 if item.resolve().parent == target_dir.resolve() or item.resolve().parent == unknown_dir_path.resolve():
                      continue
            except Exception: # Handle potential resolution errors
                pass

            if item.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            image_files_found += 1
            try:
                 relative_path = item.relative_to(source_dir)
            except ValueError: # Handle cases where item might not be relative (e.g. symlinks outside)
                 relative_path = item.name
            print(f"Processing: {relative_path}")
            files_processed += 1

            # 1. Check for exact duplicate content
            file_hash = calculate_hash(item)
            if file_hash:
                if file_hash in processed_hashes:
                    print(f"  [Skipped] Exact duplicate content of a previously processed file.")
                    files_skipped_duplicate_content += 1
                    continue
            else:
                 print(f"  [Warning] Could not hash file {item.name}, cannot check for content duplication.", file=sys.stderr)
                 errors_encountered += 1

            file_datetime: Optional[datetime] = None
            date_source: str = "None"
            added_exif: bool = False

            # 2. Try getting date from EXIF (Priority 1)
            file_datetime = get_exif_datetime(item)
            if file_datetime:
                date_source = "EXIF"
            else:
                # 3. Try getting date from filename (Priority 2)
                file_datetime = parse_filename_datetime(item.name)
                if file_datetime:
                    date_source = "Filename"
                    print(f"  Found date in filename: {file_datetime.strftime('%Y-%m-%d %H:%M:%S')}. Attempting to add to EXIF...")
                    if set_exif_datetime(item, file_datetime):
                        added_exif = True
                        exif_added_count += 1
                    else:
                        print(f"  [Warning] Failed to add EXIF date derived from filename for {item.name}.", file=sys.stderr)
                        errors_encountered +=1
                else:
                    date_source = "Unknown"

            # 6. Determine target path, new filename, and handle collisions
            target_folder: Path
            target_filename_base: str
            target_filename: str

            if file_datetime:
                # *** MODIFIED: Generate filename format: YYYY-MM-DD_HHMMSS ***
                formatted_date = file_datetime.strftime('%Y-%m-%d_%H%M%S')
                target_filename_base = formatted_date # Just the date/time string
                target_filename = f"{target_filename_base}{item.suffix}"
                target_folder = target_dir # Flat hierarchy
                print(f"  Determined date: {file_datetime.strftime('%Y-%m-%d %H:%M:%S')}. Target name base: {target_filename_base}")

            else:
                # Keep original name for unknown date files
                target_filename_base = item.stem # Use original stem for collision handling base
                target_filename = item.name
                target_folder = unknown_dir_path
                files_unknown_date += 1
                print(f"  No date found. Moving original name to: {unknown_dir_path.name}")


            try:
                target_folder.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                print(f"  [Error] Could not create target folder {target_folder}: {e}", file=sys.stderr)
                errors_encountered += 1
                continue

            # Handle potential filename collisions
            target_filepath: Path = target_folder / target_filename
            counter: int = 1
            original_target_filename_base = target_filename_base # Store original base for collision naming
            while target_filepath.exists():
                existing_hash = calculate_hash(target_filepath)
                if file_hash and existing_hash and file_hash == existing_hash:
                     try: # Make relative path printing more robust
                         relative_target_path = target_filepath.relative_to(target_dir.parent)
                     except ValueError:
                         relative_target_path = target_filepath.name
                     print(f"  [Skipped] Identical file already exists at destination: {relative_target_path}")
                     files_skipped_duplicate_content +=1
                     target_filepath = None
                     break

                # Append counter to the *original* base name determined for this file
                target_filename = f"{original_target_filename_base}_{counter}{item.suffix}"
                target_filepath = target_folder / target_filename
                counter += 1
                if counter > 100: # Safety break
                     print(f"  [Error] Too many filename collisions for base '{original_target_filename_base}' in {target_folder.name}. Skipping.", file=sys.stderr)
                     errors_encountered += 1
                     target_filepath = None
                     break

            # 7. Move the file if valid target path determined
            if target_filepath:
                try:
                    try:
                         relative_target_path = target_filepath.relative_to(target_dir.parent)
                    except ValueError:
                         relative_target_path = target_filepath.name
                    print(f"  Moving to: {relative_target_path}")
                    shutil.move(str(item), str(target_filepath))
                    files_moved += 1
                    if file_hash:
                        processed_hashes.add(file_hash)
                except Exception as e:
                    print(f"  [Error] Could not move file {item.name} to {target_filepath}: {type(e).__name__} - {e}", file=sys.stderr)
                    errors_encountered += 1

    # --- Summary ---
    print("-" * 30)
    print("Scan complete.")
    print(f"Total items scanned: {total_files_scanned}")
    print(f"Supported image files found: {image_files_found}")
    print(f"Image files processed: {files_processed}")
    print(f"Files moved/sorted: {files_moved}")
    print(f"EXIF dates added/updated: {exif_added_count}")
    print(f"Exact content duplicates skipped: {files_skipped_duplicate_content}")
    print(f"Files moved to '{UNKNOWN_DATE_FOLDER_NAME}': {files_unknown_date}")
    print(f"Errors encountered: {errors_encountered}")
    print("-" * 30)

# --- Script Execution ---
if __name__ == "__main__":
    main()
