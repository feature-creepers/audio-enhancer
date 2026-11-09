"""Copy ID3 metadata from source MP3 to enhanced output."""

from __future__ import annotations

import logging
from pathlib import Path

from mutagen.id3 import ID3, ID3NoHeaderError

logger = logging.getLogger(__name__)

# Human-readable labels for common ID3 frames
_TAG_LABELS = {
    "TIT2": "title",
    "TPE1": "artist",
    "TPE2": "album_artist",
    "TALB": "album",
    "TRCK": "track",
    "TYER": "date",
    "TDRC": "date",
    "TCON": "genre",
    "APIC": "cover",
}


def copy_tags(source: Path, dest: Path) -> list[str]:
    """Copy all ID3 tags (including cover art) from source to dest."""
    source = Path(source)
    dest = Path(dest)

    try:
        source_tags = ID3(source)
    except ID3NoHeaderError:
        logger.warning("No ID3 tags found in %s", source.name)
        return []

    dest_tags = ID3()
    copied_frames: list[str] = []

    for key in source_tags.keys():
        dest_tags[key] = source_tags[key]
        label = _TAG_LABELS.get(key, key)
        if label not in copied_frames:
            copied_frames.append(label)

    dest_tags.save(dest, v2_version=3)
    return copied_frames


def format_copied_tags(tags: list[str]) -> str:
    if not tags:
        return "none"
    priority = ["title", "artist", "album", "album_artist", "track", "genre", "date", "cover"]
    ordered = [t for t in priority if t in tags]
    ordered.extend(t for t in tags if t not in ordered)
    return ", ".join(ordered)
