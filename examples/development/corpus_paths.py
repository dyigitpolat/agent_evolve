"""Resolve a research-corpus path across the 2026-07-28 archive split.

On 2026-07-28 the research artifacts were split: the authoring surface kept the
live material and everything superseded moved, unmodified and with relative
structure preserved, into ``archive/``. The corpus README states the rule
directly -- "if a path appears in ``EVIDENCE_INVENTORY.md``, prefix it with
``archive/`` to resolve it" -- but the campaign scripts were never told, so they
still compute pre-split paths and fail on files that are present and intact one
directory away.

**Why a fallback rather than repointing the root.** Repointing moves every
path at once, including paths that legitimately resolve live, so a file
existing in both places would silently start resolving to the archived copy.
The fallback fires only when the live path is absent, which is the narrower
change and the one that cannot alter a path that already works.

**Why this cannot alter what a sealed run resolves to.** These files are read
behind frozen content hashes -- ``run_boils_agentic_pilot_v2`` checks
``EXPECTED_LEGAL_FILE_SHA256`` plus three further identity constants before
using anything. So either the archived bytes are the sealed bytes and the
resolution is identical, or they are not and the run fails loudly on the hash
rather than proceeding with different evidence. Verified for the file that
gated collection: the archived copy hashes to ``49f14616...``, exactly the
constant the loader expects.

The receipts are the evidence, so the resolver never rewrites, copies, or
normalizes anything. It only answers *where a file is*.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

__all__ = ["resolve_corpus_path", "corpus_path_or_none", "ARCHIVE_DIRNAME"]

ARCHIVE_DIRNAME = "archive"


def _split_at_corpus(path: Path) -> Optional[tuple]:
    """Return ``(corpus_root, relative)`` when *path* lies inside a corpus."""
    parts = path.parts
    try:
        index = len(parts) - 1 - parts[::-1].index("research_artifacts")
    except ValueError:
        return None
    root = Path(*parts[: index + 1])
    relative = Path(*parts[index + 1 :]) if index + 1 < len(parts) else Path()
    return root, relative


def corpus_path_or_none(path) -> Optional[Path]:
    """Return the readable location of *path*, or ``None`` if there is none.

    Tries the given path first, so anything that already resolves is untouched.
    Falls back to the same relative location under ``archive/``.
    """
    candidate = Path(path)
    if candidate.exists():
        return candidate
    split = _split_at_corpus(candidate)
    if split is None:
        return None
    root, relative = split
    if not relative.parts or relative.parts[0] == ARCHIVE_DIRNAME:
        return None
    archived = root / ARCHIVE_DIRNAME / relative
    return archived if archived.exists() else None


def resolve_corpus_path(path) -> Path:
    """Like :func:`corpus_path_or_none`, but explain rather than raise blankly."""
    resolved = corpus_path_or_none(path)
    if resolved is not None:
        return resolved
    candidate = Path(path)
    split = _split_at_corpus(candidate)
    if split is not None:
        root, relative = split
        raise FileNotFoundError(
            f"{candidate} is absent, and so is the archived location "
            f"{root / ARCHIVE_DIRNAME / relative} that the 2026-07-28 split "
            "would have moved it to"
        )
    raise FileNotFoundError(str(candidate))
