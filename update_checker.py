"""GitHub release update checks for GZModel Editor."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


APP_VERSION = "3.0.0"
REPOSITORY_URL = "https://github.com/Digitzaki/GZ-Model-Editor"
LATEST_RELEASE_API = "https://api.github.com/repos/Digitzaki/GZ-Model-Editor/releases/latest"
RELEASE_ASSET_NAME = "GZModel Editor.exe"
USER_AGENT = f"GZModel-Editor/{APP_VERSION}"


def _version_tuple(tag: str) -> tuple[int, ...] | None:
    match = re.fullmatch(r"v?(\d+(?:\.\d+)*)", str(tag).strip(), flags=re.IGNORECASE)
    if not match:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()


def _release_asset(release: dict) -> dict | None:
    assets = [asset for asset in release.get("assets", []) if asset.get("state") == "uploaded"]
    exact = next(
        (asset for asset in assets if str(asset.get("name", "")).casefold() == RELEASE_ASSET_NAME.casefold()),
        None,
    )
    if exact:
        return exact
    return next(
        (
            asset
            for asset in assets
            if str(asset.get("name", "")).lower().endswith(".exe")
            and "gzmodel" in re.sub(r"[^a-z0-9]", "", str(asset.get("name", "")).lower())
        ),
        None,
    )


def evaluate_release(
    release: dict,
    *,
    current_version: str = APP_VERSION,
    executable: Path | None = None,
    frozen: bool | None = None,
) -> dict:
    latest_tag = str(release.get("tag_name") or "").strip()
    release_url = str(release.get("html_url") or REPOSITORY_URL + "/releases/latest")
    asset = _release_asset(release)
    download_url = str(asset.get("browser_download_url") or release_url) if asset else release_url
    current_key = _version_tuple(current_version)
    latest_key = _version_tuple(latest_tag)
    tag_comparison_failed = False

    if current_key is not None and latest_key is not None:
        width = max(len(current_key), len(latest_key))
        current_key += (0,) * (width - len(current_key))
        latest_key += (0,) * (width - len(latest_key))
        if latest_key > current_key:
            return {
                "status": "update",
                "reason": "version",
                "current_version": current_version,
                "latest_tag": latest_tag,
                "release_url": release_url,
                "download_url": download_url,
            }
        if latest_key < current_key:
            return {
                "status": "ahead",
                "current_version": current_version,
                "latest_tag": latest_tag,
                "release_url": release_url,
            }
    elif latest_tag.casefold().lstrip("v") != current_version.casefold().lstrip("v"):
        tag_comparison_failed = True

    is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else bool(frozen)
    digest_text = str(asset.get("digest") or "") if asset else ""
    algorithm, separator, remote_digest = digest_text.partition(":")
    if is_frozen and asset and algorithm.lower() == "sha256" and separator and remote_digest:
        local_path = executable or Path(sys.executable)
        local_digest = _sha256(local_path)
        if local_digest != remote_digest.strip().lower():
            return {
                "status": "update",
                "reason": "checksum",
                "current_version": current_version,
                "latest_tag": latest_tag,
                "release_url": release_url,
                "download_url": download_url,
                "asset_name": str(asset.get("name") or RELEASE_ASSET_NAME),
            }
        return {
            "status": "current",
            "current_version": current_version,
            "latest_tag": latest_tag,
            "release_url": release_url,
            "checksum_checked": True,
        }

    if tag_comparison_failed:
        return {
            "status": "unknown_tag",
            "current_version": current_version,
            "latest_tag": latest_tag or "unknown",
            "release_url": release_url,
        }

    return {
        "status": "current",
        "current_version": current_version,
        "latest_tag": latest_tag,
        "release_url": release_url,
        "checksum_checked": False,
    }


def check_for_updates() -> dict:
    request = Request(
        LATEST_RELEASE_API,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": USER_AGENT,
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urlopen(request, timeout=10) as response:
            release = json.load(response)
    except HTTPError as exc:
        if exc.code == 404:
            return {"status": "no_release", "current_version": APP_VERSION}
        raise RuntimeError(f"GitHub returned HTTP {exc.code}.") from exc
    except URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise RuntimeError(f"Could not connect to GitHub: {reason}") from exc
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read GitHub release information: {exc}") from exc
    return evaluate_release(release)
