"""Fetch the pinned RIDE Gold Lite tabular training split and its schema."""

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import signal
import socket
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DATASET = "orailix/ride-gold-lite"
REVISION = "e5c1ed1126bca1784eb6ea93816dd6cc89afa544"
BASE_URL = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}"
LICENSE = "CC-BY-4.0"
ATTRIBUTION = "RIDE (OrAilix), based on data from Infrabel and Open-Meteo"
FILES = {
    "tabular/train/x.npy": "e2ce7c514ac9b4f4a681a0a0c310fed5e6b72ba76cae8de4d24119acf686e23d",
    "tabular/train/y.npy": "f754a20c23de8d53ff50aede297a7351516be4c658ff3ce8bad275be038cd21b",
    "tabular/train/y_mask.npy": "a185872be2db48080de931e6820ff828c08a6e429c199a23c3a31335dfeadc8f",
    "tabular/train/md.npy": "6b229312b8cc8b3225c919edb421618d7ecd4819e0c79efbdee19c3e887bc0a6",
    "tabular/scheme.yaml": "83edb64b8ffb39903078f581eef248e020bd996de53984930707be7aab54637c",
    "tabular/normalization.yaml": (
        "e770c47b45cbe91f89745dc3dfdd98a3ce4e79b690dbbb60059477c132be318a"
    ),
}


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _download_with_hub(output_dir: Path, relative_path: str, *, disable_xet: bool = False) -> Path:
    """Run the Hub transfer in a process group that can be stopped on Ctrl+C."""
    env = os.environ.copy()
    if disable_xet:
        env["HF_HUB_DISABLE_XET"] = "1"
    script = (
        "import sys; from huggingface_hub import hf_hub_download; "
        "hf_hub_download(repo_id=sys.argv[1], filename=sys.argv[2], "
        "repo_type='dataset', revision=sys.argv[3], local_dir=sys.argv[4], "
        "token=None, user_agent='osrd-ops-intelligence/0.1')"
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            script,
            DATASET,
            relative_path,
            REVISION,
            str(output_dir),
        ],
        env=env,
        start_new_session=(os.name != "nt"),
    )
    try:
        return_code = process.wait()
    except KeyboardInterrupt:
        print("Stopping transfer; keeping Hugging Face's resumable cache…", flush=True)
        if os.name == "nt":
            process.terminate()
        else:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=5)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            if os.name == "nt":
                process.kill()
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait()
        raise
    if return_code:
        raise subprocess.CalledProcessError(return_code, process.args)
    return output_dir / relative_path


def _download_http_fallback(
    destination: Path, relative_path: str, expected_sha256: str, token: str | None
) -> int:
    """Resume a sequential HTTPS transfer when Xet reconstruction fails."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.part")
    retryable_http_statuses = {408, 425, 429, 500, 502, 503, 504}
    retryable_network_errors = (
        http.client.HTTPException,
        socket.timeout,
        ssl.SSLError,
        TimeoutError,
        urllib.error.URLError,
        ConnectionError,
    )

    for attempt in range(1, 7):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "osrd-ops-intelligence/0.1"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(
            f"{BASE_URL}/{relative_path}?download=true", headers=headers
        )
        if token:
            request.add_unredirected_header("Authorization", f"Bearer {token}")

        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                append = offset > 0 and response.status == 206
                if append:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-"):
                        partial.unlink(missing_ok=True)
                        print(
                            "Server returned an invalid range; restarting from byte zero.",
                            flush=True,
                        )
                        continue
                elif response.status == 200:
                    offset = 0

                content_range = response.headers.get("Content-Range", "")
                total_size: int | None = None
                if "/" in content_range:
                    total_part = content_range.rsplit("/", 1)[1]
                    if total_part.isdigit():
                        total_size = int(total_part)
                if total_size is None:
                    content_length = response.headers.get("Content-Length")
                    if content_length and content_length.isdigit():
                        total_size = offset + int(content_length)

                started = time.monotonic()
                last_report = started
                last_report_size = offset
                if total_size is None:
                    print(
                        f"HTTP fallback {relative_path}: resuming at {offset:,} bytes",
                        flush=True,
                    )
                else:
                    print(
                        f"HTTP fallback {relative_path}: {offset:,} / {total_size:,} bytes",
                        flush=True,
                    )
                with partial.open("ab" if append else "wb") as output:
                    size = offset
                    while chunk := response.read(256 * 1024):
                        output.write(chunk)
                        size += len(chunk)
                        now = time.monotonic()
                        if now - last_report >= 5:
                            elapsed = max(now - last_report, 1e-6)
                            rate = (size - last_report_size) / elapsed
                            if total_size is not None and rate > 0:
                                eta = max(total_size - size, 0) / rate
                                print(
                                    f"  {relative_path}: {size:,} / {total_size:,} bytes "
                                    f"({100 * size / total_size:.1f}%), "
                                    f"{rate / 1024:.1f} KiB/s, ETA {eta / 60:.1f} min",
                                    flush=True,
                                )
                            else:
                                print(f"  {relative_path}: {size:,} bytes", flush=True)
                            last_report = now
                            last_report_size = size
                    output.flush()
                    os.fsync(output.fileno())

        except urllib.error.HTTPError as error:
            if error.code == 416 and partial.exists():
                if _hash_file(partial) == expected_sha256:
                    partial.replace(destination)
                    return destination.stat().st_size
                partial.unlink()
                continue
            elif error.code not in retryable_http_statuses or attempt == 6:
                raise
            else:
                retry_after = error.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
                print(f"HTTP {error.code}; retrying in {delay:.0f}s", flush=True)
                time.sleep(min(delay, 60))
                continue
        except retryable_network_errors as error:
            if attempt == 6:
                raise RuntimeError(
                    f"HTTPS fallback failed after {attempt} attempts: {error}"
                ) from error
            delay = min(2**attempt, 60)
            print(
                f"Transfer interrupted ({error}); resuming in {delay}s (attempt {attempt + 1}/6)",
                flush=True,
            )
            time.sleep(delay)
            continue

        size = partial.stat().st_size
        if total_size is not None and size < total_size:
            if attempt == 6:
                raise RuntimeError(
                    f"Transfer ended early at {size:,} of {total_size:,} bytes after 6 attempts"
                )
            delay = min(2**attempt, 60)
            print(
                f"Transfer ended at {size:,} of {total_size:,} bytes; "
                f"resuming in {delay}s (attempt {attempt + 1}/6)",
                flush=True,
            )
            time.sleep(delay)
            continue

        actual = _hash_file(partial)
        if actual == expected_sha256:
            partial.replace(destination)
            return size
        partial.unlink()
        if attempt == 6:
            raise ValueError(
                f"Integrity check failed for {relative_path} after 6 transfers: sha256={actual}"
            )
        delay = min(2**attempt, 60)
        print(
            f"Checksum mismatch; restarting transfer in {delay}s (attempt {attempt + 1}/6)",
            flush=True,
        )
        time.sleep(delay)

    raise RuntimeError(f"Could not download {relative_path}")


def _download_file(output_dir: Path, relative_path: str, expected_sha256: str) -> int:
    try:
        from huggingface_hub import get_token
    except ImportError as error:
        raise RuntimeError(
            "Install the Hugging Face download extra with `uv sync --extra rail-data`."
        ) from error

    # Use a child process so Ctrl+C can stop native hf_xet work promptly.
    try:
        downloaded = _download_with_hub(output_dir, relative_path)
    except Exception as error:
        print(
            f"Hub/Xet transfer failed for {relative_path} ({error}); "
            "retrying with the Hugging Face Hub's non-Xet downloader.",
            flush=True,
        )
        try:
            downloaded = _download_with_hub(output_dir, relative_path, disable_xet=True)
            actual = _hash_file(downloaded)
            if actual == expected_sha256:
                return downloaded.stat().st_size
            downloaded.unlink(missing_ok=True)
            raise ValueError(f"Integrity check failed after non-Xet retry: sha256={actual}")
        except Exception as fallback_error:
            print(
                f"Non-Xet retry failed ({fallback_error}); trying resumable HTTPS ranges.",
                flush=True,
            )
            try:
                return _download_http_fallback(
                    output_dir / relative_path, relative_path, expected_sha256, get_token()
                )
            except Exception as http_error:
                raise RuntimeError(
                    f"Xet download failed ({error}); non-Xet retry failed "
                    f"({fallback_error}); HTTPS range fallback failed ({http_error})"
                ) from http_error

    actual = _hash_file(downloaded)
    if actual != expected_sha256:
        downloaded.unlink(missing_ok=True)
        raise ValueError(
            f"Integrity check failed for {relative_path}: sha256={actual}; "
            "the downloaded file was removed"
        )
    return downloaded.stat().st_size


def fetch(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, object]] = {}
    for relative_path, expected_sha256 in FILES.items():
        destination = output_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            digest = _hash_file(destination)
            if digest != expected_sha256:
                raise ValueError(f"Existing file failed integrity check: {destination}")
            size = destination.stat().st_size
        else:
            # Preserve old single-stream partials if this accelerated download
            # fails. They are not interchangeable with the Xet chunk cache.
            legacy_partial = destination.with_name(f".{destination.name}.part")
            if legacy_partial.exists():
                print(
                    f"Note: using Xet's resumable cache; legacy partial remains at "
                    f"{legacy_partial} until this file verifies.",
                    flush=True,
                )
            size = _download_file(output_dir, relative_path, expected_sha256)
            legacy_partial.unlink(missing_ok=True)
        files[relative_path] = {"sha256": expected_sha256, "bytes": size}

    manifest: dict[str, object] = {
        "schema": "osrd-ride-training-inputs/v1",
        "dataset": DATASET,
        "revision": REVISION,
        "license": LICENSE,
        "attribution": ATTRIBUTION,
        "training_period": "2023-01 through 2024-12",
        "split": "provider-defined train split only; no evaluation arrays fetched",
        "files": files,
        "usage": "Observed delay forecasting; not OSRD candidate feasibility labels.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/rail/ride-gold-lite-training"),
        help="Ignored local directory for pinned RIDE arrays and provenance manifest",
    )
    args = parser.parse_args()
    try:
        manifest = fetch(args.output_dir)
    except KeyboardInterrupt:
        print("Download cancelled; verified files and resumable cache were kept.", file=sys.stderr)
        raise SystemExit(130) from None
    except Exception as error:
        parser.error(str(error))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
