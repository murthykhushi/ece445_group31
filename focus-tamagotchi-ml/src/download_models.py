"""Explicit model setup, separate from camera/inference startup."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import ssl
import certifi

ROOT = Path(__file__).resolve().parents[1]
from vision.model_artifacts import ARTIFACTS



def download(name, directory, expected_sha256=None):
    artifact = ARTIFACTS[name]
    filename, url = artifact['filename'], artifact['url']
    expected_sha256 = expected_sha256 or artifact['sha256']
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.partial')
    try:
        if not destination.exists():
            with urllib.request.urlopen(url, timeout=60, context=ssl.create_default_context(cafile=certifi.where())) as response, temporary.open('wb') as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
            candidate = temporary
        else:
            candidate = destination
        digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if expected_sha256 and digest != expected_sha256:
            raise ValueError(f'SHA256 mismatch for {filename}')
        if candidate == temporary:
            temporary.replace(destination)
        manifest = dict(url=url, filename=filename, sha256=digest,
                        checksum_pinned=bool(expected_sha256))
        destination.with_suffix(destination.suffix + '.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print(json.dumps(manifest))
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fan', action='store_true')
    parser.add_argument('--yunet', action='store_true')
    parser.add_argument('--mediapipe', action='store_true')
    parser.add_argument('--directory', type=Path, default=ROOT / 'models')
    parser.add_argument('--sha256', help='Require a recorded SHA256 for one selected model')
    args = parser.parse_args()
    selected = [name for name in ARTIFACTS if getattr(args, name)]
    if not selected or (args.sha256 and len(selected) != 1):
        parser.error('Choose --fan, --yunet and/or --mediapipe; --sha256 applies to one model')
    for name in selected:
        download(name, args.directory, args.sha256)


if __name__ == '__main__':
    main()
