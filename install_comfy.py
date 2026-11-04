"""Install a pinned ComfyUI source archive without Git or executing archive paths."""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
import urllib.request

source = json.loads(Path('/iabot/comfy-source.json').read_text())
with urllib.request.urlopen(source['url'], timeout=120) as response:
    payload = response.read(100 * 1024 * 1024)
if hashlib.sha256(payload).hexdigest() != source['sha256']:
    raise RuntimeError('ComfyUI source integrity mismatch')
root = Path('/comfyui')
root.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as archive:
    for item in archive:
        parts = PurePosixPath(item.name).parts[1:]
        if not parts or '..' in parts or item.issym() or item.islnk():
            continue
        target = root.joinpath(*parts)
        if item.isdir():
            target.mkdir(parents=True, exist_ok=True)
        elif item.isfile():
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(item) as content, target.open('wb') as output:
                output.write(content.read())
