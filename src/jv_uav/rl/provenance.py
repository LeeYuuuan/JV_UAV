"""Small, per-invocation records linking results to configuration and source."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import uuid
import zipfile


def record_run(root, output, trainer, args):
    root, output = Path(root), Path(output)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '_' + uuid.uuid4().hex[:8]
    folder = output / 'provenance' / stamp
    folder.mkdir(parents=True, exist_ok=False)
    config = {'environment': trainer.env_cfg, 'training': trainer.cfg}
    # Normalize YAML integer keys to JSON strings before computing a reproducible hash.
    config = json.loads(json.dumps(config))
    config_bytes = json.dumps(config, sort_keys=True, ensure_ascii=False).encode('utf-8')
    paths = sorted(set(root.glob('*.py')) | set((root / 'src').rglob('*.py'))
                   | set((root / 'configs').rglob('*.yaml')) | set((root / 'configs').rglob('*.yml')))
    hashes = {}
    with zipfile.ZipFile(folder / 'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            relative = path.relative_to(root).as_posix()
            content = path.read_bytes()
            hashes[relative] = hashlib.sha256(content).hexdigest()
            archive.writestr(relative, content)

    def git(*arguments):
        try:
            return subprocess.check_output(['git', '-C', str(root), *arguments],
                                           stderr=subprocess.DEVNULL, timeout=5).decode('utf-8').strip()
        except (OSError, subprocess.SubprocessError):
            return None

    manifest = {
        'schema_version': 1, 'invocation_id': stamp,
        'description': args.run_note or 'See effective_reward and resolved_config for this invocation.',
        'git_commit': git('rev-parse', 'HEAD'),
        'git_status': git('status', '--porcelain', '--untracked-files=normal'),
        'source_sha256': hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        'source_file_hashes': hashes, 'source_archive': 'source.zip',
        'config_sha256': hashlib.sha256(config_bytes).hexdigest(),
        'effective_reward': trainer.env_cfg['reward'],
        'arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        'resume_from': str(args.resume.resolve()) if args.resume else None,
        'starting_counters': {key: getattr(trainer, key) for key in ('upper_steps', 'low_steps', 'episodes', 'sac_updates', 'mappo_updates')},
        'python': sys.version, 'platform': platform.platform(), 'device': str(trainer.device),
    }
    import numpy
    import torch
    manifest['packages'] = {'numpy': numpy.__version__, 'torch': torch.__version__, 'cuda': torch.version.cuda}
    (folder / 'resolved_config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    return folder
