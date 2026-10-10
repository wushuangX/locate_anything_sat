"""Archive completed inference checkpoints without changing a running trainer."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import time


def files(checkpoint):
    return sorted(p for p in checkpoint.iterdir() if p.is_file()
                  and not p.name.startswith(('._', 'dataloader_state', 'rng_state'))
                  and p.name not in {'training_args.bin', 'zero_to_fp32.py', 'latest',
                                     'archive_complete.json'})


def validate(checkpoint, step):
    state = json.loads((checkpoint / 'trainer_state.json').read_text())
    if state['global_step'] != step:
        raise ValueError('trainer state step mismatch')
    index = json.loads((checkpoint / 'model.safetensors.index.json').read_text())
    for name in set(index['weight_map'].values()):
        path = checkpoint / name
        with path.open('rb') as stream:
            header_size = struct.unpack('<Q', stream.read(8))[0]
            header = json.loads(stream.read(header_size))
        offsets = [v['data_offsets'][1] for k, v in header.items() if k != '__metadata__']
        if path.stat().st_size != 8 + header_size + max(offsets):
            raise ValueError(f'incomplete weights: {name}')
    for name in ('config.json', 'tokenizer_config.json', 'preprocessor_config.json'):
        json.loads((checkpoint / name).read_text())
    manifest = json.loads((checkpoint / 'runtime_manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if name.startswith('._'):
            continue
        if hashlib.sha256((checkpoint / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'runtime hash mismatch: {name}')


def archive(source, target, step):
    if (target / 'archive_complete.json').exists():
        return
    temporary = target.with_name('.' + target.name + '.partial')
    # A prior process may have died while linking. Only this private staging path is removed.
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    try:
        for path in files(source):
            os.link(path, temporary / path.name)
        validate(temporary, step)
        record = {'step': step, 'source': str(source), 'created_unix': time.time(),
                  'purpose': 'inference_only', 'storage': 'hardlinks',
                  'files': {p.name: p.stat().st_size for p in files(temporary)}}
        (temporary / 'archive_complete.json').write_text(json.dumps(record, indent=2) + '\n')
        if target.exists():
            # Preserve an existing archive, validate it, and only add its completion record.
            validate(target, step)
            (target / 'archive_complete.json').write_text(json.dumps(record, indent=2) + '\n')
            shutil.rmtree(temporary)
        else:
            temporary.rename(target)
        print(f'Archived checkpoint-{step}: {target}', flush=True)
    except Exception:
        shutil.rmtree(temporary)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--archive-dir', type=Path, required=True)
    parser.add_argument('--interval', type=int, default=5000)
    parser.add_argument('--poll-seconds', type=float, default=15)
    parser.add_argument('--stable-seconds', type=float, default=30)
    parser.add_argument('--preserve-existing', action='store_true')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.interval <= 0 or args.poll_seconds <= 0 or args.stable_seconds < 0:
        parser.error('interval/poll must be positive and stable-seconds nonnegative')
    args.archive_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.archive_dir / '.archive.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    initial = set(args.run_dir.glob('checkpoint-*')) if args.preserve_existing else set()
    seen = {}
    while True:
        for source in sorted(args.run_dir.glob('checkpoint-*')):
            if not source.name[11:].isdigit():
                continue
            step = int(source.name[11:])
            target = args.archive_dir / source.name
            if (step % args.interval and source not in initial) or (target / 'archive_complete.json').exists():
                continue
            try:
                signature = tuple((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in files(source))
                previous, since = seen.get(source, (None, time.monotonic()))
                if signature != previous:
                    seen[source] = (signature, time.monotonic())
                elif time.monotonic() - since >= args.stable_seconds:
                    validate(source, step)
                    archive(source, target, step)
            except (OSError, ValueError, KeyError, struct.error) as error:
                print(f'Retry {source.name}: {error}', flush=True)
        if args.once:
            # One-shot validates immediately; useful for known-complete checkpoints.
            for source in args.run_dir.glob('checkpoint-*'):
                if source.name[11:].isdigit():
                    step = int(source.name[11:])
                    if step % args.interval == 0 or source in initial:
                        archive(source, args.archive_dir / source.name, step)
            return
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
