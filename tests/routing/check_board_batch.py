# SPDX-License-Identifier: GPL-3.0-or-later
"""End-to-end CPU oracle for the multi-board runner, with disjoint outputs."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('--binary', required=True, type=Path)
parser.add_argument('--single', required=True, type=Path)
parser.add_argument('--scratch', required=True, type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
board = root / 'tests/boards/plane_smd/plane_smd.kicad_pcb'


def run(command):
    return subprocess.run([str(x) for x in command], capture_output=True, text=True, timeout=120)


def signature(summary):
    keys = ('routed', 'connections', 'tracks', 'vias', 'expansions', 'pitch_mm',
            'failures', 'variant', 'variant_name', 'copper_layers', 'work',
            'work_all_variants', 'time_limited', 'open', 'completion_pct', 'style')
    return {k: summary[k] for k in keys if k in summary}


args.scratch.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix='batch-runner-', dir=args.scratch) as tmp:
    tmp = Path(tmp)
    baseline = tmp / 'single.kicad_pcb'
    summary = tmp / 'single.json'
    p = run([args.single, 'route', board, '-o', baseline, '--json', summary,
             '--threads', '1', '--variants', '1', '--seed', '1', '--no-kb',
             '--no-gpu', '--no-rescue', '--no-why-open', '--work', '10000', '--time', '3600'])
    if p.returncode not in (0, 3):
        raise RuntimeError(p.stderr[-3000:])
    expected = baseline.read_bytes()
    expected_summary = signature(json.loads(summary.read_text()))
    jobs = [dict(id=f'board-{i}', input=str(board), output=str(tmp / f'batch-{i}.kicad_pcb'), work=10000, seed=1)
            for i in range(8)]
    manifest = tmp / 'jobs.json'
    manifest.write_text(json.dumps(jobs))
    report = tmp / 'result.json'
    p = run([args.binary, manifest, '--no-gpu', '--no-rescue', '--no-why-open', '--workers', '2', '--report', report])
    if p.returncode not in (0, 3):
        raise RuntimeError(p.stderr[-3000:])
    results = json.loads(report.read_text())
    if not results['complete'] or results['workers'] != 2 or len(results['boards']) != 8:
        raise ValueError('Batch completion/worker bound differs')
    for job, result in zip(jobs, results['boards'], strict=True):
        if not result['processed'] or Path(job['output']).read_bytes() != expected or signature(result['summary']) != expected_summary:
            actual_signature = signature(result.get('summary', {}))
            differences = {k: [v, actual_signature.get(k)] for k, v in expected_summary.items() if actual_signature.get(k) != v}
            raise ValueError('Output bytes or work differ: ' + job['id'] + ' ' + json.dumps(differences) +
                             ' bytes_equal=' + str(Path(job['output']).read_bytes() == expected))
        for extension in ['.kicad_pro', '.kicad_dru']:
            reference = baseline.with_suffix(extension)
            actual = Path(job['output']).with_suffix(extension)
            if reference.exists() != actual.exists() or reference.exists() and reference.read_bytes() != actual.read_bytes():
                raise ValueError('Project sidecar differs')
    progress = [json.loads(line) for line in Path(str(report)+'.progress.jsonl').read_text().splitlines()]
    if {x['id'] for x in progress} != {x['id'] for x in jobs}:
        raise ValueError('Progress missed a board')
    before = hashlib.sha256(board.read_bytes()).hexdigest()
    invalid = [dict(jobs[0], output=str(board))]
    manifest.write_text(json.dumps(invalid))
    p = run([args.binary, manifest, '--no-gpu', '--report', tmp / 'invalid.json'])
    if p.returncode == 0 or (tmp/'invalid.json').exists() or hashlib.sha256(board.read_bytes()).hexdigest() != before:
        raise ValueError('Input overwrite was not rejected')
    invalid = [dict(jobs[0], output=str(tmp/'collision.kicad_pcb')),
               dict(jobs[1], output=str(tmp/'collision.kicad_pcb'))]
    manifest.write_text(json.dumps(invalid))
    p = run([args.binary, manifest, '--no-gpu', '--report', tmp/'collision.json'])
    if p.returncode == 0 or (tmp/'collision.kicad_pcb').exists():
        raise ValueError('Output collision was not rejected')
print(json.dumps(dict(complete=True, boards=8, workers=2, output_bytes_exact=True,
                     work_exact=True, project_bytes_exact=True, input_overwrite_rejected=True,
                     output_collision_rejected=True)))
