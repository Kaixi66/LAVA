"""Observe revised V6.3 production startup without changing or restarting training."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import subprocess
import time


FIELDS = ('Loss', 'Loss_Flow', 'Loss_Future', 'Loss_LAVA', 'Lambda_LAVA',
          'Grad_Norm', 'Grad_Norm_LAVA_Branch', 'FiLM_Grad_Norm',
          'FiLM_Gamma_RMS', 'FiLM_Beta_RMS', 'FiLM_Context_Count',
          'LAVA_Samples', 'State_Endpoint_Error', 'GPU_Peak_Allocated_GB',
          'Episode_Balanced_Mode', 'LAVA_Sampled_Anchors', 'LAVA_Encoded_Anchors',
          'LAVA_Scored_Anchors', 'LAVA_No_Negative_Anchors', 'LAVA_No_Same_Episode', 'LAVA_No_Cross_Episode',
          'Same_Episode_Negative_Count', 'Cross_Episode_Negative_Count',
          'Same_Episode_Negative_Min', 'Same_Episode_Negative_Max',
          'Cross_Episode_Negative_Min', 'Cross_Episode_Negative_Max',
          'Negative_Candidate_Count', 'Negative_Candidate_Availability', 'Order_Negatives')


def audit_rows(rows):
    errors = []
    values = {key: [float(row[key]) for row in rows] for key in FIELDS}
    for key, numbers in values.items():
        if not all(math.isfinite(x) for x in numbers):
            errors.append('Non-finite core metric: ' + key)
    for i in range(len(rows)):
        if values['Episode_Balanced_Mode'][i] != 1:
            errors.append('Wrong negative mode at row ' + str(i))
        encoded = values['LAVA_Encoded_Anchors'][i]
        scored = values['LAVA_Scored_Anchors'][i]
        skipped = values['LAVA_No_Negative_Anchors'][i]
        if encoded != values['LAVA_Samples'][i] or encoded != values['LAVA_Sampled_Anchors'][i] or scored + skipped != encoded:
            errors.append('Broken sampled/encoded/scored accounting at row ' + str(i))
        if not 0 <= scored <= encoded or (scored == 0 and values['Loss_LAVA'][i] != 0):
            errors.append('Invalid skipped-anchor loss at row ' + str(i))
        for source in ['Same', 'Cross']:
            for suffix in ['Count', 'Min', 'Max']:
                if not 0 <= values[f'{source}_Episode_Negative_{suffix}'][i] <= 4:
                    errors.append('Candidate count outside 0..4 at row ' + str(i))
        if abs(values['Negative_Candidate_Count'][i] - values['Same_Episode_Negative_Count'][i] - values['Cross_Episode_Negative_Count'][i]) > 3e-6:
            errors.append('Extra negative family at row ' + str(i))
        if values['Order_Negatives'][i] != 0:
            errors.append('Unexpected order candidates at row ' + str(i))
    active = [i for i, n in enumerate(values['LAVA_Samples']) if n > 0]
    if not active:
        errors.append('LAVA branch has no active samples')
    elif len(active) < .99 * len(rows):
        errors.append('Fewer than 99% of startup steps have LAVA samples; inspect sampling')
    if any(values['FiLM_Context_Count'][i] != values['LAVA_Samples'][i] for i in range(len(rows))):
        errors.append('FiLM context count does not match LAVA anchors')
    if active and not any(values['Loss_LAVA'][i] > 0 for i in active):
        errors.append('Active samples but all logged LAVA losses are zero')
    if not any(x > 0 for x in values['FiLM_Grad_Norm']):
        errors.append('No positive FiLM gradient at CSV precision')
    if not any(x > 0 for x in values['FiLM_Gamma_RMS']) or not any(x > 0 for x in values['FiLM_Beta_RMS']):
        errors.append('FiLM gamma or beta remains zero throughout observation')
    if any(x > 1e-4 for x in values['State_Endpoint_Error']):
        errors.append('Logged telescoping error exceeds 1e-4')
    scales = sorted({int(float(row['Scale_Mean'])) for row in rows})
    if scales != [1, 2, 4, 8, 16]:
        errors.append('Not all five scales observed')
    spikes = [dict(step=int(row['Global_Step']), scale=float(row['Scale_Mean']),
                   branch_grad=float(row['Grad_Norm_LAVA_Branch']), film_grad=float(row['FiLM_Grad_Norm']))
              for row in rows if float(row['Grad_Norm_LAVA_Branch']) > 1]
    return dict(passed=not errors, errors=errors, rows=len(rows),
                active_steps=len(active), empty_lava_steps=len(rows)-len(active),
                total_lava_anchors=sum(values['LAVA_Samples']), scales=scales,
                total_scored_anchors=sum(values['LAVA_Scored_Anchors']),
                total_skipped_anchors=sum(values['LAVA_No_Negative_Anchors']),
                stats={key: dict(min=min(nums), median=median(nums), max=max(nums)) for key, nums in values.items()},
                branch_gradient_spikes_gt1=spikes,
                scope='CSV checks cover the observed training prefix only. They do not prove full-run stability or replace semantic FiLM tests. Finite gradient spikes are reported, not silently removed.')


def job_state(job_id):
    result = subprocess.run(['squeue', '-h', '-j', job_id, '-o', '%T'], text=True, capture_output=True)
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip().splitlines()[0]
    # squeue may return nonzero for a job already removed from the live queue.
    # Resolve its state through accounting before treating this as unavailable.
    result = subprocess.run(['sacct', '-X', '-n', '-P', '-j', job_id, '--format=JobID,State'], text=True, capture_output=True, check=True)
    for line in result.stdout.splitlines():
        fields = line.split('|')
        if fields[0] == job_id:
            return fields[1].split()[0]
    return 'UNKNOWN'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--min-steps', type=int, default=200)
    parser.add_argument('--wait-seconds', type=int, default=3600)
    args = parser.parse_args()
    deadline = time.monotonic() + args.wait_seconds
    report = dict(train_job=args.job_id, required_steps=args.min_steps, status='waiting_for_training_logs')
    terminal = {'CANCELLED', 'FAILED', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE'}
    while True:
        try:
            state = job_state(args.job_id)
        except subprocess.CalledProcessError as exc:
            # A scheduler observation failure is not a training failure.
            state = 'OBSERVATION_UNAVAILABLE'
            report['scheduler_observation_error'] = str(exc)
        report['train_state'] = state
        if state in terminal:
            report.update(status='training_terminal_before_startup_audit', passed=False)
            break
        files = list(args.run_dir.glob('sft*/log/train_loss_*.csv'))
        if len(files) > 1:
            report.update(status='ambiguous_training_logs_require_review', passed=False, csv_files=[str(p) for p in files])
            break
        if files:
            with files[0].open() as stream:
                rows = []
                for row in csv.DictReader(stream):
                    if any(row.get(key) is None for key in FIELDS):
                        break  # A concurrent CSV write may not yet be complete.
                    rows.append({key: row[key] for key in (*FIELDS, 'Global_Step', 'Scale_Mean')})
                    if len(rows) >= args.min_steps:
                        break
            report.update(csv=str(files[0]), observed_rows=len(rows))
            if len(rows) >= args.min_steps and state in {'RUNNING', 'COMPLETING', 'COMPLETED', 'SUSPENDED'}:
                report.update(audit_rows(rows))
                provenance = args.run_dir / 'provenance'
                expected = json.loads((provenance / 'source_sha256.json').read_text())
                mismatches = [name for name, digest in expected.items()
                              if hashlib.sha256((provenance / 'source' / name).read_bytes()).hexdigest() != digest]
                report['source_hash_mismatches'] = mismatches
                report['observed_at_unix'] = time.time()
                if mismatches:
                    report['errors'].append('Frozen source hash mismatch')
                    report['passed'] = False
                report['status'] = 'observed_training_passed' if report['passed'] else 'observed_training_failed'
                break
        if state == 'COMPLETED':
            report.update(status='training_completed_with_insufficient_logged_steps', passed=False)
            break
        if time.monotonic() >= deadline:
            report.update(status='observation_window_elapsed', passed=None)
            # This is an observation timeout, never a reason to restart training.
            break
        print(json.dumps(report), flush=True)
        time.sleep(30)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    if report.get('passed') is False:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
