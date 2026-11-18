"""One-command execution with per-stage transcripts and an explicit completion record."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
from datetime import datetime,timezone

root=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser()
ap.add_argument('--report-only',action='store_true',help='Regenerate outputs and compile the report from completed experiments; no training.')
ap.add_argument('--resume-ablation',action='store_true',help='Allow verified-signature ablation repeat checkpoints.')
ap.add_argument('--dry-run',action='store_true',help='Print stage order without executing or changing files.')
args=ap.parse_args()
analysis=['00_fetch_data.py','01_audit.py','02_predictive.py','03_explanations.py','04_stability.py','07_ablation.py']
outputs=['05_figures_tables.py','08_extension_outputs.py']
if not args.report_only:
    outputs += ['06_checks.py','09_extended_checks.py']
outputs += ['11_check_saved_results.py']
stages=[]
for file in ([] if args.report_only else analysis)+outputs:
    extra=['--resume'] if file=='07_ablation.py' and args.resume_ablation else []
    stages.append((file,[sys.executable,str(root/'tools/run_python.py'),str(root/'scripts'/file),*extra]))
stages.append(('build_report',[sys.executable,str(root/'tools/build_report.py')]))
if args.dry_run:
    for name,_ in stages:print(name)
    raise SystemExit(0)
logs=root/'results/logs';logs.mkdir(parents=True,exist_ok=True)
state={'started_utc':datetime.now(timezone.utc).isoformat(),'mode':'report-only' if args.report_only else 'full','status':'running','completed_stages':[]}
state_path=root/'results/pipeline_status.json'
def save():state_path.write_text(json.dumps(state,indent=2))
save()
for name,command in stages:
    print(f'Running {name}',flush=True)
    state['current_stage']=name;save()
    with (logs/f'pipeline_{name}.out').open('w') as log:
        process=subprocess.Popen(command,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in process.stdout:
            log.write(line);log.flush();print(line,end='',flush=True)
        rc=process.wait()
    if rc:
        state.update(status='failed',returncode=rc);save()
        raise SystemExit(rc)
    state['completed_stages'].append(name);save()
state.update(status='complete',finished_utc=datetime.now(timezone.utc).isoformat())
state.pop('current_stage',None);save()
print('Pipeline complete.',flush=True)
