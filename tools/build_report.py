"""Compile the working research report using project-local Tectonic or an installed LaTeX toolchain."""
from pathlib import Path
import os
import shutil
import subprocess

root=Path(__file__).resolve().parents[1]
out=root/'output/pdf'
out.mkdir(parents=True,exist_ok=True)
env=os.environ.copy()
env['TECTONIC_CACHE_DIR']=str(root/'.cache/tectonic')
tectonic=root/'tools/tectonic'
for source,filename in [('report.tex','Research_Notes.pdf'),('submission.tex','Report.pdf')]:
    if tectonic.exists():
        command=[str(tectonic),'--untrusted','--keep-logs','--outdir',str(out),source]
    elif shutil.which('tectonic'):
        command=['tectonic','--untrusted','--keep-logs','--outdir',str(out),source]
    elif shutil.which('latexmk'):
        command=['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error',f'-outdir={out}',source]
    else:
        raise SystemExit('Install Tectonic (macOS helper: python3 tools/install_tectonic.py) or latexmk; see README.')
    with (root/'results/logs'/f'build_{Path(source).stem}.out').open('w') as log:
        process=subprocess.Popen(command,cwd=root/'report',env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in process.stdout:
            log.write(line);log.flush()
            if not line.startswith('note: downloading '):print(line,end='',flush=True)
        rc=process.wait()
    if rc:raise SystemExit(rc)
    made=out/(Path(source).stem+'.pdf')
    shutil.copy2(made,out/filename);made.unlink()
    print('Built:',out/filename)
