import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import pandas as pd
from evaluate_spectra import load_pair, metrics


def main():
    p=argparse.ArgumentParser(description='Select a checkpoint by spectral validation.')
    p.add_argument('--checkpoint-dir',required=True);p.add_argument('--meta',required=True)
    p.add_argument('--dataset',required=True);p.add_argument('--event-table',required=True)
    p.add_argument('--output-dir',required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--batch-size',type=int,default=2);p.add_argument('--num-workers',type=int,default=0)
    a=p.parse_args();out=Path(a.output_dir)
    if (out/'selected_checkpoint.json').exists():raise RuntimeError('Selection already frozen; use a new output directory')
    data=pd.read_parquet(a.dataset,columns=['split'])
    if set(data.split)!={'val'}:raise ValueError('Checkpoint selection requires validation rows only')
    prefix=Path(a.meta).stem
    checkpoints=[]
    for cp in Path(a.checkpoint_dir).glob(prefix+'.*.model'):
        match=re.fullmatch(re.escape(prefix)+r'\.(\d+)\.model',cp.name)
        if match:checkpoints.append((int(match[1]),cp))
    if not checkpoints:raise ValueError('No checkpoints match the supplied meta file')
    out.mkdir(parents=True,exist_ok=True);records=[]
    for epoch,cp in sorted(checkpoints):
        npz=out/f'epoch_{epoch:08d}.npz';report=npz.with_suffix('.json')
        subprocess.run([sys.executable,str(Path(__file__).with_name('predict.py')),
            '--dataset',a.dataset,'--event-table',a.event_table,'--checkpoint',str(cp),'--meta',a.meta,
            '--output-npz',str(npz),'--output-report',str(report),'--device',a.device,
            '--batch-size',str(a.batch_size),'--num-workers',str(a.num_workers),'--eval-id-column','eval_id',
            '--target-free'],check=True)
        frame,ids,pred,true=load_pair(npz,a.dataset)
        records.append({'epoch':epoch,'checkpoint':str(cp),
            'validation_SDP':float(metrics(pred,true)['SDP'].mean()),'n':len(ids)})
    best=sorted(records,key=lambda r:(-r['validation_SDP'],r['epoch']))[0]
    frozen={'selection':'maximum full-validation physical SDP, earliest epoch on exact tie',
            'selected':best,'candidates':records,'meta':a.meta,
            'validation_dataset':a.dataset}
    (out/'selected_checkpoint.json').write_text(json.dumps(frozen,indent=2)+'\n')
    print('Selected',best['checkpoint'])
if __name__=='__main__':main()
