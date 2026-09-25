import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def metrics(prediction, target):
    pred, true = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if pred.shape != true.shape or pred.ndim != 2 or pred.shape[1] != 512:
        raise ValueError('Expected matching [N,512] arrays')
    if not np.isfinite(pred).all() or not np.isfinite(true).all() or np.any(pred < 0) or np.any(true < 0) or np.any(true.sum(axis=1) <= 0):
        raise ValueError('Invalid intensity or empty reference spectrum')
    def cosine(a,b):
        return np.sum(a*b,axis=1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)+1.e-10)
    mass = np.arange(1,513,dtype=np.float64)[None,:]
    zero = np.all(pred == 0,axis=1)
    return {'DP':cosine(np.sqrt(pred),np.sqrt(true)),
            'SDP':cosine(mass**3*pred**0.6,mass**3*true**0.6),
            'COS':cosine(pred,true),
            'Top_1':((np.argmax(pred,axis=1)==np.argmax(true,axis=1)) & ~zero).astype(np.float64)}


def load_pair(npz_path, dataset_path):
    from clef.msutil import binutils
    frame = pd.read_parquet(dataset_path)
    if not {'eval_id','molecule_id','spect'}.issubset(frame.columns):
        raise ValueError('Dataset must carry eval_id, molecule_id and spect')
    if frame.eval_id.duplicated().any():
        raise ValueError('Duplicate eval_id')
    with np.load(npz_path,allow_pickle=False) as z:
        ids, pred = z['eval_ids'],z['pred_spect']
    if len(ids)!=len(set(ids.tolist())) or set(ids.tolist())!=set(frame.eval_id.tolist()):
        raise ValueError('Predictions must cover the dataset exactly once')
    frame = frame.set_index('eval_id').loc[ids]
    if 'is_observed_spectrum' in frame and not frame.is_observed_spectrum.all():

        kinds=set(frame.spectrum_kind)
        if not kinds.issubset({'synthetic_example'}):
            raise ValueError('Cannot score placeholder targets')
    bins=binutils.create_spectrum_bins(first_bin_center=1.,bin_width=1.,bin_number=512)
    truth=np.stack([bins.histogram(np.asarray(list(s), dtype=np.float64)[:,0],np.asarray(list(s), dtype=np.float64)[:,1])[2] for s in frame.spect])
    return frame,ids,pred,truth


def main():
    p=argparse.ArgumentParser(description='Evaluate paired EI spectra.')
    p.add_argument('--predictions',required=True);p.add_argument('--dataset',required=True)
    p.add_argument('--output-dir',required=True);a=p.parse_args()
    frame,ids,pred,true=load_pair(a.predictions,a.dataset)
    result=metrics(pred,true);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    rows=pd.DataFrame({'eval_id':ids,'molecule_id':frame.molecule_id.to_numpy(),**result})
    rows.to_parquet(out/'per_molecule_metrics.parquet',index=False)
    summary={'n':len(rows),'metrics':{k:float(v.mean()) for k,v in result.items()},'zero_predictions':int(np.all(pred==0,axis=1).sum())}
    (out/'metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))
if __name__=='__main__':main()
