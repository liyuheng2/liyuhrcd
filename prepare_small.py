"""Convert published features; hold out validation groups without test tuning."""
import argparse, json, pickle, hashlib
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from prepare import atomic_cache, pooled

ROOT=Path('data/raw')
OUT=Path('data/features')

def save(name, split, views, labels, ids, groups):
    views=[np.asarray(x,dtype=np.float32) for x in views]
    mask=np.stack([np.isfinite(x).all(1)&(x!=0).any(1) for x in views],1)
    keep=mask.any(1)
    out=OUT/name; out.mkdir(parents=True,exist_ok=True)
    atomic_cache(out/f'{split}.npz',[np.nan_to_num(x[keep]) for x in views],mask[keep],np.asarray(labels,dtype=np.int64)[keep],np.asarray(ids)[keep],np.asarray(groups)[keep])
    print(name,split,'n=',sum(keep),'dims=',[x.shape[1] for x in views],flush=True)

def mmsa(name):
    path=ROOT/name/'Processed'/('aligned_50.pkl' if name=='cmu_mosi' else 'unaligned_39.pkl')
    with path.open('rb') as f: source=pickle.load(f)
    details={}; blocks=[]
    for split,key in zip(('train','val','test'),('train','valid','test')):
        block=source[key]
        labels=np.sign(np.asarray(block['regression_labels']).reshape(-1)).astype(np.int64)+1
        views=[pooled(block[k],block.get(k+'_lengths'))[0] for k in ('text','audio','vision')]
        ids=np.asarray(block['id']).astype(str)
        groups=np.array([x.split('$_$')[0] for x in ids])
        blocks.append((split,views,labels,ids,groups))
    group_sets=[set(b[4]) for b in blocks]
    overlap=any(group_sets[i]&group_sets[j] for i in range(3) for j in range(i))
    if overlap:
        if name!='ch_sims':raise ValueError('Unexpected group overlap in MOSI source splits')
        views=[np.concatenate([b[1][i] for b in blocks]) for i in range(3)]
        labels=np.concatenate([b[2] for b in blocks]);ids=np.concatenate([b[3] for b in blocks]);groups=np.concatenate([b[4] for b in blocks])
        if len(set(ids))!=len(ids):raise ValueError('Duplicate source sample IDs')
        folds=[hold for _,hold in StratifiedGroupKFold(5,shuffle=True,random_state=20261122).split(np.zeros(len(labels)),labels,groups)]
        blocks=[(split,[x[index] for x in views],labels[index],ids[index],groups[index]) for split,index in zip(
            ('train','val','test'),(np.concatenate(folds[:3]),folds[3],folds[4]))]
        details['split_override']='Source-video overlap in provided splits: deterministic 5-fold stratified grouped split, 3 folds train / 1 val / 1 test, seed 20261122. Group sizes make actual counts unequal to 60/20/20. Not official split results.'
        print('CH-SIMS: using fixed source-video-disjoint split due to overlap in published partition.',flush=True)
    for split,views,labels,ids,groups in blocks:
        save(name,split,views,labels,ids,groups)
        details[split]={'n':len(labels),'class_counts':np.bincount(labels,minlength=3).tolist()}
    return details

def dialogue(name):
    path=ROOT/name/('MELD_features_raw.pkl' if name=='meld_small' else 'IEMOCAP_features.pkl')
    with path.open('rb') as f: data=pickle.load(f,encoding='latin1')
    if name=='iemocap':
        ids,speakers,labels,text,audio,video,sentences,train,test=data
        modalities=[text,audio,video]
    else:
        ids,speakers,labels,text,audio,sentences,train,test,sentiment=data
        modalities=[text,audio]
    train=sorted(train);test=sorted(test)
    # Split by source video for IEMOCAP; by dialogue for MELD.
    def group(k):
        return str(k).split('_')[0] if name=='iemocap' else str(k)
    groups=[group(k) for k in train]
    a,b=next(GroupShuffleSplit(n_splits=1,test_size=.1,random_state=20261122).split(train,groups=groups))
    partitions={'train':[train[i] for i in a],'val':[train[i] for i in b],'test':test}
    details={}
    for split,keys in partitions.items():
        views=[np.concatenate([np.asarray(m[k])[:len(labels[k])] for k in keys]) for m in modalities]
        y=np.concatenate([labels[k] for k in keys])
        sample_ids=[f'{k}:{i}' for k in keys for i in range(len(labels[k]))]
        sample_groups=[group(k) for k in keys for _ in labels[k]]
        save(name,split,views,y,sample_ids,sample_groups)
        details[split]={'n':len(y),'class_counts':np.bincount(y).tolist()}
    return details

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=('cmu_mosi','ch_sims','meld_small','iemocap'));args=p.parse_args()
    name=args.dataset
    details=mmsa(name) if name in ('cmu_mosi','ch_sims') else dialogue(name)
    (OUT/name/'provenance.json').write_text(json.dumps({'dataset':name,'splits':details,'classification':'sign(score)+1 for MMSA; published emotion labels for dialogue datasets','validation':details.get('split_override','MMSA provided validation, otherwise fixed 10% grouped split of published train set'),'scope':'pre-extracted features, feature-space corruption only'},indent=2),encoding='utf-8')
