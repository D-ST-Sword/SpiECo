"""CH-SIMS normalization fitted only on valid training frames."""
import numpy as np


def normalize_records(records):
    statistics={}
    for modality in ('audio','vision'):
        training=np.asarray(records['train'][modality],np.float64)
        lengths=np.asarray(records['train'][modality+'_lengths'],np.int64)
        valid=np.arange(training.shape[1])[None,:]<lengths[:,None]
        values=training[valid]
        assert np.isfinite(values).all()
        mean=values.mean(0)
        std=np.maximum(values.std(0),1e-5)
        statistics[modality]=dict(mean=mean.tolist(),std=std.tolist(),valid_training_frames=int(valid.sum()))
        for record in records.values():
            x=np.asarray(record[modality],np.float64)
            n=np.asarray(record[modality+'_lengths'],np.int64)
            mask=np.arange(x.shape[1])[None,:]<n[:,None]
            x=(x-mean)/std
            x[~mask]=0
            assert np.isfinite(x).all()
            record[modality]=x.astype(np.float32)
    return statistics
