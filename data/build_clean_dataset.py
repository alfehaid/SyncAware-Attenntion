"""Build the NSL-KDD set with LABEL-BLIND synchronization features.

Governing rule: the `label` column is never read at any step of feature
generation. Every synchronization feature is derived only from timing and
network properties of the record. A randomized control block (`rnd_*`) is
emitted alongside, generated the same way but from shuffled inputs, so that
any reported gain can be checked against a no-signal baseline.
"""
import numpy as np, pandas as pd, hashlib, sys

COLS = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
"wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
"root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files",
"num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate",
"srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
"srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
"dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
"dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate",
"dst_host_srv_rerror_rate","label","difficulty"]

SYNC_COLS = ["sync_delay_ms","sync_jitter_ms","sync_accuracy_score",
             "update_frequency_hz","convergence_time_ms","sync_reliability",
             "temporal_consistency","sync_status_binary"]

def load(path):
    df = pd.read_csv(path, names=COLS)
    df["is_attack"] = (df["label"].str.strip() != "normal").astype(int)
    return df

def make_sync(df, seed, randomize=False):
    """Derive synchronization features from timing/network properties only.

    Never touches df['label'] or df['is_attack'].
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    if randomize:   # التجربة الضابطة: ضوضاء خالصة
        out = pd.DataFrame({c: rng.normal(0, 1, n) for c in SYNC_COLS})
        out["sync_status_binary"] = rng.integers(0, 2, n)
        return out
    dur  = df["duration"].to_numpy(float)
    cnt  = df["count"].to_numpy(float)
    srv  = df["srv_count"].to_numpy(float)
    sb   = df["src_bytes"].to_numpy(float)
    db   = df["dst_bytes"].to_numpy(float)
    load_ = np.log1p(cnt + srv)
    # تأخير المزامنة: يزيد بالحمل وبطول الاتصال، بضوضاء لوغ-طبيعية
    delay = 12.0 + 6.5*load_ + 0.35*np.log1p(dur)*10 + rng.lognormal(0, 0.45, n)*8
    jitter = np.abs(rng.normal(0, 1, n)) * (0.18*delay)
    freq = np.clip(1000.0/np.maximum(delay, 1.0), 0.5, 200.0)
    conv = delay*2.1 + rng.normal(0, 5, n)
    acc  = np.clip(1.0 - (delay/1000.0) - (jitter/500.0) + rng.normal(0, 0.02, n), 0, 1)
    rel  = np.clip(1.0 - jitter/np.maximum(delay, 1.0) + rng.normal(0, 0.03, n), 0, 1)
    vol  = np.log1p(sb + db)
    temp = np.clip(1.0 - (vol/vol.max())*0.3 - (jitter/np.maximum(delay,1.0))*0.4 + rng.normal(0,0.03,n), 0, 1)
    return pd.DataFrame({
        "sync_delay_ms": delay, "sync_jitter_ms": jitter, "sync_accuracy_score": acc,
        "update_frequency_hz": freq, "convergence_time_ms": conv, "sync_reliability": rel,
        "temporal_consistency": temp, "sync_status_binary": (delay < 60).astype(int)})

if __name__ == "__main__":
    base = "/private/tmp/claude-502/-Users-haith/dbe9baf8-3387-482f-a0b5-b8e1415ec4c4/scratchpad/syncrepo/data/real_datasets/NSL-KDD"
    tr = load(f"{base}/KDDTrain+.txt"); te = load(f"{base}/KDDTestPlus.txt")
    for name, df, seed in (("train", tr, 20261006), ("test", te, 20261007)):
        s  = make_sync(df, seed)
        sr = make_sync(df, seed+1, randomize=True).add_prefix("rnd_")
        out = pd.concat([df.drop(columns=["difficulty"]), s, sr], axis=1)
        out.to_csv(f"nslkdd_{name}_cleansync.csv", index=False)
        print(f"{name}: {out.shape}  هجوم={out['is_attack'].mean()*100:.1f}%")
    # تحقّق: لا سمة مزامنة ترتبط بالتصنيف
    d = pd.read_csv("nslkdd_train_cleansync.csv")
    print("\nارتباط سمات المزامنة بالهدف (يجب أن تكون قريبة من الصفر):")
    for c in SYNC_COLS:
        print(f"   {c:22} r={np.corrcoef(d[c], d['is_attack'])[0,1]:+.3f}")
