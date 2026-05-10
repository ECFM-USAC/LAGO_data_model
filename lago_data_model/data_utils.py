import numpy as np
import pandas as pd
from tqdm import tqdm


# ===================================================================
# Helpers para construir matrices de señal (entrada al modelo MLP/CNN).
# Usados desde MLP_model.ipynb e inference.ipynb.
# ===================================================================

def cut_around_min(seg, n_before, n_after):
    """Recorta el segmento alrededor del mínimo: [kmin-n_before, kmin+n_after]."""
    if len(seg) == 0:
        return seg
    kmin = int(np.argmin(seg))
    a = max(0, kmin - n_before)
    b = min(len(seg), kmin + n_after + 1)
    return seg[a:b]


def align_and_normalize(seg, normalize=True, scale="robust", eps=1e-9):
    """Normaliza por segmento: resta mediana y opcionalmente escala (maxabs o robust p5-p95)."""
    s = seg.astype(float).copy()
    if normalize:
        s = s - np.median(s)
    if scale == "maxabs":
        m = np.max(np.abs(s))
        if m > eps:
            s = s / m
    elif scale == "robust":
        p5, p95 = np.percentile(s, [5, 95])
        rng = p95 - p5
        if rng > eps:
            s = (s - p5) / rng
    return s


def resample_to_fixed(seg, out_len):
    """Interpolación lineal a longitud fija out_len."""
    L = len(seg)
    if L == 0:
        return np.zeros(out_len)
    if L == out_len:
        return seg
    return np.interp(np.linspace(0, 1, out_len), np.linspace(0, 1, L), seg)


def signal_to_vector(values, n_before=40, n_after=87, resample_len=128,
                     normalize=True, scale="robust"):
    """Aplica el pipeline de pre-procesamiento a UNA señal de pulso."""
    seg = np.asarray(values).astype(float)
    seg = cut_around_min(seg, n_before=n_before, n_after=n_after)
    seg = align_and_normalize(seg, normalize=normalize, scale=scale)
    seg = resample_to_fixed(seg, out_len=resample_len)
    return seg.astype(np.float32)


def segments_to_matrix(segments, n_before=40, n_after=87, resample_len=128,
                       normalize=True, scale="robust"):
    """Convierte una lista de segments (dicts con 'values' o arrays planos) en matriz (N, resample_len)."""
    X = np.zeros((len(segments), resample_len), dtype=np.float32)
    for i, s in enumerate(segments):
        values = s["values"] if isinstance(s, dict) else s
        X[i] = signal_to_vector(values, n_before=n_before, n_after=n_after,
                                resample_len=resample_len, normalize=normalize, scale=scale)
    return X


# ===================================================================
# Extracción de pulsos desde la señal cruda
# ===================================================================

def extract_pulses(
    y: np.ndarray,
    nsigma_threshold: float = 3.0,
    crossback_nsigma: float = 1.0,
    min_len: int = 3,
    baseline_win: int = 25
) -> pd.DataFrame:
    """
    Detects downward pulses (dips) and extracts features for each pulse.
    - Threshold = median(y) - nsigma * MAD
    - A pulse is a contiguous region below Threshold, minimum length min_len
    - Start/End refined where the signal crosses back above (median - crossback_nsigma*MAD)
    - Baseline per pulse = median of a local window before & after the pulse
    
    Returns a DataFrame: one row per pulse, with features.
    """
    y = np.asarray(y).astype(float)
    n = len(y)
    # Robust baseline stats
    med = np.median(y)
    mad = np.median(np.abs(y - med)) + 1e-9  # avoid divide-by-zero
    thr = med - nsigma_threshold * mad
    cross_thr = med - crossback_nsigma * mad
    
    # Identify candidate regions (below thr)
    below = y < thr
    # Find contiguous regions
    idx = np.arange(n)
    pulses = []
    in_region = False
    start = None
    for i, b in enumerate(below):
        if b and not in_region:
            in_region = True
            start = i
        elif not b and in_region:
            end = i - 1
            if end - start + 1 >= min_len:
                pulses.append([start, end])
            in_region = False
    if in_region:
        end = n - 1
        if end - start + 1 >= min_len:
            pulses.append([start, end])
    
    feats = []
    for (a, b) in pulses:
        # refine start: walk left from 'a' until previous sample > cross_thr
        i0 = a
        while i0 > 0 and y[i0-1] < cross_thr:
            i0 -= 1
        # refine end: walk right from 'b' until next sample > cross_thr
        i1 = b
        while i1 < n-1 and y[i1+1] < cross_thr:
            i1 += 1
        
        seg = y[i0:i1+1]
        seg_idx = np.arange(i0, i1+1)
        kmin_rel = int(np.argmin(seg))
        kmin = seg_idx[kmin_rel]
        ymin = float(y[kmin])
        
        # local baseline: median of a window before and after, avoiding overlap
        L = max(i0 - baseline_win, 0)
        R = min(i1 + baseline_win + 1, n)
        left = y[L:i0]
        right = y[i1+1:R]
        if len(left) + len(right) > 0:
            baseline_local = float(np.median(np.concatenate([left, right]) if len(left) and len(right) else (left if len(left) else right)))
        else:
            baseline_local = float(med)
        
        depth = baseline_local - ymin
        width = i1 - i0 + 1
        area = float(np.sum(np.clip(baseline_local - seg, a_min=0, a_max=None)))
        # FWHM: width where y <= baseline - depth/2 within [i0,i1]
        half = baseline_local - depth/2.0
        mask_half = seg <= half
        if np.any(mask_half):
            idx_half = np.where(mask_half)[0]
            fwhm = int(idx_half[-1] - idx_half[0] + 1)
        else:
            fwhm = 0
        
        # rise and recovery times (indices)
        rise_time = int(kmin - i0)
        recovery_time = int(i1 - kmin)
        
        feats.append({
            "start_idx": int(i0),
            "min_idx": int(kmin),
            "end_idx": int(i1),
            "min_value": ymin,
            "baseline_local": baseline_local,
            "depth": float(depth),
            "width_samples": int(width),
            "area_under_pulse": area,
            "fwhm_samples": int(fwhm),
            "rise_time": rise_time,
            "recovery_time": recovery_time,
            "snr_depth_over_mad": float(depth / mad)
        })
    
    return pd.DataFrame(feats)


def extract_pulses_and_segments(
    y: np.ndarray,
    nsigma_threshold: float = 3.0,
    crossback_nsigma: float = 1.0,
    min_len: int = 3,
    baseline_win: int = 25
):
    """
    Returns
    -------
    df : pd.DataFrame
        Mismas columnas que extract_pulses (1 fila por pulso).
    segments : list[dict]
        Lista con elementos {'start','end','min_idx','values'} por pulso.
    """
    df = extract_pulses(
        y=y,
        nsigma_threshold=nsigma_threshold,
        crossback_nsigma=crossback_nsigma,
        min_len=min_len,
        baseline_win=baseline_win
    )

    # 2) Construir los segmentos con los índices ya calculados
    segments = []
    if not df.empty:
        y = np.asarray(y).astype(float)
        for _, r in df.iterrows():
            i0, i1 = int(r.start_idx), int(r.end_idx)
            seg = y[i0:i1+1].copy()
            segments.append({
                "start": i0,
                "end": i1,
                "min_idx": int(r.min_idx),
                "values": seg
            })
    return df, segments


def process_dataframe_readings(
    data: pd.DataFrame,
    readings_col: str = "readings",
    **extract_kwargs
):
    """
    Aplica extract_pulses_and_segments a cada fila.
    Devuelve:
      - pulses_df: todas las features (1 fila por pulso) con 'row_id'
      - segments: lista de segmentos con 'row_id'
    """
    all_feats = []
    all_segments = []
    for idx, row in tqdm(data.iterrows(), total=len(data), desc="Pulsos por fila"):
        y = row[readings_col]
        df, segs = extract_pulses_and_segments(y, **extract_kwargs)
        if not df.empty:
            df.insert(0, "row_id", idx)
            all_feats.append(df)
            for s in segs:
                s["row_id"] = idx
            all_segments.extend(segs)

    pulses_df = pd.concat(all_feats, ignore_index=True) if all_feats else pd.DataFrame()
    return pulses_df, all_segments
