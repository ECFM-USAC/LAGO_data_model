import os
import torch
from datetime import datetime
from xgboost import XGBModel
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    accuracy_score
)

class EarlyStopping:
    def __init__(self, patience=5, delta=0, model_name=None):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.model_name = model_name

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Save model when validation loss decrease.'''
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        torch.save(model.state_dict(), f"checkpoint_timestamp.pt")

def eval_predictions(
    y_true,
    y_pred,
    threshold=0.5,
    labels=None,
    average='macro',
    multi_class='ovr',
    plot_cm=True,
    cm_normalize='true',   # 'true'|'pred'|'all'|None
    plot_roc=True,
    plot_pr=True,
):
    """
    Evalúa predicciones binarias o multiclase con métricas + gráficos:
    - PR/F1 (macro/weighted), classification_report
    - Matriz de confusión
    - ROC-AUC (binario y multiclase OvR: micro y macro)
    - Curvas ROC por clase + micro
    - Curvas Precision-Recall por clase + micro (AP)
    
    y_true: (n_samples,) etiquetas verdaderas (strings o ints)
    y_pred:
      * binario: prob. clase positiva (n_samples,)
      * multiclase: probs por clase (n_samples, n_classes)
      * etiquetas discretas (sin probas) también funciona, pero sin ROC/PR
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    le = LabelEncoder()
    y_true_enc = le.fit_transform(y_true)
    
    if labels is None:
        class_names = le.classes_
    else:
        class_names = np.asarray(labels)
    
    # Fuerza a string para evitar TypeError con numpy.int64
    class_names_str = [str(c) for c in class_names]
    
    n_classes = len(np.unique(y_true_enc))

    results = {}

    # Detectar formato de y_pred
    proba_available = False
    multiclass = n_classes > 2

    if y_pred.ndim == 1:
        # Puede ser: etiquetas discretas o probas binario
        if set(np.unique(y_pred)).issubset(set(range(n_classes))) and y_pred.dtype.kind in "iu":
            # y_pred son etiquetas (no probabilidades)
            y_pred_labels = y_pred.astype(int)
        else:
            # Se asume probas binario
            proba_available = True
            multiclass = False
            y_pred_labels = (y_pred > threshold).astype(int)
    elif y_pred.ndim == 2:
        # Multiclase con probas
        proba_available = True
        multiclass = True
        if y_pred.shape[1] != n_classes:
            raise ValueError("y_pred de shape (n_samples, n_classes). #clases no coincide con y_true.")
        y_pred_labels = np.argmax(y_pred, axis=1)
    else:
        raise ValueError("Formato de y_pred no soportado.")

    # ---- Métricas de clasificación ----
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_enc, y_pred_labels, average='macro', zero_division=0
    )
    prec_weight, rec_weight, f1_weight, _ = precision_recall_fscore_support(
        y_true_enc, y_pred_labels, average='weighted', zero_division=0
    )

    print("=== Métricas agregadas ===")
    print(f"Macro     -> P: {prec_macro:.4f} | R: {rec_macro:.4f} | F1: {f1_macro:.4f}")
    print(f"Weighted  -> P: {prec_weight:.4f} | R: {rec_weight:.4f} | F1: {f1_weight:.4f}")

    cr = classification_report(y_true_enc, y_pred_labels, target_names=class_names_str, zero_division=0)
    print("\n=== Classification report (por clase) ===")
    print(cr)

    results.update({
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "precision_weighted": prec_weight,
        "recall_weighted": rec_weight,
        "f1_weighted": f1_weight,
        "classification_report_str": cr,
        "y_pred_labels": y_pred_labels
    })

    # ---- Matriz de confusión ----
    if plot_cm:
        cm = confusion_matrix(y_true_enc, y_pred_labels, normalize=cm_normalize)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names_str)
        fig, ax = plt.subplots(figsize=(6,5))
        disp.plot(ax=ax, cmap="Blues", colorbar=True)
        ax.set_title("Matriz de confusión" + (f" (normalize={cm_normalize})" if cm_normalize else ""))
        plt.tight_layout(); plt.show()
        results["confusion_matrix"] = cm

    # Si no hay probabilidades no podemos trazar ROC/PR
    if not proba_available:
        return results

    # ---- ROC ----
    if not multiclass:
        fpr, tpr, thr = roc_curve(y_true_enc, y_pred, pos_label=1)
        roc_auc = auc(fpr, tpr)
        print(f"\nROC-AUC (binario): {roc_auc:.4f}")
        if plot_roc:
            plt.figure(figsize=(6,5))
            plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC={roc_auc:.3f})')
            plt.plot([0,1],[0,1], linestyle='--')
            plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
            plt.title('ROC (binario)')
            plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()
        results.update({"roc_auc": roc_auc, "roc_curve": {"fpr": fpr, "tpr": tpr, "thresholds": thr}})
    else:
        y_true_bin = label_binarize(y_true_enc, classes=np.arange(n_classes))
        fpr_d, tpr_d, auc_d = {}, {}, {}
        for c in range(n_classes):
            fpr_d[c], tpr_d[c], _ = roc_curve(y_true_bin[:,c], y_pred[:,c])
            auc_d[c] = auc(fpr_d[c], tpr_d[c])

        # micro y macro
        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_pred.ravel())
        auc_micro = auc(fpr_micro, tpr_micro)
        auc_macro = roc_auc_score(y_true_enc, y_pred, multi_class=multi_class, average="macro")
        print("\nROC-AUC multiclase:")
        print(f"  micro-average: {auc_micro:.4f}")
        print(f"  macro-average ({multi_class}): {auc_macro:.4f}")

        if plot_roc:
            plt.figure(figsize=(7,6))
            for c in range(n_classes):
                plt.plot(fpr_d[c], tpr_d[c], lw=1.4, label=f"Clase {class_names_str[c]} (AUC={auc_d[c]:.3f})")
            plt.plot(fpr_micro, tpr_micro, lw=2, linestyle='--', label=f"Micro-avg (AUC={auc_micro:.3f})")
            plt.plot([0,1],[0,1], linestyle=':', color='black')
            plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
            plt.title('ROC (multiclase, OvR)')
            plt.legend(loc='lower right', fontsize=8)
            plt.grid(True); plt.tight_layout(); plt.show()

        results.update({
            "roc_auc_micro": auc_micro,
            "roc_auc_macro": auc_macro,
            "roc_auc_per_class": auc_d
        })

    # ---- Precision-Recall ----
    if not multiclass:
        prec, rec, thr = precision_recall_curve(y_true_enc, y_pred, pos_label=1)
        ap = average_precision_score(y_true_enc, y_pred)
        print(f"\nAverage Precision (binario): {ap:.4f}")
        if plot_pr:
            plt.figure(figsize=(6,5))
            plt.plot(rec, prec, lw=2, label=f'PR (AP={ap:.3f})')
            plt.xlabel('Recall'); plt.ylabel('Precision')
            plt.title('Precision-Recall (binario)')
            plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()
        results.update({"ap": ap, "pr_curve": {"precision": prec, "recall": rec, "thresholds": thr}})
    else:
        y_true_bin = label_binarize(y_true_enc, classes=np.arange(n_classes))
        pr_d, rc_d, ap_d = {}, {}, {}
        for c in range(n_classes):
            prec_c, rec_c, _ = precision_recall_curve(y_true_bin[:,c], y_pred[:,c])
            ap_c = average_precision_score(y_true_bin[:,c], y_pred[:,c])
            pr_d[c], rc_d[c], ap_d[c] = prec_c, rec_c, ap_c

        # Micro-average PR
        prec_micro, rec_micro, _ = precision_recall_curve(y_true_bin.ravel(), y_pred.ravel())
        ap_micro = average_precision_score(y_true_bin, y_pred, average="micro")

        print("\nAverage Precision multiclase:")
        print(f"  micro-average AP: {ap_micro:.4f}")
        for c in range(n_classes):
            print(f"  clase {class_names_str[c]} AP: {ap_d[c]:.4f}")

        if plot_pr:
            plt.figure(figsize=(7,6))
            for c in range(n_classes):
                plt.plot(rc_d[c], pr_d[c], lw=1.4, label=f"Clase {class_names_str[c]} (AP={ap_d[c]:.3f})")
            plt.plot(rec_micro, prec_micro, lw=2, linestyle='--', label=f"Micro-avg (AP={ap_micro:.3f})")
            plt.xlabel('Recall'); plt.ylabel('Precision')
            plt.title('Precision-Recall (multiclase, OvR)')
            plt.legend(loc='lower left', fontsize=8)
            plt.grid(True); plt.tight_layout(); plt.show()

        results.update({
            "ap_micro": ap_micro,
            "ap_per_class": ap_d
        })

    return results

def compare_models(
    y_true,
    proba_dict=None,         # dict: {"Modelo": y_pred_proba}  -> (N,) binario o (N,C) multiclase
    labels_dict=None,        # dict: {"Modelo": y_pred_labels} -> (N,)
    class_labels=None,       # lista opcional de nombres de clases para mostrar
    average='macro',
    multi_class='ovr',
    plot_roc=True,
    plot_pr=True,
    show_table=True
):
    """
    Compara múltiples modelos con métricas agregadas y curvas ROC/PR superpuestas.
    - Soporta binario y multiclase.
    - Si hay probabilidades, se usan para ROC/PR; si no, se evalúan solo métricas de clasificación.

    Parámetros
    ----------
    y_true : array (N,)
    proba_dict : dict[str, np.ndarray] -> probabilidades por modelo
    labels_dict : dict[str, np.ndarray] -> etiquetas discretas por modelo
    class_labels : lista de nombres de clases (para impresión)
    average : 'macro'/'weighted' (para PR/F1 agregadas)
    multi_class : 'ovr' o 'ovo' (para ROC-AUC multiclase)
    """
    if proba_dict is None and labels_dict is None:
        raise ValueError("Proporciona al menos proba_dict o labels_dict.")

    y_true = np.asarray(y_true)
    le = LabelEncoder()
    y_true_enc = le.fit_transform(y_true)
    n_classes = len(np.unique(y_true_enc))
    is_multiclass = n_classes > 2

    # nombres de clase seguros (str)
    if class_labels is None:
        class_names = [str(c) for c in le.classes_]
    else:
        class_names = [str(c) for c in class_labels]

    # Unificar entradas en un diccionario de resultados por modelo
    models = set()
    if proba_dict: models |= set(proba_dict.keys())
    if labels_dict: models |= set(labels_dict.keys())
    models = sorted(models)

    rows = []          # para la tabla resumen
    reports = {}       # classification_report por modelo
    curves = {"roc": {}, "pr": {}}  # curvas por modelo

    # ------------------ BUCLE POR MODELO ------------------
    for name in models:
        # Probabilidades y etiquetas disponibles
        proba = proba_dict.get(name, None) if proba_dict else None
        yhat  = labels_dict.get(name, None) if labels_dict else None

        # Si hay proba, derivamos etiquetas (argmax / threshold)
        if proba is not None:
            proba = np.asarray(proba)
            if proba.ndim == 1:     # binario: prob. de clase positiva
                if is_multiclass:
                    raise ValueError(f"{name}: proba es 1D pero el problema es multiclase.")
                yhat_from_proba = (proba > 0.5).astype(int)
            elif proba.ndim == 2:
                if proba.shape[1] != n_classes:
                    raise ValueError(f"{name}: columnas de proba ({proba.shape[1]}) != n_classes ({n_classes}).")
                yhat_from_proba = np.argmax(proba, axis=1)
            else:
                raise ValueError(f"{name}: formato de probabilidades no soportado.")
            if yhat is None:
                yhat = yhat_from_proba

        if yhat is None:
            raise ValueError(f"{name}: se requieren etiquetas o probabilidades para evaluar.")

        yhat = np.asarray(yhat)

        # --- Métricas clasificatorias
        acc = accuracy_score(y_true_enc, yhat)
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true_enc, yhat, average='macro', zero_division=0)
        p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(y_true_enc, yhat, average='weighted', zero_division=0)
        report_str = classification_report(y_true_enc, yhat, target_names=class_names, zero_division=0)

        rows.append({
            "modelo": name,
            "accuracy": acc,
            "precision_macro": p_macro, "recall_macro": r_macro, "f1_macro": f1_macro,
            "precision_weighted": p_weight, "recall_weighted": r_weight, "f1_weighted": f1_weight
        })
        reports[name] = report_str

        # --- ROC / PR si hay probabilidades
        if proba is not None:
            if not is_multiclass:
                # ROC
                fpr, tpr, _ = roc_curve(y_true_enc, proba, pos_label=1)
                auc_roc = auc(fpr, tpr)
                curves["roc"][name] = {"fpr": fpr, "tpr": tpr, "auc": auc_roc}
                # PR
                prec, rec, _ = precision_recall_curve(y_true_enc, proba, pos_label=1)
                ap = average_precision_score(y_true_enc, proba)
                curves["pr"][name] = {"precision": prec, "recall": rec, "ap": ap}
            else:
                # multiclase: micro/macro
                y_true_bin = label_binarize(y_true_enc, classes=np.arange(n_classes))
                # ROC micro
                fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), proba.ravel())
                auc_micro = auc(fpr_micro, tpr_micro)
                # ROC macro (usamos roc_auc_score)
                auc_macro = roc_auc_score(y_true_enc, proba, multi_class=multi_class, average="macro")
                curves["roc"][name] = {
                    "fpr_micro": fpr_micro, "tpr_micro": tpr_micro,
                    "auc_micro": auc_micro, "auc_macro": auc_macro
                }
                # PR micro
                prec_micro, rec_micro, _ = precision_recall_curve(y_true_bin.ravel(), proba.ravel())
                ap_micro = average_precision_score(y_true_bin, proba, average="micro")
                curves["pr"][name] = {
                    "precision_micro": prec_micro, "recall_micro": rec_micro,
                    "ap_micro": ap_micro
                }

    # ------------------ TABLA RESUMEN ------------------
    summary = pd.DataFrame(rows).sort_values(by=["f1_macro", "accuracy"], ascending=False).reset_index(drop=True)
    if show_table:
        print("\n=== Resumen de modelos ===")
        display(summary)  # en notebook; si no, usa print(summary)

    # ------------------ PLOTS COMPARATIVOS ------------------
    # ROC
    if plot_roc and any(curves["roc"].values()):
        plt.figure(figsize=(7,6))
        for name, d in curves["roc"].items():
            if not is_multiclass:
                plt.plot(d["fpr"], d["tpr"], lw=2, label=f"{name} (AUC={d['auc']:.3f})")
            else:
                plt.plot(d["fpr_micro"], d["tpr_micro"], lw=2, label=f"{name} (AUC micro={d['auc_micro']:.3f}, macro={d['auc_macro']:.3f})")
        plt.plot([0,1],[0,1], linestyle='--', color='gray')
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title("ROC comparativo" + (" (multiclase: micro)" if is_multiclass else " (binario)"))
        plt.legend(loc="lower right"); plt.grid(True); plt.tight_layout(); plt.show()

    # PR
    if plot_pr and any(curves["pr"].values()):
        plt.figure(figsize=(7,6))
        for name, d in curves["pr"].items():
            if not is_multiclass:
                plt.plot(d["recall"], d["precision"], lw=2, label=f"{name} (AP={d['ap']:.3f})")
            else:
                plt.plot(d["recall_micro"], d["precision_micro"], lw=2, label=f"{name} (AP micro={d['ap_micro']:.3f})")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title("Precision-Recall comparativo" + (" (multiclase: micro)" if is_multiclass else " (binario)"))
        plt.legend(loc="lower left"); plt.grid(True); plt.tight_layout(); plt.show()

    return {
        "summary": summary,
        "reports": reports,
        "curves": curves,
        "is_multiclass": is_multiclass,
        "class_names": class_names
    }



def save_model(model, path: str, model_type: str = "auto", add_timestamp: bool = True):
    """
    Saves a model (PyTorch, XGBoost, or sklearn) and automatically adds a timestamp to the filename.

    Parameters
    ----------
    model : object
        The model object to save (PyTorch nn.Module, XGBoost, sklearn, etc.)
    path : str
        Base path for saving (e.g. 'models/mlp_best.pt')
    model_type : str
        'torch', 'xgb', 'sklearn', or 'auto' (auto-detects)
    add_timestamp : bool
        If True, appends a timestamp (e.g. '_2025-10-07_15-42-31') before the extension.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # --- add timestamp ---
    if add_timestamp:
        base, ext = os.path.splitext(path)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = f"{base}_{timestamp}{ext}"

    # --- detect type automatically ---
    if model_type == "auto":
        if isinstance(model, torch.nn.Module):
            model_type = "torch"
        elif isinstance(model, XGBModel):
            model_type = "xgb"
        else:
            model_type = "sklearn"

    # --- save model by type ---
    if model_type == "torch":
        torch.save(model.state_dict(), path)
        print(f"PyTorch model saved at: {path}")

    elif model_type == "xgb":
        if path.endswith(".pkl"):
            joblib.dump(model, path)
        else:
            model.save_model(path)
        print(f"XGBoost model saved at: {path}")

    elif model_type == "sklearn":
        joblib.dump(model, path)
        print(f"Scikit-learn model saved at: {path}")

    else:
        raise ValueError(f"Unrecognized model_type: {model_type}")

    return path



def load_model(path: str, model_class=None, model_type: str = "auto", **kwargs):
    """
    Carga un modelo guardado según su tipo.
    Para PyTorch, debes pasar model_class (la arquitectura a instanciar).
    """
    if model_type == "auto":
        if path.endswith(".pt") or path.endswith(".pth"):
            model_type = "torch"
        elif path.endswith(".json"):
            model_type = "xgb"
        elif path.endswith(".pkl"):
            model_type = "sklearn"
        else:
            raise ValueError("No se pudo inferir el tipo de modelo por extensión.")

    if model_type == "torch":
        if model_class is None:
            raise ValueError("Para PyTorch, debes pasar model_class (la arquitectura).")
        model = model_class(**kwargs)
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        print(f"Modelo PyTorch cargado desde: {path}")
        return model

    elif model_type == "xgb":
        if path.endswith(".pkl"):
            model = joblib.load(path)
        else:
            from xgboost import XGBClassifier
            model = XGBClassifier()
            model.load_model(path)
        print(f"Modelo XGBoost cargado desde: {path}")
        return model

    elif model_type == "sklearn":
        model = joblib.load(path)
        print(f"Modelo sklearn cargado desde: {path}")
        return model

    else:
        raise ValueError(f"Tipo de modelo no reconocido: {model_type}")