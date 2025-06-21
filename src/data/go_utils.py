from pathlib import Path
from functools import lru_cache
import re
import numpy as np
import networkx as nx
import obonet
from sklearn.metrics import average_precision_score as aupr


GO_SPLITS = {"mf": "### GO-terms (molecular_function)",
             "bp": "### GO-terms (biological_process)",
             "cc": "### GO-terms (cellular_component)"}


# --- GO ontology graph (loaded once) ---------------------
GO_GRAPH = obonet.read_obo(open("data/go-basic.obo", "r"))

@lru_cache(maxsize=None)
def load_go_dict(tsv_path: Path, ontology: str):
    """
    Parse `nrPDB-GO_2019.06.18_annot.tsv` and build
    {GO_ID -> class_index} for a single ontology.
    """
    anchor = GO_SPLITS[ontology]
    go2idx, idx = {}, 0

    with tsv_path.open() as fh:
        for line in fh:
            if line.startswith(anchor):          # found the right block header
                term_line = next(fh).strip()     # the actual IDs are on next line
                break
        else:
            raise RuntimeError(f"{anchor} not found in {tsv_path}")

    # the line looks like: GO:0006288 GO:0009135 ...
    for go_id in re.findall(r"GO:\d{7}", term_line):
        go2idx[go_id] = idx
        idx += 1

    return go2idx             # e.g. len(go2idx)==1943 for BP 

#Evaluation metrics standard and taken from HEAL originally dervied universaly from DEEPFRI

# --- bootstrap helper (kept for completeness) ------------
def bootstrap(Y_true, Y_pred):
    n = Y_true.shape[0]
    idx = np.random.choice(n, n)
    return Y_true[idx], Y_pred[idx]

# --- propagation (identical to HEAL) ---------------------
def propagate_go_preds(Y_hat, goterms):
    go2id = {go: ii for ii, go in enumerate(goterms)}
    for goterm in goterms:
        if goterm in GO_GRAPH:
            parents = set(goterms).intersection(nx.descendants(GO_GRAPH,
                                                               goterm))
            for parent in parents:
                Y_hat[:, go2id[parent]] = np.maximum(Y_hat[:, go2id[goterm]],
                                                     Y_hat[:, go2id[parent]])

    return Y_hat


def propagate_ec_preds(Y_hat, goterms):
    go2id = {go: ii for ii, go in enumerate(goterms)}
    for goterm in goterms:
        if goterm.find('-') == -1:
            parent = goterm.split('.')
            parent[-1] = '-'
            parent = ".".join(parent)
            if parent in go2id:
                Y_hat[:, go2id[parent]] = np.maximum(Y_hat[:, go2id[goterm]],
                                                     Y_hat[:, go2id[parent]])

    return Y_hat

# --- CAFA-style protein-centric F-max --------------------

def _cafa_go_aupr(labels, preds, goterms, ont):
        # propagate goterms (take into account goterm specificity)

        # number of test proteins
        n = labels.shape[0]

        goterms = np.asarray(goterms)
        ont2root = {'bp': 'GO:0008150', 'mf': 'GO:0003674', 'cc': 'GO:0005575'}

        prot2goterms = {}
        for i in range(0, n):
            all_gos = set()
            for goterm in goterms[np.where(labels[i] == 1)[0]]:
                all_gos = all_gos.union(nx.descendants(GO_GRAPH, goterm))
                all_gos.add(goterm)
            all_gos.discard(ont2root[ont])
            prot2goterms[i] = all_gos

        # CAFA-like F-max predictions
        F_list = []
        AvgPr_list = []
        AvgRc_list = []
        thresh_list = []

        for t in range(1, 100):
            threshold = t/100.0
            predictions = (preds > threshold).astype(np.int)

            m = 0
            precision = 0.0
            recall = 0.0
            for i in range(0, n):
                pred_gos = set()
                for goterm in goterms[np.where(predictions[i] == 1)[0]]:
                    pred_gos = pred_gos.union(nx.descendants(GO_GRAPH,
                                                             goterm))
                    pred_gos.add(goterm)
                pred_gos.discard(ont2root[ont])

                num_pred = len(pred_gos)
                num_true = len(prot2goterms[i])
                num_overlap = len(prot2goterms[i].intersection(pred_gos))
                if num_pred > 0 and num_true > 0:
                    m += 1
                    precision += float(num_overlap)/num_pred
                    recall += float(num_overlap)/num_true

            if m > 0:
                AvgPr = precision/m
                AvgRc = recall/n

                if AvgPr + AvgRc > 0:
                    F_score = 2*(AvgPr*AvgRc)/(AvgPr + AvgRc)
                    # record in list
                    F_list.append(F_score)
                    AvgPr_list.append(AvgPr)
                    AvgRc_list.append(AvgRc)
                    thresh_list.append(threshold)

        F_list = np.asarray(F_list)
        AvgPr_list = np.asarray(AvgPr_list)
        AvgRc_list = np.asarray(AvgRc_list)
        thresh_list = np.asarray(thresh_list)

        return AvgRc_list, AvgPr_list, F_list, thresh_list


def _cafa_ec_aupr(labels, preds, goterms):
    # propagate goterms (take into account goterm specificity)

    # number of test proteins
    n = labels.shape[0]

    goterms = np.asarray(goterms)

    prot2goterms = {}
    for i in range(0, n):
        prot2goterms[i] = set(goterms[np.where(labels[i] == 1)[0]])

    # CAFA-like F-max predictions
    F_list = []
    AvgPr_list = []
    AvgRc_list = []
    thresh_list = []

    for t in range(1, 100):
        threshold = t/100.0
        predictions = (preds > threshold).astype(np.int)

        m = 0
        precision = 0.0
        recall = 0.0
        for i in range(0, n):
            pred_gos = set(goterms[np.where(predictions[i] == 1)[0]])
            num_pred = len(pred_gos)
            num_true = len(prot2goterms[i])
            num_overlap = len(prot2goterms[i].intersection(pred_gos))
            if num_pred > 0:
                m += 1
                precision += float(num_overlap)/num_pred
            if num_true > 0:
                recall += float(num_overlap)/num_true

        if m > 0:
            AvgPr = precision/m
            AvgRc = recall/n

            if AvgPr + AvgRc > 0:
                F_score = 2*(AvgPr*AvgRc)/(AvgPr + AvgRc)
                # record in list
                F_list.append(F_score)
                AvgPr_list.append(AvgPr)
                AvgRc_list.append(AvgRc)
                thresh_list.append(threshold)

    F_list = np.asarray(F_list)
    AvgPr_list = np.asarray(AvgPr_list)
    AvgRc_list = np.asarray(AvgRc_list)
    thresh_list = np.asarray(thresh_list)

    return AvgRc_list, AvgPr_list, F_list, thresh_list

#Calculate f-max and threshold for CAFA
def cafa_fmax(labels, preds, goterms, ont):
    if ont == "ec":
        Rc, Pr, F, th = _cafa_ec_aupr(labels, preds, goterms)
    else:
        Rc, Pr, F, th = _cafa_go_aupr(labels, preds, goterms, ont)
    idx = np.argmax(F) 
    return float(F[idx]), float(th[idx]) #returns only F-score and threshold

# --- Information–theoretic S-min -------------------------
def normalizedSemanticDistance(Ytrue, Ypred, termIC, avg=False):
    ru = normalizedRemainingUncertainty(Ytrue, Ypred, termIC, False)
    mi = normalizedMisInformation(Ytrue, Ypred, termIC, False)
    sd = np.sqrt(ru ** 2 + mi ** 2)
    if avg:
        ru = np.mean(ru)
        mi = np.mean(mi)
        sd = np.sqrt(ru ** 2 + mi ** 2)
    return sd

def normalizedRemainingUncertainty(Ytrue, Ypred, termIC, avg=False):
    num = np.logical_and(Ytrue == 1, Ypred == 0).astype(float).dot(termIC)
    denom = np.logical_or(Ytrue == 1, Ypred == 1).astype(float).dot(termIC)
    nru = num / denom
    return np.mean(nru) if avg else nru

def normalizedMisInformation(Ytrue, Ypred, termIC, avg=False):
    num = np.logical_and(Ytrue == 0, Ypred == 1).astype(float).dot(termIC)
    denom = np.logical_or(Ytrue == 1, Ypred == 1).astype(float).dot(termIC)
    nmi = num / denom
    return np.mean(nmi) if avg else nmi

def smin(labels, preds, termIC):
    ss = []
    for t in np.linspace(0.0, 1.0, 100):
        ss.append(
            normalizedSemanticDistance(
                labels, (preds >= t).astype(int), termIC, avg=True
            )
        )
    return float(np.min(ss))

# --- Function-centric AUPR (micro & macro) ---------------
def function_centric_aupr(labels, preds):
    keep_cols = labels.sum(axis=0) > 0
    return ( 
        aupr(labels[:, keep_cols], preds[:, keep_cols], average="macro"),
        aupr(labels[:, keep_cols], preds[:, keep_cols], average="micro"),
    ) #macro, micro output 