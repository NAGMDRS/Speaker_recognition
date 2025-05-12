
import os, numpy
from sklearn import metrics
from operator import itemgetter


def init_args(args):
    """
    Initialize paths for saving model and score files.

    Args:
        args (Namespace): Argument object containing a 'save_path' attribute.

    Returns:
        Namespace: Updated argument object with 'score_save_path' and 'model_save_path'.
    """
    args.score_save_path = os.path.join(args.save_path, 'score.txt')
    args.model_save_path = os.path.join(args.save_path, 'model')
    os.makedirs(args.model_save_path, exist_ok=True)
    return args


def tuneThresholdfromScore(scores, labels, target_fa, target_fr=None):
    """
    Tune score thresholds to achieve target false acceptance (FA) and false rejection (FR) rates.

    Args:
        scores (list or np.ndarray): Prediction scores.
        labels (list or np.ndarray): Ground truth binary labels (0 or 1).
        target_fa (list): List of target false acceptance rates.
        target_fr (list, optional): List of target false rejection rates.

    Returns:
        tuple: (tuned thresholds, equal error rate, false positive rates, false negative rates)
    """
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    tunedThreshold = [];
    if target_fr:
        for tfr in target_fr:
            idx = numpy.nanargmin(numpy.absolute((tfr - fnr)))
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
    for tfa in target_fa:
        idx = numpy.nanargmin(numpy.absolute((tfa - fpr)))  # numpy.where(fpr<=tfa)[0][-1]
        tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
    idxE = numpy.nanargmin(numpy.absolute((fnr - fpr)))
    eer = max(fpr[idxE], fnr[idxE]) * 100

    return tunedThreshold, eer, fpr, fnr

def ComputeErrorRates(scores, labels):
    """
    Compute false negative rates (FNR) and false positive rates (FPR) across thresholds.

    Args:
        scores (list or np.ndarray): Prediction scores.
        labels (list): Corresponding binary ground truth labels (0 or 1).

    Returns:
        tuple: (list of FNRs, list of FPRs, list of thresholds)
    """
    sorted_indexes, thresholds = zip(*sorted(
        [(index, threshold) for index, threshold in enumerate(scores)],
        key=itemgetter(1)))
    sorted_labels = []
    labels = [labels[i] for i in sorted_indexes]
    fnrs = []
    fprs = []
    for i in range(0, len(labels)):
        if i == 0:
            fnrs.append(labels[i])
            fprs.append(1 - labels[i])
        else:
            fnrs.append(fnrs[i - 1] + labels[i])
            fprs.append(fprs[i - 1] + 1 - labels[i])
    fnrs_norm = sum(labels)
    fprs_norm = len(labels) - fnrs_norm
    fnrs = [x / float(fnrs_norm) for x in fnrs]

    fprs = [1 - x / float(fprs_norm) for x in fprs]
    return fnrs, fprs, thresholds

def ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
    """
    Compute the minimum Detection Cost Function (DCF).

    Args:
        fnrs (list): False negative rates.
        fprs (list): False positive rates.
        thresholds (list): Thresholds corresponding to FNR and FPR.
        p_target (float): Prior probability of the target class.
        c_miss (float): Cost of a miss (false negative).
        c_fa (float): Cost of a false alarm (false positive).

    Returns:
        tuple: (minimum DCF value, threshold corresponding to minimum DCF)
    """
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold


def accuracy(output, target, topk=(1,)):
    """
    Compute the top-k accuracy for model predictions.

    Args:
        output (Tensor): Model output logits.
        target (Tensor): Ground truth labels.
        topk (tuple): Tuple of top-k values (e.g., (1,), (1, 5)).

    Returns:
        list: List of top-k accuracy values as percentages.
    """
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res