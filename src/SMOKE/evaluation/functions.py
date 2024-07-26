from typing import List, Dict, Tuple, Any

import matplotlib.pyplot as plt

import numpy as np
from evaluation.metrics import DetectionMetricData


from evaluation.constants import DETECTION_COLORS, DETECTION_NAMES, PRETTY_DETECTION_NAMES, TP_METRICS, PRETTY_TP_METRICS, TP_METRICS_UNITS

def center_distance(gt_box, pred_box) -> float:
    """
    L2 distance between the box centers (xy only).
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: L2 distance.
    """
    return np.linalg.norm(np.array(pred_box['location_cam'][:2]) - np.array(gt_box['location_cam'][:2]))


def scale_iou(sample_annotation, sample_result) -> float:
    """
    This method compares predictions to the ground truth in terms of scale.
    It is equivalent to intersection over union (IOU) between the two boxes in 3D,
    if we assume that the boxes are aligned, i.e. translation and rotation are considered identical.
    :param sample_annotation: GT annotation sample.
    :param sample_result: Predicted sample.
    :return: Scale IOU.
    """
    # Validate inputs.
    sa_size = np.array(sample_annotation['dimension'])
    sr_size = np.array(sample_result['dimension'])
    assert all(sa_size > 0), 'Error: sample_annotation sizes must be >0.'
    assert all(sr_size > 0), 'Error: sample_result sizes must be >0.'

    # Compute IOU.
    min_wlh = np.minimum(sa_size, sr_size)
    volume_annotation = np.prod(sa_size)
    volume_result = np.prod(sr_size)
    intersection = np.prod(min_wlh)  # type: float
    union = volume_annotation + volume_result - intersection  # type: float
    iou = intersection / union

    return iou


def angle_diff(x: float, y: float, period: float) -> float:
    """
    Get the smallest angle difference between 2 angles: the angle from y to x.
    :param x: To angle.
    :param y: From angle.
    :param period: Periodicity in radians for assessing angle difference.
    :return: <float>. Signed smallest between-angle difference in range (-pi, pi).
    """

    # calculate angle difference, modulo to [0, 2*pi]
    diff = (x - y + period / 2) % period - period / 2
    if diff > np.pi:
        diff = diff - (2 * np.pi)  # shift (pi, 2*pi] to (-pi, 0]

    return diff


def yaw_diff(gt_box, pred_box, period: float = 2*np.pi) -> float:
    """
    Returns the yaw angle difference between the orientation of two boxes.
    :param gt_box: Ground truth box.
    :param eval_box: Predicted box.
    :param period: Periodicity in radians for assessing angle difference.
    :return: Yaw angle difference in radians in [0, pi].
    """
    yaw_gt = gt_box['rotation']
    yaw_est = pred_box['rotation']

    return abs(angle_diff(yaw_gt, yaw_est, period))


def cummean(x: np.array) -> np.array:
    """
    Computes the cumulative mean up to each position in a NaN sensitive way
    - If all values are NaN return an array of ones.
    - If some values are NaN, accumulate arrays discording those entries.
    """
    if sum(np.isnan(x)) == len(x):
        # Is all numbers in array are NaN's.
        # If all errors are NaN set to error to 1 for all operating points.
        return np.ones(len(x))
    else:
        # Accumulate in a nan-aware manner.
        # Cumulative sum ignoring nans.
        sum_vals = np.nancumsum(x.astype(float))
        # Number of non-nans up to each position.
        count_vals = np.cumsum(~np.isnan(x))
        return np.divide(sum_vals, count_vals, out=np.zeros_like(sum_vals), where=count_vals != 0)


def accumulate(pred_boxes,
               gt_boxes,
               class_name: str,
               dist_th: float,
               preddict,
               gtdict,
               ):  # -> DetectionMetricData:
    """
    Average Precision over predefined different recall thresholds for a single distance threshold.
    The recall/conf thresholds and other raw metrics will be used in secondary metrics.
    :param gt_boxes: Maps every sample_token to a list of its sample_annotations.
    :param pred_boxes: Maps every sample_token to a list of its sample_results.
    :param class_name: Class to compute AP on.
    :param dist_fcn: Distance function used to match detections and ground truths.
    :param dist_th: Distance threshold for a match.
    :param verbose: If true, print debug messages.
    :return: (average_prec, metrics). The average precision value and raw data for a number of metrics.
    """
    # ---------------------------------------------
    # Organize input and initialize accumulators.
    # ---------------------------------------------

    # Count the positives.
    npos = len([1 for i in gt_boxes if i['class_name'] == class_name])
    print("Dist Threshold - {} , Found {} GT of class {} out of {} total across {} samples.".
          format( dist_th, npos, class_name, len(gt_boxes), len(gtdict)))

    # # For missing classes in the GT, return a data structure corresponding to no predictions.
    # if npos == 0:
    #     return DetectionMetricData.no_predictions()

    # Organize the predictions in a single list.
    pred_boxes_list = [
        box for box in pred_boxes if box['class_name'] == class_name]
    pred_confs = [box['score'] for box in pred_boxes_list]

    print("Dist Threshold - {}  Found {} PRED of class {} out of {} total across {} samples.".
          format(dist_th, len(pred_confs), class_name, len(pred_boxes), len(preddict)))

    # Sort by confidence.
    sortind = [i for (v, i) in sorted((v, i)
                                      for (i, v) in enumerate(pred_confs))][::-1]
    #print(sortind)
    # Do the actual matching.
    tp = []  # Accumulator of true positives.
    fp = []  # Accumulator of false positives.
    conf = []  # Accumulator of confidences.

    # match_data holds the extra metrics we calculate for each match.
    match_data = {'trans_err': [],
                  'scale_err': [],
                  'orient_err': [],
                  'conf': []}

    # ---------------------------------------------
    # Match and accumulate match data.
    # ---------------------------------------------

    taken = set()  # Initially no gt bounding box is matched.
    for ind in sortind:
        pred_box = pred_boxes_list[ind]
        #print(type(pred_box))
        #print(pred_box)
        min_dist = np.inf
        match_gt_idx = None

        for gt_idx, gt_box in enumerate(gtdict[pred_box['sample_frame']]):
            #print(gt_idx, gt_box)

            # Find closest match among ground truth boxes
            if gt_box['class_name'] == class_name and not (pred_box['sample_frame'], gt_idx) in taken:
                this_distance = np.linalg.norm(
                    np.array(pred_box['location_cam'][:2]) - np.array(gt_box['location_cam'][:2]))
                if this_distance < min_dist:
                    min_dist = this_distance
                    match_gt_idx = gt_idx

        # If the closest match is close enough according to threshold we have a match!
        is_match = min_dist < dist_th
    
        if is_match:
            taken.add((pred_box['sample_frame'], match_gt_idx))

            #  Update tp, fp and confs.
            tp.append(1)
            fp.append(0)
            conf.append(pred_box['score'])

            # Since it is a match, update match data also.
            gt_box_match = gtdict[pred_box['sample_frame']][match_gt_idx]

            match_data['trans_err'].append(
                center_distance(gt_box_match, pred_box))
            #match_data['vel_err'].append(velocity_l2(gt_box_match, pred_box))
            match_data['scale_err'].append(
                1 - scale_iou(gt_box_match, pred_box))

            # Barrier orientation is only determined up to 180 degree. (For cones orientation is discarded later)
            period = np.pi if class_name == 'barrier' else 2 * np.pi
            match_data['orient_err'].append(
                yaw_diff(gt_box_match, pred_box, period=period))

            #match_data['attr_err'].append(1 - attr_acc(gt_box_match, pred_box))
            match_data['conf'].append(pred_box['score'])

        else:
            # No match. Mark this as a false positive.
            tp.append(0)
            fp.append(1)
            conf.append(pred_box['score'])

    # Check if we have any matches. If not, just return a "no predictions" array.
    if len(match_data['trans_err']) == 0:
        return DetectionMetricData.no_predictions()

    # ---------------------------------------------
    # Calculate and interpolate precision and recall
    # ---------------------------------------------

    # Accumulate.
    tp = np.cumsum(tp).astype(float)
    fp = np.cumsum(fp).astype(float)
    conf = np.array(conf)

    # Calculate precision and recall.
    prec = tp / (fp + tp)
    rec = tp / float(npos)

    # 101 steps, from 0% to 100% recall.
    rec_interp = np.linspace(0, 1, DetectionMetricData.nelem)
    prec = np.interp(rec_interp, rec, prec, right=0)
    conf = np.interp(rec_interp, rec, conf, right=0)
    rec = rec_interp

    # ---------------------------------------------
    # Re-sample the match-data to match, prec, recall and conf.
    # ---------------------------------------------

    for key in match_data.keys():
        if key == "conf":
            # Confidence is used as reference to align with fp and tp. So skip in this step.
            continue

        else:
            # For each match_data, we first calculate the accumulated mean.
            tmp = cummean(np.array(match_data[key]))

            # Then interpolate based on the confidences. (Note reversing since np.interp needs increasing arrays)
            match_data[key] = np.interp(
                conf[::-1], match_data['conf'][::-1], tmp[::-1])[::-1]

    # ---------------------------------------------
    # Done. Instantiate MetricData and return
    # ---------------------------------------------
    return DetectionMetricData(recall=rec,
                               precision=prec,
                               confidence=conf,
                               trans_err=match_data['trans_err'],
                               scale_err=match_data['scale_err'],
                               orient_err=match_data['orient_err'],
                               )


def calc_ap(md: DetectionMetricData, min_recall: float, min_precision: float) -> float:
    """ Calculated average precision. """

    assert 0 <= min_precision < 1
    assert 0 <= min_recall <= 1

    prec = np.copy(md.precision)
    # Clip low recalls. +1 to exclude the min recall bin.
    prec = prec[round(100 * min_recall) + 1:]
    prec -= min_precision  # Clip low precision
    prec[prec < 0] = 0
    return float(np.mean(prec)) / (1.0 - min_precision)


def calc_tp(md: DetectionMetricData, min_recall: float, metric_name: str) -> float:
    """ Calculates true positive errors. """

    # +1 to exclude the error at min recall.
    first_ind = round(100 * min_recall) + 1
    # First instance of confidence = 0 is index of max achieved recall.
    last_ind = md.max_recall_ind
    if last_ind < first_ind:
        # Assign 1 here. If this happens for all classes, the score for that TP metric will be 0.
        return 1.0
    else:
        # +1 to include error at max recall.
        return float(np.mean(getattr(md, metric_name)[first_ind: last_ind + 1]))


Axis = Any


def setup_axis(xlabel: str = None,
               ylabel: str = None,
               xlim: int = None,
               ylim: int = None,
               title: str = None,
               min_precision: float = None,
               min_recall: float = None,
               ax: Axis = None,
               show_spines: str = 'none'):
    """
    Helper method that sets up the axis for a plot.
    :param xlabel: x label text.
    :param ylabel: y label text.
    :param xlim: Upper limit for x axis.
    :param ylim: Upper limit for y axis.
    :param title: Axis title.
    :param min_precision: Visualize minimum precision as horizontal line.
    :param min_recall: Visualize minimum recall as vertical line.
    :param ax: (optional) an existing axis to be modified.
    :param show_spines: Whether to show axes spines, set to 'none' by default.
    :return: The axes object.
    """
    if ax is None:
        ax = plt.subplot()

    ax.get_xaxis().tick_bottom()
    ax.tick_params(labelsize=16)
    ax.get_yaxis().tick_left()

    # Hide the selected axes spines.
    if show_spines in ['bottomleft', 'none']:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if show_spines == 'none':
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
    elif show_spines in ['all']:
        pass
    else:
        raise NotImplementedError

    if title is not None:
        ax.set_title(title, size=24)
    if xlabel is not None:
        ax.set_xlabel(xlabel, size=16)
    if ylabel is not None:
        ax.set_ylabel(ylabel, size=16)
    if xlim is not None:
        ax.set_xlim(0, xlim)
    if ylim is not None:
        ax.set_ylim(0, ylim)
    if min_recall is not None:
        ax.axvline(x=min_recall, linestyle='--', color=(0, 0, 0, 0.3))
    if min_precision is not None:
        ax.axhline(y=min_precision, linestyle='--', color=(0, 0, 0, 0.3))

    return ax


def class_pr_curve(md_list,
                   metrics,
                   detection_name: str,
                   min_precision: float,
                   min_recall: float,
                   savepath: str = None,
                   ax: Axis = None) -> None:
    """
    Plot a precision recall curve for the specified class.
    :param md_list: DetectionMetricDataList instance.
    :param metrics: DetectionMetrics instance.
    :param detection_name: The detection class.
    :param min_precision:
    :param min_recall: Minimum recall value.
    :param savepath: If given, saves the the rendering here instead of displaying.
    :param ax: Axes onto which to render.
    """
    # Prepare axis.
    if ax is None:
        ax = setup_axis(title=PRETTY_DETECTION_NAMES[detection_name], xlabel='Recall', ylabel='Precision', xlim=1,
                        ylim=1, min_precision=min_precision, min_recall=min_recall)

    # Get recall vs precision values of given class for each distance threshold.
    data = md_list.get_class_data(detection_name)

    # Plot the recall vs. precision curve for each distance threshold.
    for md, dist_th in data:
        md: DetectionMetricData
        ap = metrics.get_label_ap(detection_name, dist_th)
        ax.plot(md.recall, md.precision,
                label='Dist. : {}, AP: {:.1f}'.format(dist_th, ap * 100))

    ax.legend(loc='best')
    if savepath is not None:
        plt.savefig(savepath)
        plt.close()


def class_tp_curve(md_list,
                   metrics,
                   detection_name: str,
                   min_recall: float,
                   dist_th_tp: float,
                   savepath: str = None,
                   ax: Axis = None) -> None:
    """
    Plot the true positive curve for the specified class.
    :param md_list: DetectionMetricDataList instance.
    :param metrics: DetectionMetrics instance.
    :param detection_name:
    :param min_recall: Minimum recall value.
    :param dist_th_tp: The distance threshold used to determine matches.
    :param savepath: If given, saves the the rendering here instead of displaying.
    :param ax: Axes onto which to render.
    """
    # Get metric data for given detection class with tp distance threshold.
    md = md_list[(detection_name, dist_th_tp)]
    min_recall_ind = round(100 * min_recall)
    if min_recall_ind <= md.max_recall_ind:
        # For traffic_cone and barrier only a subset of the metrics are plotted.
        rel_metrics = [m for m in TP_METRICS if not np.isnan(
            metrics.get_label_tp(detection_name, m))]
        ylimit = max([max(getattr(md, metric)[min_recall_ind:md.max_recall_ind + 1])
                     for metric in rel_metrics]) * 1.1
    else:
        ylimit = 1.0

    # Prepare axis.
    if ax is None:
        ax = setup_axis(title=PRETTY_DETECTION_NAMES[detection_name], xlabel='Recall', ylabel='Error', xlim=1,
                        min_recall=min_recall)
    ax.set_ylim(0, ylimit)

    # Plot the recall vs. error curve for each tp metric.
    for metric in TP_METRICS:
        tp = metrics.get_label_tp(detection_name, metric)

        # Plot only if we have valid data.
        if tp is not np.nan and min_recall_ind <= md.max_recall_ind:
            recall, error = md.recall[:md.max_recall_ind +
                                      1], getattr(md, metric)[:md.max_recall_ind + 1]
        else:
            recall, error = [], []

        # Change legend based on tp value
        if tp is np.nan:
            label = '{}: n/a'.format(PRETTY_TP_METRICS[metric])
        elif min_recall_ind > md.max_recall_ind:
            label = '{}: nan'.format(PRETTY_TP_METRICS[metric])
        else:
            label = '{}: {:.2f} ({})'.format(
                PRETTY_TP_METRICS[metric], tp, TP_METRICS_UNITS[metric])
        ax.plot(recall, error, label=label)
    ax.axvline(x=md.max_recall, linestyle='-.', color=(0, 0, 0, 0.3))
    ax.legend(loc='best')

    if savepath is not None:
        plt.savefig(savepath)
        plt.close()

def dist_pr_curve(md_list,
                  metrics,
                  dist_th: float,
                  min_precision: float,
                  min_recall: float,
                  savepath: str = None) -> None:
    """
    Plot the PR curves for different distance thresholds.
    :param md_list: DetectionMetricDataList instance.
    :param metrics: DetectionMetrics instance.
    :param dist_th: Distance threshold for matching.
    :param min_precision: Minimum precision value.
    :param min_recall: Minimum recall value.
    :param savepath: If given, saves the the rendering here instead of displaying.
    """
    # Prepare axis.
    fig, (ax, lax) = plt.subplots(ncols=2, gridspec_kw={"width_ratios": [4, 1]},
                                  figsize=(7.5, 5))
    ax = setup_axis(xlabel='Recall', ylabel='Precision',
                    xlim=1, ylim=1, min_precision=min_precision, min_recall=min_recall, ax=ax)

    # Plot the recall vs. precision curve for each detection class.
    data = md_list.get_dist_data(dist_th)
    for md, detection_name in data:
        md = md_list[(detection_name, dist_th)]
        ap = metrics.get_label_ap(detection_name, dist_th)
        ax.plot(md.recall, md.precision, label='{}: {:.1f}%'.format(PRETTY_DETECTION_NAMES[detection_name], ap * 100),
                color=DETECTION_COLORS[detection_name])
    hx, lx = ax.get_legend_handles_labels()
    lax.legend(hx, lx, borderaxespad=0)
    lax.axis("off")
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
        plt.close()


def summary_plot(md_list,
                 metrics,
                 min_precision: float,
                 min_recall: float,
                 dist_th_tp: float,
                 savepath: str = None) -> None:
    """
    Creates a summary plot with PR and TP curves for each class.
    :param md_list: DetectionMetricDataList instance.
    :param metrics: DetectionMetrics instance.
    :param min_precision: Minimum precision value.
    :param min_recall: Minimum recall value.
    :param dist_th_tp: The distance threshold used to determine matches.
    :param savepath: If given, saves the the rendering here instead of displaying.
    """
    n_classes = len(DETECTION_NAMES)
    _, axes = plt.subplots(nrows=n_classes, ncols=2,
                           figsize=(15, 5 * n_classes))
    for ind, detection_name in enumerate(DETECTION_NAMES):
        title1, title2 = ('Recall vs Precision',
                          'Recall vs Error') if ind == 0 else (None, None)

        ax1 = setup_axis(xlim=1, ylim=1, title=title1, min_precision=min_precision,
                         min_recall=min_recall, ax=axes[ind, 0])
        ax1.set_ylabel('{} \n \n Precision'.format(
            PRETTY_DETECTION_NAMES[detection_name]), size=20)

        ax2 = setup_axis(xlim=1, title=title2,
                         min_recall=min_recall, ax=axes[ind, 1])
        if ind == n_classes - 1:
            ax1.set_xlabel('Recall', size=20)
            ax2.set_xlabel('Recall', size=20)

        class_pr_curve(md_list, metrics, detection_name,
                       min_precision, min_recall, ax=ax1)
        class_tp_curve(md_list, metrics, detection_name,
                       min_recall, dist_th_tp=dist_th_tp, ax=ax2)

    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath)
        plt.close()
