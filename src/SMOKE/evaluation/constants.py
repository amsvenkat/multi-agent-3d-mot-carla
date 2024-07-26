DETECTION_NAMES = ['Car',  'Pedestrian']

PRETTY_DETECTION_NAMES = {'Car': 'Car',
                          'Pedestrian': 'Pedestrian',
}

TP_METRICS = ['trans_err', 'scale_err', 'orient_err']

PRETTY_TP_METRICS = {'trans_err': 'Trans.', 'scale_err': 'Scale', 'orient_err': 'Orient.'}

TP_METRICS_UNITS = {'trans_err': 'm',
                    'scale_err': '1-IOU',
                    'orient_err': 'rad.'
}

DETECTION_COLORS = {'Car': 'C0',
                    'Pedestrian': 'C5',
}
