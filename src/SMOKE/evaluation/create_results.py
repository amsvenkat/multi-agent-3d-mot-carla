import numpy as np
import os


def get_ap(prec, ap_type):
    prec = np.asarray(prec)
    sums = 0
    if ap_type == 11:
        for i in range(0, prec.shape[-1], 4):
            sums = sums + prec[..., i]
        ap = sums / 11 * 100
    else:
        for i in range(1, prec.shape[-1]):
            sums = sums + prec[..., i]
        ap = sums / 40 * 100
    return ap


def get_aps(results_dir):
    def print_file(s, f):
        f.write('{}\n'.format(s))
        print(s)

    labels = ['car', 'pedestrian', 'cyclist']
    eval_types = ['detection', 'detection_ground', 'detection_3d', 'orientation']
    eval_types_short = {'detection': '2d', 'detection_3d': '3d', 'orientation': 'aos', 'detection_ground': 'bev'}
    difficulties = ['easy', 'moderate', 'hard']

    res_path = os.path.join(results_dir, 'parsed_res.txt')
    f = open(res_path, 'w')
    for label in labels:
        for ap_type in [11, 40]:
            print_file('\n{}_R{}'.format(label, ap_type), f)
            for eval_type in eval_types:
                res_file = os.path.join(results_dir, 'stats_{}_{}.txt'.format(label, eval_type))
                with open(res_file, 'r') as fl:
                    lines = fl.readlines()
                diff_res = [eval_types_short[eval_type]]
                for i, difficulty in enumerate(difficulties):
                    prec = [float(tmp) for tmp in lines[i].strip().split(' ')]
                    ap_res = get_ap(prec, ap_type)
                    diff_res.append('{:.2f}'.format(ap_res))
                print_file(', '.join(diff_res), f)
    f.close()
    print('Saved parsed results to {}'.format(res_path))


if __name__ == '__main__':
    results_dir = '/export/amsvenkat/project/git_final/ma_perception/SMOKE/tools/logs/inference/kitti_test'
    get_aps(results_dir)