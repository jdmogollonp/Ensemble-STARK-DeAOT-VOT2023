import os
import sys
import argparse

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import Tracker


def driver_img_seq(tracker_name, tracker_param, img_seq, output_dir, optional_box=None, debug=None, save_results=False, obj_num = "1", sam_model=None):
    """Run the tracker on your webcam.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        debug: Debug level.
    """
    tracker = Tracker(tracker_name, tracker_param, "video", sam_model=sam_model)
    tracker.run_img_seq(img_seq_path=img_seq, output_dir=output_dir, tracker_name=tracker_name, optional_box=optional_box, debug=debug, obj_num=obj_num, save_results=save_results)


def main():
    parser = argparse.ArgumentParser(description='Run the tracker on your webcam.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of parameter file.')
    parser.add_argument('img_seq', type=str, help='path to an image sequence.')
    parser.add_argument('--output_dir', type=str, help='path to an root image sequence.')
    parser.add_argument('--optional_box', type=float, default=None, nargs="+", help='optional_box with format x y w h.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--save_results', dest='save_results', action='store_true', help='Save bounding boxes')
    parser.add_argument('--obj_num', type=str, default='1', help='specify object number that should be tracked, if only one object in the scene just give 1.')
    parser.add_argument('--sam_model', type=str, default=None, help='Which SAM (Segment Anything Model) model to use.')
    parser.set_defaults(save_results=True)

    args = parser.parse_args()

    driver_img_seq(args.tracker_name, args.tracker_param, args.img_seq, args.output_dir, args.optional_box, args.debug, args.save_results, args.obj_num, args.sam_model)


if __name__ == '__main__':
    main()
