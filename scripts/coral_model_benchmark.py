#!/usr/bin/env python3
"""Compare SSDLite-MobileDet vs EfficientDet-Lite2 on images from a ROS 2 rosbag.

Runs each model sequentially on the same Coral USB Accelerator (unload first interpreter
before loading the second).

Writes PNG composites: MobileDet draws **solid** boxes with palette shift 0; EfficientDet draws
**dashed** boxes with a larger palette shift so class colors diverge visibly.

Requires: Coral USB + feranick pycoral wheels; ROS 2 Jazzy sourced; bag at path with metadata.

Example:

    source /opt/ros/jazzy/setup.bash
    source ~/ros2_ws/install/setup.bash
    ros2 run deepgis_vision coral_model_benchmark.py -- \\
       ~/earth-rover-bags/your_record --topic /stereo/left/image_raw -n 30
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore

from cv_bridge import CvBridge, CvBridgeError
from rosbag2_py import ConverterOptions
from rosbag2_py import SequentialReader
from rosbag2_py import StorageOptions

from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from ai_vision_node import (  # noqa: E402
    AIServerConfig,
    CLASS_COLOR_PALETTE,
    CoralClient,
    ModelType,
    draw_detection_overlay,
)


def _download_hints() -> str:
    return """Download models:

  mkdir -p ~/Downloads/coral/samples && cd ~/Downloads/coral/samples
  curl -L -O https://github.com/google-coral/test_data/raw/master/ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite
  curl -L -O https://github.com/google-coral/test_data/raw/master/efficientdet_lite2_448_ptq_edgetpu.tflite
  curl -L -O https://github.com/google-coral/test_data/raw/master/coco_labels.txt
"""


def _sniff_storage_id(bag_path: str) -> str:
    meta = Path(bag_path).expanduser() / 'metadata.yaml'
    if meta.is_file() and yaml is not None:
        tree = yaml.safe_load(meta.read_text())
        info = tree.get('rosbag2_bagfile_information', {})
        sid = info.get('storage_id')
        if sid is None and 'storage_identifier' in info:
            si = info['storage_identifier']
            sid = si.get('name') if isinstance(si, dict) else si
        sid = sid or 'sqlite3'
        sid = str(sid).split('+')[0]
        logging.info('metadata.yaml storage_id → %r', sid)
        return sid
    logging.warning('Could not parse metadata.yaml → assuming sqlite3')
    return 'sqlite3'


def read_image_frames(
    bag_path: str,
    topic_name: str,
    max_frames: int,
    storage_id_override: str = '',
) -> List[np.ndarray]:
    bag_uri = str(Path(bag_path).expanduser().resolve())
    sid = storage_id_override or _sniff_storage_id(bag_uri)

    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=bag_uri, storage_id=sid),
        ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr',
        ),
    )
    md = reader.get_metadata()
    names = sorted({t.name for t in md.topics_with_message_count})
    logging.info(
        '%d topics, %d total messages.',
        len(names), md.message_count,
    )

    bridge = CvBridge()
    frames: List[np.ndarray] = []
    if topic_name not in names:
        reader.close()
        raise ValueError(
            f'Topic {topic_name!r} not found. Topics include: '
            + ', '.join(names[:35])
        )

    while reader.has_next() and len(frames) < max_frames:
        tname, data, _stamp = reader.read_next()
        if tname != topic_name:
            continue

        img_msg = deserialize_message(data, Image)
        try:
            img = bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        except CvBridgeError:
            enc = getattr(img_msg, 'encoding', '') or ''
            enc_try = ''
            for e in ('bgr8', 'rgb8', 'bgra8', 'rgba8', 'mono8'):
                if enc == e or enc.startswith(e):
                    enc_try = e
                    break
            if not enc_try:
                logging.warning('Skipping frame; unsupported encoding %r', enc)
                continue
            img = bridge.imgmsg_to_cv2(img_msg, enc_try)

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        frames.append(img)

    reader.close()
    logging.info('Collected %d BGR frames from %s.', len(frames), topic_name)
    return frames


def coral_sweep_frames(
    model_path: str,
    labels_path: str,
    model_tag: str,
    frames: Sequence[np.ndarray],
    score_thr: float,
    logger: logging.Logger,
    warmup: int = 2,
) -> tuple[list[float], list[list[Dict]]]:
    cfg = AIServerConfig(
        name=model_tag,
        url='benchmark://local',
        model_type=ModelType.CUSTOM,
    )
    cli = CoralClient(
        cfg,
        logger,
        model_path=model_path,
        labels_path=labels_path,
        score_threshold=score_thr,
        model_name_tag=model_tag,
    )

    lats: list[float] = []
    batches: list[list[dict]] = []

    if frames:
        n_w = min(max(0, warmup), len(frames))
        for i in range(n_w):
            cli.detect_image(frames[i], confidence=score_thr)

    for img in frames:
        dets = cli.detect_image(img, confidence=score_thr)
        lats.append(cli.last_latency_ms)
        batches.append([asdict(d) for d in dets])

    del cli

    gc.collect()
    time.sleep(0.2)
    return lats, batches


def agg_counts(det_lists: List[List[Dict]]) -> Dict[str, int]:
    c: Dict[str, int] = defaultdict(int)
    for frame_dets in det_lists:
        for d in frame_dets:
            c[d.get('class_name', '?')] += 1
    return dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))


def print_summary(title: str, lats: List[float], cnt: Dict[str, int]) -> None:
    arr = np.array(lats, dtype=np.float64) if lats else np.zeros(1)
    print('\n===== %s =====' % title)
    print(' Frames:             %d' % len(lats))
    if lats:
        print(' latency mean/med:  %.2f / %.2f ms'
              % (float(arr.mean()), float(np.median(arr))))
        print(' latency p95:       %.2f ms' %
              float(np.percentile(arr, 95)))
    print(' Detection events: %d' % sum(cnt.values()))
    for lab, n in list(cnt.items())[:20]:
        print('   %-34s %5d' % (lab, n))
    print()


def compose_dual_visual(
    bgr: np.ndarray,
    mob: List[Dict],
    eff: List[Dict],
) -> np.ndarray:
    shift_b = max(10, len(CLASS_COLOR_PALETTE) // 2)
    out = bgr.copy()
    draw_detection_overlay(
        out,
        mob,
        palette_shift=0,
        dashed_boxes=False,
        header_top=(
            'A MobileDet SOLID palette+0  |  '
            'B EffDetLite2 DASH palette+%d' % shift_b),
        header_sub='compare overlaps — urban/street detectors',
        latency_ms=0.0,
    )
    draw_detection_overlay(
        out,
        eff,
        palette_shift=shift_b,
        dashed_boxes=True,
        latency_ms=0.0,
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('bag_path', help='Rosbag DIRECTORY (folder with metadata.yaml)')
    p.add_argument('--topic', '-t', default='/stereo/left/image_raw')
    p.add_argument('--max-frames', '-n', type=int, default=40)
    p.add_argument('--score', type=float, default=0.4)
    p.add_argument(
        '--output-dir',
        default=str(Path.home() / 'Downloads/coral/benchmark_compare'),
    )
    p.add_argument('--storage-id', default='')
    hp = Path.home()
    dm = hp / 'Downloads/coral/samples/ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite'
    de = hp / 'Downloads/coral/samples/efficientdet_lite2_448_ptq_edgetpu.tflite'
    dl = hp / 'Downloads/coral/samples/coco_labels.txt'
    p.add_argument('--mobiledet-model', default=str(dm))
    p.add_argument('--eff-model', default=str(de))
    p.add_argument('--labels', '-l', default=str(dl))
    p.add_argument('--no-images', action='store_true')
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    log = logging.getLogger('coral_benchmark')

    for pt, lbl in ((args.mobiledet_model, 'MobileDet'),
                    (args.eff_model, 'EffDet Lite2')):
        if not os.path.isfile(pt):
            logging.error('%s missing: %s', lbl, pt)
            print(_download_hints())
            sys.exit(1)

    if not Path(args.labels).is_file():
        logging.error(f'coco labels missing: {args.labels}')
        sys.exit(1)

    try:
        frames = read_image_frames(
            args.bag_path,
            args.topic,
            args.max_frames,
            args.storage_id,
        )
    except Exception as err:  # noqa: BLE001
        logging.exception('Opening bag failed: %s', err)
        sys.exit(2)

    if not frames:
        logging.error('No frames decoded. Check `--topic`.')
        sys.exit(3)

    log.info('')
    log.info('[1/2] MobileDet pass …')
    mlats, mr = coral_sweep_frames(
        args.mobiledet_model,
        args.labels,
        'MobileDet',
        frames,
        args.score,
        log,
    )
    mc = agg_counts(mr)

    log.info('')
    log.info('[2/2] EfficientDet-Lite2 pass …')
    elats, er = coral_sweep_frames(
        args.eff_model,
        args.labels,
        'EffDetL2',
        frames,
        args.score,
        log,
    )
    ec = agg_counts(er)

    print_summary('SSDLite-MobileDet', mlats, mc)
    print_summary('EfficientDet-Lite2 x448', elats, ec)

    mr_arr = np.array(mlats, dtype=np.float64)
    er_arr = np.array(elats, dtype=np.float64)

    delta = mr_arr.mean() - er_arr.mean()
    ratio = mr_arr.mean() / max(0.01, er_arr.mean())
    print(
        '[Δ latency] MobileDet − EffDetL2 mean: %+.2f ms '
        '(MobileDet %.2fx of EffDetL2 FPS budget)\n'
        % (delta, ratio),
    )

    if args.no_images:
        return

    outp = Path(args.output_dir).expanduser()
    outp.mkdir(parents=True, exist_ok=True)
    exported = min(12, len(frames), len(mr), len(er))
    for i in range(exported):
        fp = outp / ('frame_%03d_compare.png' % i)
        png = compose_dual_visual(frames[i], mr[i], er[i])
        cv2.imwrite(str(fp), png)
        log.info('Wrote composite %s', fp)

    print('Composite overlays → %s' % outp.resolve())


if __name__ == '__main__':
    main()
