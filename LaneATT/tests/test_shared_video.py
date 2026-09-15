"""CPU checks for the shared video pipeline; TensorRT execution is simulated."""
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
from lib.video import VideoInference
from lib.ego_lanes import get_ego_lanes2


class SharedVideoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.video = self.folder / 'input.avi'
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*'MJPG'), 20, (160, 120))
        self.assertTrue(writer.isOpened())
        for i in range(12):
            writer.write(np.full((120, 160, 3), i * 10, np.uint8))
        writer.release()

    def decode(self, path):
        cap = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return frames, fps

    def pipeline(self, limit=5):
        return VideoInference(video_path=str(self.video), frame_limit=limit,
                              output_folder=self.folder / 'runs', initialize_models=False)

    def test_shared_selection_and_rendering(self):
        y = np.linspace(10, 110, 25)
        points = [np.column_stack((np.full_like(y, x), y)) for x in (10, 50, 110, 150)]
        left, right, mid, _ = get_ego_lanes2(160, points)
        self.assertTrue(np.all(left[:, 0] == 50))
        self.assertTrue(np.all(right[:, 0] == 110))
        self.assertTrue(np.all(mid[:, 0] == 80))
        lanes = [types.SimpleNamespace(points=p / [160, 120]) for p in points]
        pipe = self.pipeline()
        # Same get_frame entry point used by PyTorch and TensorRT.
        frame = pipe.get_frame(np.zeros((120, 160, 3), np.uint8), evaluation=lanes, yolo_results=[])
        self.assertEqual(frame[60, 50].tolist(), [255, 0, 0])
        self.assertEqual(frame[60, 110].tolist(), [0, 0, 255])
        self.assertEqual(frame[60, 80].tolist(), [0, 0, 255])
        self.assertEqual(get_ego_lanes2(160, points[:1]), (None, None, None, None))

    def test_unknown_count_start_limit_and_saved_video(self):
        capture = cv2.VideoCapture
        class UnknownCount:
            def __init__(self, path): self.cap = capture(path)
            def get(self, prop):
                return 0 if prop == cv2.CAP_PROP_FRAME_COUNT else self.cap.get(prop)
            def __getattr__(self, name): return getattr(self.cap, name)
        seen, warmed = [], []
        def process(frame):
            seen.append(round(float(frame.mean())))
            return frame
        output = self.folder / 'slice.mp4'
        with patch('lib.video.cv2.VideoCapture', UnknownCount):
            stats = self.pipeline().video_eval(start_frame=3, output_path=output,
                      frame_processor=process, warmup=lambda frame: warmed.append(round(float(frame.mean()))))
        frames, fps = self.decode(output)
        self.assertEqual(seen, [30, 40, 50, 60, 70])
        self.assertEqual(warmed, [30])
        self.assertEqual(stats['frames'], 5)
        self.assertEqual(len(frames), 5)
        self.assertEqual(frames[0].shape, (120, 320, 3))
        self.assertEqual(fps, 20)

    def test_default_backend_path_and_eof(self):
        pipe = self.pipeline(limit=2000)
        class LaneBackend:
            def reset_video_state(self): pass
            def frame_eval(self, frame): return []
        pipe.laneatt = LaneBackend()
        stats = pipe.video_eval(start_frame=10)
        self.assertEqual(stats['frames'], 2)
        self.assertEqual(len(self.decode(stats['output'])[0]), 2)
        self.assertTrue(str(stats['output']).endswith('input/run1/output.mp4'))
        with self.assertRaises(RuntimeError):
            pipe.video_eval(start_frame=12, render=False)

    def test_writer_finalized_on_processing_error(self):
        output = self.folder / 'error.mp4'
        calls = []
        def process(frame):
            calls.append(1)
            if len(calls) == 3: raise RuntimeError('simulated failure')
            return frame
        with self.assertRaisesRegex(RuntimeError, 'simulated failure'):
            self.pipeline().video_eval(output_path=output, frame_processor=process)
        self.assertEqual(len(self.decode(output)[0]), 2)

    def test_benchmark_calls_shared_loop_without_torch(self):
        import importlib.abc
        class NoTorch(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('torch', 'torchvision', 'ultralytics'):
                    raise AssertionError(f'Unexpected dependency: {fullname}')
        # Import video.py again under the dependency guard.
        guard = NoTorch()
        sys.meta_path.insert(0, guard)
        self.addCleanup(sys.meta_path.remove, guard)
        import importlib
        importlib.reload(sys.modules['lib.video'])
        engine = self.folder / 'fake.engine'
        engine.touch()
        instances = []
        class FakeCuda:
            def mem_info(self): return 1, 2
            def compute_capability(self): return (8, 7)
        class FakeEngine:
            def __init__(self, *args, **kwargs):
                self.outputs = {'proposals': None}
                self.calls = 0
                self.closed = False
                instances.append(self)
            def describe(self): return []
            def run(self, arr): self.calls += 1
            def fetch(self): return {'proposals': np.zeros((1, 1000, 77), np.float32)}
            def close(self): self.closed = True
        fake_runner = types.ModuleType('jetson_tools.trt_runner')
        fake_runner.CudaRT, fake_runner.TrtEngine = FakeCuda, FakeEngine
        with patch.dict(sys.modules, {'tensorrt': types.SimpleNamespace(__version__='test'),
                                      'jetson_tools.trt_runner': fake_runner}):
            spec = importlib.util.spec_from_file_location('benchmark_test', ROOT.parent / 'trt_video_benchmark.py')
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            for render in (True, False):
                out = self.folder / ('render' if render else 'no_render')
                argv = ['benchmark', '--models', 'laneatt', '--laneatt-engine', str(engine),
                        '--video', str(self.video), '--start-frame', '2', '--frames', '4', '--warmup', '2',
                        '--render', str(out / 'out.mp4'), '--json', str(out)]
                if not render: argv.append('--no-render')
                with patch.object(sys, 'argv', argv): mod.main()
                result = json.loads((out / 'Benchmark_1.json').read_text())
                self.assertEqual(result['frames'], 4)
                self.assertEqual(instances[-1].calls, 6)
                self.assertTrue(instances[-1].closed)
                if render: self.assertEqual(len(self.decode(out / 'out.mp4')[0]), 4)
                else: self.assertFalse((out / 'out.mp4').exists())


if __name__ == '__main__':
    unittest.main()
