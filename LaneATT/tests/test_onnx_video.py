"""Run real ONNX Runtime on small deterministic graphs and real video IO."""
import importlib.util
import sys
import unittest
from unittest.mock import patch

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import test_shared_video as shared
ROOT = shared.ROOT
from lib.onnx_video import parse_args, run
from lib.video import VideoInference


class OnnxVideoTests(unittest.TestCase):
    setUp = shared.SharedVideoTests.setUp
    decode = shared.SharedVideoTests.decode

    def model(self, label, raw=False):
        shape = [1, 3, 360, 640] if label == 'laneatt' else [1, 3, 640, 640]
        if label == 'laneatt':
            data = np.zeros((1, 2, 77), np.float32)
            data[0, :, :2] = [-10, 10]
            data[0, :, 4] = 72
            data[0, 0, 5:] = 200
            data[0, 1, 5:] = 440
        elif raw:
            data = np.zeros((1, 8, 8400), np.float32)
        else:
            data = np.array([[[80, 160, 400, 480, .9, 1], [0, 0, 1, 1, .1, 0]]], np.float32)
        graph = helper.make_graph(
            [helper.make_node('Constant', [], ['output'], value=numpy_helper.from_array(data))],
            label, [helper.make_tensor_value_info('image', TensorProto.FLOAT, shape)],
            [helper.make_tensor_value_info('output', TensorProto.FLOAT, list(data.shape))])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
        model.ir_version = 8
        path = self.folder / f'{label}.onnx'
        onnx.save(model, path)
        return path

    def test_real_onnx_render_both_entrypoints(self):
        for label, filename in [('laneatt','onnx_video_LaneaTT.py'), ('yolo','onnx_video_Yolo.py')]:
            with self.subTest(label=label):
                model = self.model(label)
                output = self.folder / label / 'out.mp4'
                argv = [filename, '--provider', 'cpu', '--onnx', str(model), '--video', str(self.video),
                        '--start-frame', '3', '--frames', '5', '--warmup', '1', '--render', str(output),
                        '--json', str(output.parent)]
                spec = importlib.util.spec_from_file_location(label+'_entry', ROOT.parent / filename)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                calls = []
                original = VideoInference.get_frame
                def get_frame(pipe, frame, **kwargs):
                    calls.append(kwargs)
                    return original(pipe, frame, **kwargs)
                with patch.object(sys, 'argv', argv), patch.object(VideoInference, 'get_frame', get_frame):
                    result = module.main()
                self.assertEqual(result['frames'], 5)
                self.assertEqual(len(calls), 5)
                frames, fps = self.decode(output)
                self.assertEqual(len(frames), 5)
                self.assertEqual(frames[0].shape, (120, 320, 3))
                self.assertEqual(fps, 20)
                self.assertAlmostEqual(float(frames[0][:, :160].mean()), 30, delta=5)
                if label == 'laneatt':
                    self.assertEqual(len(calls[0]['evaluation']), 2)
                    self.assertGreater(int(frames[0][60, 210, 0]), 180)  # blue left
                    self.assertGreater(int(frames[0][60, 240, 2]), 180)  # red center
                else:
                    boxes = calls[0]['yolo_results']
                    self.assertEqual(len(boxes), 1)
                    self.assertEqual(boxes[0][:2], ((20, 20), (100, 100)))
                    self.assertGreater(int(frames[0][60, 180, 0]), 170)

    def test_onnx_no_render_eof_and_raw_yolo_rejection(self):
        model = self.model('laneatt')
        args = parse_args('laneatt', ['--provider', 'cpu', '--onnx', str(model), '--video', str(self.video),
                                     '--start-frame', '10', '--frames', '2000', '--warmup', '0',
                                     '--no-render', '--json', str(self.folder/'stats')])
        result = run('laneatt', args)
        self.assertEqual(result['frames'], 2)
        self.assertIsNone(result['render'])
        self.assertEqual(result['render_ms'], 0)
        raw_model = self.model('yolo', raw=True)
        args = parse_args('yolo', ['--onnx', str(raw_model), '--provider', 'cpu'])
        with self.assertRaisesRegex(ValueError, 'embedded NMS'):
            run('yolo', args)


if __name__ == '__main__':
    unittest.main()
