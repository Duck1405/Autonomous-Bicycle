"""CPU checks; TensorRT engine execution is mocked, not GPU validation."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import trt_video_benchmark as benchmark


class BenchmarkTests(unittest.TestCase):
    def test_defaults_and_integer_topk(self):
        args = benchmark.parse_args([])
        self.assertEqual(args.conf_threshold, .5)
        self.assertIsInstance(benchmark.parse_args(['--nms_topk', '4']).nms_topk, int)
        self.assertIsNone(benchmark.parse_args(['--no-render']).render)

    def test_bad_arguments_fail_before_cuda(self):
        cases = [ ['--nms_topk','4.5'], ['--nms_topk','0'], ['--frames','0'],
                  ['--conf_threshold','nan'], ['--keep_threshold','inf'],
                  ['--match_tolerance','-1'], ['--nms_thres','-1'],
                  ['--keep_threshold','.8','--conf_threshold','.5'],
                  ['--models',''], ['--models','unknown'],
                  ['--models','laneatt','--laneatt-on','false'], ['--codec','toolong'] ]
        for argv in cases:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as result:
                    benchmark.parse_args(argv)
                self.assertEqual(result.exception.code, 2)

    def run_benchmark(self, extra=(), fail=False, frames=2):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        video = root/'input.avi'; video.touch()
        engine_path = root/'model.engine'; engine_path.touch()
        runtime = types.ModuleType('jetson_tools.trt_runner')
        cuda = MagicMock(); cuda.mem_info.return_value=(2**30,2**31); cuda.compute_capability.return_value=(8,7)
        runtime.CudaRT=MagicMock(return_value=cuda)
        engine=MagicMock(); engine.describe.return_value=[]; engine.outputs={'output':None}
        engine.fetch.return_value={'output':np.zeros((1,1,77))}
        runtime.TrtEngine=MagicMock(return_value=engine)
        trt=types.ModuleType('tensorrt'); trt.__version__='mock'
        pipeline=MagicMock(); pipeline.conf_threshold=.7; pipeline.keep_threshold=.2
        pipeline.nms_thres=25.; pipeline.nms_topk=3
        frame=np.zeros((8,8,3),np.uint8)
        accepted=[]
        pipeline.get_frame.side_effect=lambda canvas, **kw: accepted.append(kw['evaluation']) or canvas
        def video_eval(**kw):
            if fail: raise RuntimeError('simulated video failure')
            kw['warmup'](frame)
            for _ in range(frames):kw['frame_processor'](frame)
            return dict(frames=frames, elapsed_seconds=1., write_seconds=0.,
                        read_seconds=0., output=str(root/'render.mp4'))
        pipeline.video_eval.side_effect=video_eval
        def decode(*a, **kw):
            return [{'points':np.array([[.2,.5],[.2,1.]]),'conf':.6}]
        argv=['trt_video_benchmark.py','--video',str(video),'--models','laneatt',
              '--laneatt-engine',str(engine_path),'--conf_threshold','.7',
              '--keep_threshold','.2','--match_tolerance','.12','--nms_topk','3',
              '--nms_thres','25','--warmup','0','--json',str(root/'results.json'),*extra]
        with patch.dict(sys.modules,{'tensorrt':trt,'jetson_tools.trt_runner':runtime}), \
             patch.object(sys,'argv',argv), \
             patch.object(benchmark,'VideoInference',return_value=pipeline), \
             patch.object(benchmark,'_pre_laneatt',return_value=(frame,None)), \
             patch.object(benchmark,'laneatt_decode',side_effect=decode) as decoder, \
             patch.object(benchmark,'LaneHysteresis',wraps=benchmark.LaneHysteresis) as hysteresis, \
             contextlib.redirect_stdout(io.StringIO()):
            if fail or not frames:
                with self.assertRaises(RuntimeError):benchmark.main()
            else:benchmark.main()
        engine.close.assert_called_once()
        return root, decoder, hysteresis, accepted, pipeline

    def test_hysteresis_receives_cli_values(self):
        root, decoder, hysteresis, accepted, _=self.run_benchmark()
        hysteresis.assert_called_once_with(conf_threshold=.7,keep_threshold=.2,match_tolerance=.12)
        self.assertEqual(decoder.call_args.kwargs['conf_threshold'],.2)
        self.assertEqual(accepted,[[],[]])  # .6 must NOT acquire with requested .7 threshold
        result=json.loads((root/'results.json').read_text())
        self.assertEqual(result['frames'],2)
        self.assertEqual(result['laneatt_parameters']['nms_topk'],3)

    def test_no_hysteresis_uses_acquisition_threshold(self):
        _,decoder,hysteresis,_,_=self.run_benchmark(['--no-hysteresis'])
        hysteresis.assert_not_called()
        self.assertEqual(decoder.call_args.kwargs['conf_threshold'],.7)

    def test_no_render_skips_decode(self):
        _,decoder,_,_,pipeline=self.run_benchmark(['--no-render'])
        decoder.assert_not_called()
        self.assertFalse(pipeline.video_eval.call_args.kwargs['render'])

    def test_engine_cleanup_on_failure(self):
        self.run_benchmark(fail=True)

    def test_zero_frames_clear_error(self):
        self.run_benchmark(frames=0)


if __name__=='__main__':unittest.main()
