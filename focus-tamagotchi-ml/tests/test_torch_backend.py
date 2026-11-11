import importlib.util
import unittest
import numpy as np

HAS_TORCH = importlib.util.find_spec('torch') is not None


@unittest.skipUnless(HAS_TORCH, 'Optional torch dependency not installed')
class TorchNumericsTests(unittest.TestCase):
    def test_peak_decode_and_pixel_center(self):
        import torch
        from vision.backends.torch_backend import decode_heatmaps
        heatmaps = torch.zeros(1,68,64,64)
        heatmaps[:,:,10,20] = 1
        points,scores = decode_heatmaps(heatmaps)
        np.testing.assert_allclose(points[0,0].numpy(),[20.5,10.5])
        self.assertEqual(float(scores.min()),1)

    def test_quarter_pixel_refinement_and_border(self):
        import torch
        from vision.backends.torch_backend import decode_heatmaps
        heatmaps = torch.zeros(1,68,64,64)
        heatmaps[:,:,10,20]=1; heatmaps[:,:,10,21]=.5
        points,_=decode_heatmaps(heatmaps)
        self.assertEqual(float(points[0,0,0]),20.75)
        heatmaps.zero_();heatmaps[:,:,0,0]=1
        points,_=decode_heatmaps(heatmaps)
        np.testing.assert_allclose(points[0,0].numpy(),[.5,.5])

    def test_crop_geometry_preserves_center_and_scale(self):
        from vision.backends.torch_backend import crop_geometry
        x,y,side=crop_geometry([100,100,200,200])
        self.assertAlmostEqual(x+side/2,150)
        self.assertAlmostEqual(y+side/2,138)
        self.assertAlmostEqual(side,200*200/195)

    def test_crop_converts_bgr_bytes_to_rgb_unit_values(self):
        import torch
        from vision.backends.torch_backend import TorchBackend
        backend=TorchBackend.__new__(TorchBackend)
        backend.torch=torch
        backend.dtype=torch.float32
        backend.grid_axis=(torch.arange(256,dtype=torch.float32)+.5)/256
        image=torch.zeros(1,3,400,400)
        image[:,0]=10;image[:,1]=20;image[:,2]=30
        crop,_=backend._crop(image,[150,150,250,250])
        np.testing.assert_allclose(crop[0,:,128,128].numpy(),np.array([30,20,10])/255,rtol=1e-6)

    def test_gpu_unavailable_fails_before_loading_weights(self):
        import torch
        from unittest.mock import patch
        from vision.backends.torch_backend import TorchBackend
        with patch.object(torch.cuda, 'is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError,'CUDA unavailable'):
                TorchBackend('missing.zip')

    def test_cuda_decode_parity_when_available(self):
        import torch
        from vision.backends.torch_backend import decode_heatmaps
        if not torch.cuda.is_available():
            self.skipTest('NVIDIA CUDA device unavailable on this host')
        torch.manual_seed(2)
        maps=torch.rand(1,68,64,64)
        cpu=decode_heatmaps(maps)
        gpu=decode_heatmaps(maps.cuda())
        for left,right in zip(cpu,gpu):
            torch.testing.assert_close(left,right.cpu())
