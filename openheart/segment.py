"""Segmentation with a trained nnU-Net v2 model, in process.

The model is loaded once and kept on the device; each image goes through nnU-Net's
own preprocessing, sliding-window inference and export maths, but as arrays — no
nnUNet_* environment variables, no temporary imagesTs/labelsTs folders, no
filename correspondence files. nnunetv2 >= 2.1 is driven through its
``nnUNetPredictor``; nnunetv2 2.0 (which has no in-memory API) through the same
internal functions its ``predict_from_raw_data`` uses.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import logging

import numpy as np

from .case import Case
from .image import Image
from .models import resolve_model
from .pipeline import Step

log = logging.getLogger(__name__)


class Segmenter:
    """A loaded nnU-Net model that labels one 3D short-axis stack at a time.

    Output labels are those of the model's ``dataset.json`` (for the OpenHeart model:
    1=LV, 2=myo, 3=RV, i.e. the canonical convention).
    """

    def __init__(self, model, folds=None, checkpoint="checkpoint_final.pth", device="cuda",
                 tile_step_size=0.5, use_mirroring=True):
        import torch

        self.model_dir = resolve_model(model)
        self.device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        self.tile_step_size = tile_step_size
        self.use_mirroring = use_mirroring
        folds = None if folds is None else tuple(int(f) if f != "all" else f for f in folds)
        # nnunetv2 prints warnings about unset nnUNet_* path variables on import; inference
        # in process does not use them, so keep that noise out of the user's console.
        with contextlib.redirect_stdout(io.StringIO()):
            import nnunetv2.paths  # noqa: F401
        try:
            from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor  # nnunetv2 >= 2.1
        except ImportError:
            self._backend = _Nnunet20(self.model_dir, folds, checkpoint, self.device,
                                      tile_step_size, use_mirroring)
        else:
            self._backend = _NnunetPredictorBackend(nnUNetPredictor, self.model_dir, folds, checkpoint,
                                                    self.device, tile_step_size, use_mirroring)

    def predict(self, image: Image) -> Image:
        """Segment ``image``; returns a label Image on the same grid."""
        # nnU-Net's SimpleITK reader sees the voxel grid reversed: (k, j, i) with spacing (sk, sj, si).
        data = np.asarray(image.array, dtype=np.float32).transpose(2, 1, 0)
        spacing = list(image.spacing)[::-1]
        labels = self._backend.predict(data, spacing)
        return image.with_array(np.ascontiguousarray(labels.transpose(2, 1, 0)).astype(np.uint8))


class _NnunetPredictorBackend:
    def __init__(self, predictor_cls, model_dir, folds, checkpoint, device, tile_step_size, use_mirroring):
        wanted = dict(tile_step_size=tile_step_size, use_gaussian=True, use_mirroring=use_mirroring,
                      perform_everything_on_device=device.type == "cuda",
                      perform_everything_on_gpu=device.type == "cuda",  # name in nnunetv2 2.1
                      device=device, verbose=False, verbose_preprocessing=False, allow_tqdm=False)
        accepted = inspect.signature(predictor_cls.__init__).parameters
        with contextlib.redirect_stdout(io.StringIO()):  # nnU-Net prints device/plans notices
            self.predictor = predictor_cls(**{k: v for k, v in wanted.items() if k in accepted})
            self.predictor.initialize_from_trained_model_folder(str(model_dir), use_folds=folds,
                                                                checkpoint_name=checkpoint)

    def predict(self, data, spacing):
        return self.predictor.predict_single_npy_array(data[None], {"spacing": spacing}, None, None, False)


class _Nnunet20:
    """nnunetv2 2.0: DefaultPreprocessor.run_case + predict_from_raw_data + export, on arrays."""

    def __init__(self, model_dir, folds, checkpoint, device, tile_step_size, use_mirroring):
        import torch
        from nnunetv2.inference.predict_from_raw_data import auto_detect_available_folds, load_what_we_need
        from nnunetv2.inference.sliding_window_prediction import compute_gaussian

        if folds is None:
            folds = auto_detect_available_folds(str(model_dir), checkpoint)
        (self.parameters, self.cm, mirror_axes, self.pm, self.dataset_json,
         self.network, _) = load_what_we_need(str(model_dir), folds, checkpoint)
        self.mirror_axes = mirror_axes if use_mirroring else None
        self.device = device
        self.on_gpu = device.type == "cuda"
        self.tile_step_size = tile_step_size
        self.network = self.network.to(device).eval()
        self.preprocessor = self.cm.preprocessor_class(verbose=False)
        self.label_manager = self.pm.get_label_manager(self.dataset_json)
        gaussian = torch.from_numpy(compute_gaussian(self.cm.patch_size)).half()
        self.gaussian = gaussian.to(device) if self.on_gpu else gaussian

    def predict(self, data, spacing):
        import torch
        from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
        from nnunetv2.inference.sliding_window_prediction import predict_sliding_window_return_logits
        from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
        from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape

        pm, cm = self.pm, self.cm
        # --- DefaultPreprocessor.run_case (no reference segmentation)
        data = data[None].transpose([0, *[i + 1 for i in pm.transpose_forward]])
        original_spacing = [spacing[i] for i in pm.transpose_forward]
        shape_before_cropping = data.shape[1:]
        data, seg, bbox = crop_to_nonzero(data, None)
        shape_after_cropping = data.shape[1:]
        target_spacing = list(cm.spacing)
        if len(target_spacing) < len(data.shape[1:]):
            target_spacing = [original_spacing[0]] + target_spacing
        new_shape = compute_new_shape(data.shape[1:], original_spacing, target_spacing)
        data = self.preprocessor._normalize(data, seg, cm, pm.foreground_intensity_properties_per_channel)
        data = cm.resampling_fn_data(data, new_shape, original_spacing, target_spacing)

        # --- sliding-window inference, averaged over folds
        x = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32))
        n_heads = self.label_manager.num_segmentation_heads
        logits = None
        with torch.no_grad():
            for params in self.parameters:
                self.network.load_state_dict(params)
                out = predict_sliding_window_return_logits(
                    self.network, x, n_heads, cm.patch_size, mirror_axes=self.mirror_axes,
                    tile_step_size=self.tile_step_size, use_gaussian=True,
                    precomputed_gaussian=self.gaussian, perform_everything_on_gpu=self.on_gpu,
                    verbose=False, device=self.device)
                logits = out if logits is None else logits + out
        logits = (logits / len(self.parameters)).to("cpu").numpy().astype(np.float32)

        # --- export_prediction_from_softmax, without writing a file
        current_spacing = (list(cm.spacing) if len(cm.spacing) == len(shape_after_cropping)
                           else [spacing[0], *cm.spacing])
        probs = cm.resampling_fn_probabilities(logits, shape_after_cropping, current_spacing, spacing)
        labels = self.label_manager.convert_logits_to_segmentation(probs)
        full = np.zeros(shape_before_cropping, dtype=np.uint8)
        full[bounding_box_to_slice(bbox)] = labels
        return full.transpose(pm.transpose_backward)


class Segment(Step):
    """Segment every phase image of a case with an nnU-Net model.

    ``model`` is a registered model name or the nnU-Net training output folder
    (``.../Dataset003_RBHSHare/nnUNetTrainer__nnUNetPlans__2d``). ``folds=None`` uses
    every fold found (an ensemble); ``folds=[1]`` reproduces a single-fold run.
    """

    name = "segment"
    volatile = ("device",)
    writes = ("segs",)

    def __init__(self, model, folds=None, checkpoint="checkpoint_final.pth", device="cuda"):
        self.model = str(model)
        self.folds = None if folds is None else [f if f == "all" else int(f) for f in folds]
        self.checkpoint = checkpoint
        self.device = device

    def setup(self):
        self._segmenter = Segmenter(self.model, self.folds, self.checkpoint, self.device)

    def process(self, case: Case) -> Case:
        if not case.images:
            raise ValueError(f"case {case.id} has no images to segment")
        for phase, image in case.images.items():
            case.segs[phase] = self._segmenter.predict(image)
        return case
