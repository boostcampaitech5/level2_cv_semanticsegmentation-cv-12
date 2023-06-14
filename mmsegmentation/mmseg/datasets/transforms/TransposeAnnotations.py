from mmseg.registry import TRANSFORMS
from .transforms import BaseTransform
import numpy as np

@TRANSFORMS.register_module()
class TransposeAnnotations(BaseTransform):
    def transform(self, result):
        result["gt_seg_map"] = np.transpose(result["gt_seg_map"], (2, 0, 1))
        
        return result