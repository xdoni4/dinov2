# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from .adapters import DatasetWithEnumeratedTargets
from .loaders import (
    make_data_loader,
    make_dataset,
    SamplerType,
    InfinitePrefetchedDataloader
)
from .collate import (
    collate_data_and_cast,
    collate_data_for_test,
    collate_data_for_test_any
)
from .augmentations import (
    DataAugmentationDINO,
    DataAugmentationDINO3D,
    DataAugmentation3DForClassification,
    DataAugmentation3DForClassificationVal
)
from .masking import MaskingGenerator, MaskingGenerator3D

from .dataset import (
    CTRATETrainPreprocessed,
    CTRATEValPreprocessed,
    OpenNeuroPreprocessed,
    BraTSClassificationTrain,
    BraTSClassificationVal,
    AMOSCTUnlabeledTrainPreprocessed,
    AbdomenAtlasPreprocessed
)

data_source_name_to_data_class = {
    "ct_rate_train" : CTRATETrainPreprocessed,
    "ct_rate_val" : CTRATEValPreprocessed,
    "openneuro" : OpenNeuroPreprocessed,
    "brats_cls_train" : BraTSClassificationTrain,
    "brats_cls_val" : BraTSClassificationVal,
    "amos_train" : AMOSCTUnlabeledTrainPreprocessed,
    "abdomen_atlas" : AbdomenAtlasPreprocessed
}

# data_
