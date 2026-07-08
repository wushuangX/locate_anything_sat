#!/usr/bin/env python
from PIL import Image

from eaglevl.model.locany.configuration_locateanything import MoonViTConfig
from eaglevl.model.locany.configuration_qwen2 import Qwen2Config
from eaglevl.model.locany.modeling_locateanything import (
    build_mlp_projector as build_train_mlp_projector,
    get_visual_projector_input_size as get_train_visual_projector_input_size,
)
from eaglevl.train.merge_kernel_utils import MERGE_KERNEL_VALUE_ERROR, parse_merge_kernel_size
from eaglevl.utils.locany.image_processing_locateanything import LocateAnythingImageProcessor
from eaglevl.utils.locany.modeling_locateanything import (
    build_mlp_projector as build_utils_mlp_projector,
    get_visual_projector_input_size as get_utils_visual_projector_input_size,
)


def assert_projector(module_name, get_input_size, build_projector, merge_kernel, expected_input):
    vision_config = MoonViTConfig(merge_kernel_size=merge_kernel)
    text_config = Qwen2Config(hidden_size=2048)
    assert get_input_size(vision_config) == expected_input, module_name
    projector = build_projector(vision_config, text_config)
    assert projector[0].normalized_shape == (expected_input,), module_name
    assert projector[1].in_features == expected_input, module_name


def context_tokens(image_processor, image):
    batch = image_processor.preprocess(image, return_tensors=None)
    h, w = [int(x) for x in batch["image_grid_hws"][0]]
    merge_h, merge_w = [int(x) for x in image_processor.merge_kernel_size]
    return h * w // (merge_h * merge_w)


def assert_parse_helper():
    assert parse_merge_kernel_size("1,1") == [1, 1]
    assert parse_merge_kernel_size("") is None
    try:
        parse_merge_kernel_size("bad")
    except ValueError as exc:
        assert str(exc) == MERGE_KERNEL_VALUE_ERROR
    else:
        raise AssertionError("parse_merge_kernel_size('bad') did not raise")


def main():
    for module_name, get_input_size, build_projector in [
        ("train", get_train_visual_projector_input_size, build_train_mlp_projector),
        ("utils", get_utils_visual_projector_input_size, build_utils_mlp_projector),
    ]:
        assert_projector(module_name, get_input_size, build_projector, [1, 1], 1152)
        assert_projector(module_name, get_input_size, build_projector, [2, 2], 4608)

    image_504 = Image.new("RGB", (504, 504), color=(0, 0, 0))
    tokens_1x1 = context_tokens(LocateAnythingImageProcessor(merge_kernel_size=[1, 1]), image_504)
    tokens_2x2 = context_tokens(LocateAnythingImageProcessor(merge_kernel_size=[2, 2]), image_504)
    assert tokens_1x1 == 4 * tokens_2x2, (tokens_1x1, tokens_2x2)

    image_512 = Image.new("RGB", (512, 512), color=(0, 0, 0))
    processor_1x1 = LocateAnythingImageProcessor(merge_kernel_size=[1, 1])
    batch_512 = processor_1x1.preprocess(image_512, return_tensors=None)
    h, w = [int(x) for x in batch_512["image_grid_hws"][0]]
    assert context_tokens(processor_1x1, image_512) == h * w

    assert_parse_helper()
    print("rs merge kernel smoke passed")


if __name__ == "__main__":
    main()
