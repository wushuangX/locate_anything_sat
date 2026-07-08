from typing import Optional


MERGE_KERNEL_VALUE_ERROR = "vision_merge_kernel_size must be two positive integers like '1,1'"


def parse_merge_kernel_size(value: Optional[str]) -> Optional[list[int]]:
    if value is None or str(value).strip() == "":
        return None
    parts = [p.strip() for p in str(value).split(",")]
    if len(parts) != 2:
        raise ValueError(MERGE_KERNEL_VALUE_ERROR)
    try:
        kernel = [int(parts[0]), int(parts[1])]
    except ValueError as exc:
        raise ValueError(MERGE_KERNEL_VALUE_ERROR) from exc
    if kernel[0] <= 0 or kernel[1] <= 0:
        raise ValueError(MERGE_KERNEL_VALUE_ERROR)
    return kernel


def merge_kernel_product(kernel: Optional[list[int]], default: tuple[int, int] = (2, 2)) -> int:
    k = kernel if kernel is not None else list(default)
    return int(k[0]) * int(k[1])
