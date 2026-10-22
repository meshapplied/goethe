"""Build mesh_kernels CUDAExtension for a single SM target."""

import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

sm = os.environ.get("MESH_KERNEL_SM", "sm120")
arch = os.environ.get("MESH_KERNEL_ARCH", "12.0")
arch_tag = arch.replace(".", "")

setup(
    name=f"mesh_kernels_{sm}",
    ext_modules=[
        CUDAExtension(
            name=f"libmesh_kernels_{sm}",
            sources=[
                "csrc/ops.cpp",
                "csrc/nvfp4_gemv.cu",
                "csrc/nvfp4_api.cu",
                "csrc/moe_dispatch.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    f"-gencode=arch=compute_{arch_tag},code=sm_{arch_tag}",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True, verbose=False)},
)
