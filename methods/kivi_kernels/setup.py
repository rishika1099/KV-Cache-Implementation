from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="kivi_gemv",
    ext_modules=[
        CUDAExtension(
            "kivi_gemv",
            sources=[
                "csrc/pybind.cpp",
                "csrc/gemv_cuda.cu",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
