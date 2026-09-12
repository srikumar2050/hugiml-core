from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

root = Path(__file__).resolve().parents[2]
setup(
    name="hugiml-index-validation",
    version="0.0.0",
    packages=[],
    py_modules=[],
    ext_modules=[
        Pybind11Extension(
            "_hugiml_index_validation",
            [str(root / "tests/native/csr_validation.cpp")],
            include_dirs=[str(root / "native")],
            cxx_std=17,
        )
    ],
    cmdclass={"build_ext": build_ext},
)
