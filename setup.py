from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "multivariate_gaussian",
        ["multivariate_gaussian.cpp"],
        include_dirs=["/usr/include/eigen3"],  # Ruta de Eigen
    ),
]

setup(
    name="multivariate_gaussian",
    version="1.0",
    author="Osvaldo R",
    description="Cálculo de la distribución Gaussiana Multivariada en C++ con Pybind11",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
