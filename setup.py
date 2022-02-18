import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ttml",
    version="1.0",
    author="Rik Voorhaar",
    description="Tensor train based machine learning estimator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RikVoorhaar/ttml",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "opt-einsum",
        "pandas",
        "scikit-learn",
        "scipy",
        "tqdm",
        "autoray",
        "xgboost",
    ],
    setup_requires=["pytest-runner"],
    tests_require=[
        "pytorch",
        "tensorflow",
        "pytest",
        "pycodestyle",
        "pytest-cov",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="mathematics machine-learning tensors",
)
