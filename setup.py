from setuptools import setup, find_packages

setup(
    name="atest",
    version="0.1.0",
    description="A package for extracting audio features with data augmentation.",
    author="Shubhodip Pal",
    author_email="shubhodippal01@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy==1.23.5",
        "librosa==0.10.1",
        "pandas==1.5.3",
        "tqdm==4.67.1"
    ],
    python_requires=">=3.9, <3.12",
)
