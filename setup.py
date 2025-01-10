from setuptools import setup, find_packages

setup(
    name="atest",
    version="0.1.0",
    description="A package for extracting audio features with data augmentation.",
    author="Shubhodip Pal",
    author_email="shubhodippal01@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy==2.0.2",
        "librosa==0.10.2.post1",
        "pandas==2.2.3",
        "tqdm==4.67.1"
    ],
    python_requires=">=3.9, <3.13",
)
