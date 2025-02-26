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
        "tqdm==4.67.1",
        "tensorflow-gpu == 2.10.0",
        "joblib==1.4.2",
        "keras==2.10.0",
        "keras-Preprocessing==1.1.2",
        "matplotlib==3.8.3",
        "openpyxl==3.1.5",
        "PyAudio==0.2.14",
        "scikit-learn==1.4.1.post1",
        "scipy==1.15.1",
        "seaborn==0.13.2",
    ],
    python_requires=">=3.9, <3.12",
)
