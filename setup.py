import setuptools

with open("readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rlego",
    version="0.0.1",
    author="NexusTeam",
    author_email="",
    description=("Building blocks for reinforcement learning."),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ministry-of-silly-code/RLego",
    classifiers=[
        "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=setuptools.find_namespace_packages(),
    python_requires=">=3.7",
)
