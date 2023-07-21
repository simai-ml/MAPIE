import codecs

from setuptools import find_packages, setup

DISTNAME = "MAPIE"
VERSION = "0.6.5"
DESCRIPTION = (
    "A scikit-learn-compatible module "
    "for estimating prediction intervals."
)
with codecs.open("README.rst", encoding="utf-8-sig") as f:
    LONG_DESCRIPTION = f.read()
LONG_DESCRIPTION_CONTENT_TYPE = "text/x-rst"
URL = "https://github.com/scikit-learn-contrib/MAPIE"
DOWNLOAD_URL = "https://pypi.org/project/MAPIE/#files"
PROJECT_URLS = {
    "Bug Tracker": "https://github.com/scikit-learn-contrib/MAPIE/issues",
    "Documentation": "https://mapie.readthedocs.io/en/latest/",
    "Source Code": "https://github.com/scikit-learn-contrib/MAPIE"
}
LICENSE = "new BSD"
MAINTAINER = "T. Cordier, V. Blot, L. Lacombe"
MAINTAINER_EMAIL = (
    "tcordier@quantmetry.com, "
    "vblot@quantmetry.com, "
    "llacombe@quantmetry.com"
)
PYTHON_REQUIRES = ">=3.7"
PACKAGES = find_packages()
INSTALL_REQUIRES = ["scikit-learn", "numpy>=1.21", "packaging"]
CLASSIFIERS = [
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: OSI Approved",
    "Topic :: Software Development",
    "Topic :: Scientific/Engineering",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10"
]

setup(
    name=DISTNAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    url=URL,
    download_url=DOWNLOAD_URL,
    project_urls=PROJECT_URLS,
    license=LICENSE,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    packages=PACKAGES,
    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    classifiers=CLASSIFIERS,
    zip_safe=False  # the package can run out of an .egg file
)
