# stdlib
import io
import os
import re

# third party
from setuptools import setup

here = os.path.realpath(os.path.dirname(__file__))

name = "compfs"

# for simplicity we actually store the version in the __version__ attribute in the source
with io.open(os.path.join(here, name, "__init__.py")) as f:
    meta_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if meta_match:
        version = meta_match.group(1)
    else:
        raise RuntimeError("Unable to find __version__ string.")


if __name__ == "__main__":
    try:
        setup(
            version=version,
        )
    except:  # noqa
        print(
            "\n\nAn error occurred while building the project, "
            "please ensure you have the most updated version of setuptools, "
            "setuptools_scm and wheel with:\n"
            "   pip install -U setuptools setuptools_scm wheel\n\n",
        )
        raise
