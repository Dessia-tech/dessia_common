# -*- coding: utf-8 -*-
"""
Setup

Changelog:
v1.0: add branch to local version if possible
"""

from setuptools import setup

from os.path import dirname, isdir, join
import re
from subprocess import CalledProcessError, check_output


def readme():
    with open("README.md") as f:
        return f.read()


tag_re = re.compile(r"\btag: %s([0-9][^,]*)\b")
version_re = re.compile("^Version: (.+)$", re.M)


def version_from_git_describe(version):
    if version[0] == "v":
        version = version[1:]
    # PEP 440 compatibility
    number_commits_ahead = 0
    if "-" in version:
        version, number_commits_ahead, commit_hash = version.split("-")
        number_commits_ahead = int(number_commits_ahead)

    split_versions = version.split(".")
    if "post" in split_versions[-1]:
        suffix = split_versions[-1]
        split_versions = split_versions[:-1]
    else:
        suffix = None
    for pre_release_segment in ["a", "b", "rc"]:
        if pre_release_segment in split_versions[-1]:
            if number_commits_ahead > 0:
                split_versions[-1] = str(
                    split_versions[-1].split(pre_release_segment)[0]
                )
                if len(split_versions) == 2:
                    split_versions.append("0")
                if len(split_versions) == 1:
                    split_versions.extend(["0", "0"])
                split_versions[-1] = str(int(split_versions[-1]) + 1)
                future_version = ".".join(split_versions)
                return "{}.dev{}+{}".format(future_version, number_commits_ahead, commit_hash)
            else:
                return ".".join(split_versions)
    if number_commits_ahead > 0:
        if len(split_versions) == 2:
            split_versions.append("0")
        if len(split_versions) == 1:
            split_versions.extend(["0", "0"])
        split_versions[-1] = str(int(split_versions[-1]) + 1)
        split_versions = ".".join(split_versions)
        return "{}.dev{}+{}".format(split_versions, number_commits_ahead, commit_hash)
    else:
        if suffix is not None:
            split_versions.append(suffix)
        return ".".join(split_versions)


# Just testing if get_version works well
assert version_from_git_describe("v0.1.7.post2") == "0.1.7.post2"
assert version_from_git_describe("v0.0.1-25-gaf0bf53") == "0.0.2.dev25+gaf0bf53"
assert version_from_git_describe("v0.1-15-zsdgaz") == "0.1.1.dev15+zsdgaz"
assert version_from_git_describe("v1") == "1"
assert version_from_git_describe("v1-3-aqsfjbo") == "1.0.1.dev3+aqsfjbo"
assert version_from_git_describe("v0.13rc0") == "0.13rc0"


def get_version():
    # Return the version if it has been injected into the file by git-archive
    version = tag_re.search("$Format:%D$")
    if version:
        return version.group(1)
    d = dirname(__file__)

    if isdir(join(d, ".git")):
        cmd = "git describe --tags"
        try:
            version = check_output(cmd.split()).decode().strip()[:]
        except CalledProcessError:
            raise RuntimeError("Unable to get version number from git tags")
        version = version_from_git_describe(version)
    else:
        # Extract the version from the PKG-INFO file.
        with open(join(d, "PKG-INFO")) as f:
            version = version_re.search(f.read()).group(1)

    # branch = get_branch()
    # if branch and branch != 'master':
    #     branch = re.sub('[^A-Za-z0-9]+', '', branch)
    #     if '+' in version:
    #         version += f'{branch}'
    #     else:
    #         version += f'+{branch}'

    return version


def get_branch():

    if isdir(join(dirname(__file__), ".git")):
        cmd = "git branch --show-current"
        try:
            return check_output(cmd.split()).decode().strip()[:]
        except CalledProcessError:
            pass

    return None


setup(
    name="dessia_common",
    version=get_version(),
    description="Common tools for DessIA software",
    long_description=readme(),
    long_description_content_type='text/markdown',
    keywords=["Dessia", "SDK", "engineering"],
    url="https://github.com/Dessia-tech/dessia-common",
    author="Dessia Technologies SAS",
    author_email="root@dessia.tech",
    include_package_data=True,
    packages=[
        "dessia_common",
        "dessia_common.workflow",
        "dessia_common.utils",
        "dessia_common.models",
        "dessia_common.models.workflows",
        "dessia_common.datatools"
    ],
    install_requires=[
        "orjson>=3.8.0",
        "typeguard",
        "networkx",
        "numpy",
        "pandas",
        "mypy_extensions",
        "scipy",
        "pyDOE",
        "pyDOE2",
        "dectree",
        "openpyxl",
        "parameterized",
        "matplotlib",
        "scikit-learn>=1.2.0",
        "cma"
    ],
    python_requires=">=3.8",
)
