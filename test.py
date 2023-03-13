"""
CSE 163 Final Project
Kyoko Kurihara, Robert Bonglamphone, Christine Cai

This file implements test functions for mosquito.py and analysis.py.
"""
import mosquito as m
import analysis as a
from typing import Any
import math


def assert_equals(expect: Any, actual: Any) -> None:
    """
    This function takes an expected value and an actual value,
    and checks if these two values are same.
    """
    if type(expect) in [int, float]:
        assert math.isclose(expect, actual),\
                f"Expected {expect}, but got {actual}"
    assert expect == actual, f"Expected {expect}, but got {actual}"


def main():
    pass


if __name__ == '__main__':
    main()
