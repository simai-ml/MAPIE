from __future__ import annotations

from typing import Optional

from .base import BaseCalibrator
from .ccp import GaussianCCP


def check_calibrator(
    calibrator: Optional[BaseCalibrator],
) -> BaseCalibrator:
    """
    Check if ``calibrator`` is a ``BaseCalibrator`` instance.

    Parameters
    ----------
    calibrator: Optional[BaseCalibrator]
        A ``BaseCalibrator`` instance used to estimate the conformity scores
        quantiles.

        If ``None``, use as default a ``GaussianCCP`` instance.

    Returns
    -------
    BaseCalibrator
        ``calibrator`` if defined, a ``GaussianCCP`` instance otherwise.

    Raises
    ------
    ValueError
        If ``calibrator`` is not ``None`` nor a ``BaseCalibrator`` instance.
    """
    if calibrator is None:
        return GaussianCCP()
    elif isinstance(calibrator, BaseCalibrator):
        return calibrator
    else:
        raise ValueError("Invalid `calibrator` argument. It must be `None` "
                         "or a `BaseCalibrator` instance.")
