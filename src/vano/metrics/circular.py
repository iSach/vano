"""Directional statistics for the InSAR benchmark (paper section 6.4).

Interferogram values are angles, so ordinary moments are meaningless.  Following
the paper we summarise each field by its circular variance and circular skewness
and compare the *distribution* of those summaries between data and samples.
"""

import torch


def _trigonometric_moments(phase, order):
    axes = tuple(range(1, phase.dim()))
    n = phase[0].numel()
    c = torch.cos(order * phase).sum(axes) / n
    s = torch.sin(order * phase).sum(axes) / n
    return c, s


def circular_variance(phase):
    """``1 - R``, with ``R`` the mean resultant length of each field."""
    c1, s1 = _trigonometric_moments(phase, 1)
    return 1.0 - torch.hypot(c1, s1)


def circular_skewness(phase):
    """Second-order circular skewness ``R2 sin(T2 - 2 T1) / (1 - R1)^{3/2}``."""
    c1, s1 = _trigonometric_moments(phase, 1)
    c2, s2 = _trigonometric_moments(phase, 2)
    r1, r2 = torch.hypot(c1, s1), torch.hypot(c2, s2)
    t1, t2 = torch.atan2(s1, c1), torch.atan2(s2, c2)
    return r2 * torch.sin(t2 - 2.0 * t1) / (1.0 - r1).pow(1.5)
