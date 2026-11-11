"""Validated, non-overlapping windows for this project's 10 ps MD output."""
import numpy as np

FRAME_PS = 10
TIME_TOLERANCE_PS = 0.01  # permits single-precision XTC/ns text round-off


def validated_time_ps(data, total_ns, unit='ps'):
    """Reject incomplete, duplicated, reordered or nonfinite time series."""
    data = np.asarray(data)
    if total_ns not in (6, 11):
        raise ValueError('Supported trajectory lengths are 6 and 11 ns')
    if data.ndim != 2 or data.shape[1] < 2 or not np.isfinite(data).all():
        raise ValueError('Expected finite time/value rows')
    if unit not in ('ps', 'ns'):
        raise ValueError('Time unit must be ps or ns')
    raw_ps = data[:, 0] * (1000 if unit == 'ns' else 1)
    rounded_ps = np.rint(raw_ps).astype(np.int64)
    if np.any(np.abs(raw_ps - rounded_ps) > TIME_TOLERANCE_PS):
        raise ValueError('Times do not match integer ps within tolerance')
    expected = np.arange(0, total_ns * 1000 + 1, FRAME_PS)
    if not np.array_equal(rounded_ps, expected):
        raise ValueError(f'Expected every {FRAME_PS} ps from 0 to {total_ns * 1000}; got {len(raw_ps)} frames')
    return rounded_ps


def window_mask(time_ps, index, total_ns):
    """[start, end), except the final window includes the final frame."""
    start, end = index * 1000, (index + 1) * 1000
    return (time_ps >= start) & ((time_ps <= end) if index == total_ns - 1 else (time_ps < end))


def window_metrics(data, total_ns, unit='ps', column=1):
    time_ps = validated_time_ps(data, total_ns, unit)
    result = []
    for index in range(total_ns):
        selected = window_mask(time_ps, index, total_ns)
        x, y = time_ps[selected] / 1000.0, data[selected, column]
        result.append((float(y.mean()), float(y.std()), float(np.polyfit(x, y, 1)[0]), len(y)))
    return result
