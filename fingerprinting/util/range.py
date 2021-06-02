from typing import Optional, Tuple


def intersect(range_a: range, range_b: range) -> Optional[range]:
    if range_a.start >= range_a.stop or range_b.start >= range_b.stop:
        return None
    elif range_a.start >= range_b.stop or range_a.stop <= range_b.start:
        return None

    offset = range_b.start - range_a.start

    gcd, x, y = __extended_euclid(range_a.step, range_b.step)
    interval_a, interval_b = range_a.step // gcd, range_b.step // gcd
    step = interval_a * interval_b * gcd

    if offset % gcd != 0:
        return None

    # Apply Chinese Remainder Theorem
    crt = (offset * interval_a * (x % interval_b)) % step

    filler = 0
    if offset > 0:
        gap = offset - crt
        filler = gap if 0 == gap % step else (gap // step + 1) * step

    start = range_a.start + crt + filler
    stop = min(range_a.stop, range_b.stop)

    return range(start, stop, step)


def __extended_euclid(a: int, b: int) -> Tuple[int, int, int]:
    x, y, u, v = 0, 1, 1, 0
    while a:
        q, r = b // a, b % a
        m, n = x - u * q, y - v * q
        b, a, x, y, u, v = a, r, u, v, m, n
    return b, x, y
