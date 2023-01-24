from typing import *

from bomake import bo

from pathlib import Path

import pandas as pd

from contextlib import *

import time


def first():
    @bo
    def x(a: int, b: int) -> int:
        print('x', a, b)
        if a > 1:
            return 1 + x(a - 1, b + 1)
        else:
            return a + b

    @bo
    def h(p: Path):
        print('evaluating', p, p.read_text())
        return p.read_text().upper()

    for i in range(4):
        print(x(i, i))

    p = Path('test.tmp')
    p.write_text('example')
    print(h(p))
    p.write_text('example2')
    print(h(p))


@bo
def csv_to_parquet(p: Path, **kwargs: Any) -> Path:
    df = pd.read_csv(p, **kwargs)
    pq = p.with_suffix('.parquet')
    df.to_parquet(pq)
    return pq


read_parquet = bo(pd.read_parquet)


def second():
    with bo.timeit('csv_to_parquet'):
        pq = csv_to_parquet(Path('./examples/tox.csv'), delimiter=';')
    for i in range(2):
        with bo.timeit('read_parquet'):
            df = read_parquet(pq)
        # print(df.columns)


first()
second()

bo.stats()
bo.dump()
