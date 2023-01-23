from __future__ import annotations
from typing import *
from dataclasses import *
from types import FunctionType, CellType, MethodType, CodeType
import pickle
import sqlite3
import functools
from datetime import datetime
from pathlib import Path
import hashlib
import time
from contextlib import contextmanager

A = TypeVar('A')
P = ParamSpec('P')
R = TypeVar('R')


def is_function(f: Any) -> TypeGuard[FunctionType]:
    return isinstance(getattr(f, '__code__', None), CodeType)


@dataclass(frozen=True)
class Marker:
    mark: str


FunctionMark = Marker('function')
DictMark = Marker('dict')


def replace(x: Any) -> Any:
    print(x)
    if is_function(x):
        return replace((
            FunctionMark,
            x.__code__.co_name,
            x.__code__.co_consts,
            x.__code__.co_code,
            [getattr(c, 'cell_contents', None) for c in x.__closure__ or []],
            getattr(x, '__self__', None),
        ))
    elif type(x) in (tuple, set, list, frozenset):
        return (
            Marker(str(type(x))),
            *type(x)(map(replace, x))
        )
    elif type(x) == dict:
        return (
            DictMark,
            *((replace(k), replace(v)) for k, v in x.items())
        )
    elif isinstance(x, Path):
        return x.read_bytes()
    elif is_dataclass(x):
        return (
            Marker(x.__class__.__qualname__),
            *((f.name, replace(getattr(x, f.name))) for f in fields(x))
        )
    else:
        return x


def digest(x: Any) -> str:
    with Bo.timeit('replace'):
        repr = replace(x)
    with Bo.timeit('hash'):
        return hex(hash(repr))


schema = '''
    create table if not exists data(
        arg_digest text,
        res_pickle blob
    );
    create table if not exists atimes(
        arg_digest text,
        atime real
    );
    -- create index if not exists bo_arg_digest on bo(arg_digest);
    pragma journal_mode = WAL;
    pragma wal_autocheckpoint = 0;
    pragma synchronous = OFF;
'''


def insert(db: sqlite3.Connection, table_name: str, data: dict[str, Any]):
    colnames = ','.join(data.keys())
    qstmarks = ','.join('?' for _ in data.keys())
    db.execute(
        f'insert into {table_name}({colnames}) values ({qstmarks})', [*data.values()]
    )


def lookup(db: sqlite3.Connection, table_name: str, colnames: str, where: dict[str, Any]) -> sqlite3.Cursor:
    if not where:
        where = {'1': 1}
    return db.execute(
        f'select {colnames} from {table_name} where '
        + ' and '.join(f'{k} = ?' for k in where.keys()),
        [*where.values()],
    )


def update(db: sqlite3.Connection, table_name: str, data: dict[str, Any], *, where: dict[str, Any]):
    colnames = ','.join(data.keys())
    qstmarks = ','.join('?' for _ in data.keys())
    sql = f'update {table_name} set ({colnames}) = ({qstmarks}) where '
    sql = sql + ' and '.join(f'{k} = ?' for k in where.keys())
    return db.execute(
        sql,
        [*data.values(), *where.values()],
    ).fetchall()


@dataclass
class Bo:
    path: str = 'bo.db'
    db: sqlite3.Connection = cast(Any, None)

    def __post_init__(self):
        self.db = sqlite3.connect(self.path, isolation_level=None)
        self.db.executescript(schema)

    def __call__(self, f: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(f)
        def F(*args: P.args, **kwargs: P.kwargs):
            with self.timeit('call'):
                with self.timeit('digest'):
                    arg_digest = {
                        'arg_digest': digest((f, args, kwargs))
                    }
                atime = {
                    'atime': datetime.now().timestamp()
                }
                res_pickles = lookup(self.db, 'data', 'res_pickle', arg_digest).fetchall()
                if res_pickles:
                    (res_pickle,), *_ = res_pickles
                    with Bo.timeit('unpickle'):
                        res = pickle.loads(res_pickle)
                    with self.timeit('update db'):
                        update(self.db, 'atimes', atime, where=arg_digest)
                    return res
                else:
                    with self.timeit('get result'):
                        res = f(*args, **kwargs)
                    with self.timeit('insert into db'):
                        insert(self.db, 'data', {
                            **arg_digest,
                            'res_pickle': pickle.dumps(res),
                        })
                        insert(self.db, 'atimes', {
                            **arg_digest,
                            **atime,
                        })
                    return res
        return F

    def dump(self):
        for row in self.db.execute('''
            select
                data.arg_digest, atimes.atime, length(data.res_pickle)
            from
                data, atimes
            where
                data.arg_digest = atimes.arg_digest
        '''):
            print(row)


    def stats(self):
        print(
            'rows:',
            len(list(lookup(self.db, 'data', 'rowid', {}))),
            'size:',
            self.db.execute('select sum(length(res_pickle)) from data').fetchone()[0]
        )

    depth: ClassVar[int] = 0

    @classmethod
    def timeit(cls, desc: str=''):
        # The inferred type for the decorated function is wrong hence this wrapper to get the correct type

        @contextmanager
        def worker():
            t0 = time.monotonic_ns()
            cls.depth += 1
            yield
            cls.depth -= 1
            T = time.monotonic_ns() - t0
            if cls.depth == 0:
                print(' ' * cls.depth + f'{T/1e6:.1f}ms {desc}')

        return worker()



