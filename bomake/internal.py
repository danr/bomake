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
import metrohash # type: ignore


A = TypeVar('A')
P = ParamSpec('P')
R = TypeVar('R')


def is_function(f: Any) -> TypeGuard[FunctionType]:
    return isinstance(getattr(f, '__code__', None), CodeType)


def function_repr(f: FunctionType) -> tuple[str, bytes]:
    return (f.__code__.co_name, f.__code__.co_code)


@dataclass(frozen=True, slots=True)
class Marker:
    mark: str


FunctionMark = Marker('function')
DictMark = Marker('dict')


def replace(x: Any) -> Any:
    fns: dict[Any, Any] = {}
    def go(x: Any) -> Any:
        if is_function(x):
            repr = function_repr(x)
            if repr not in fns:
                fns[repr] = ...
                fns[repr] = go((
                    x.__code__.co_name,
                    x.__code__.co_consts,
                    x.__code__.co_code,
                    [getattr(c, 'cell_contents', None) for c in x.__closure__ or []],
                    getattr(x, '__self__', None),
                ))
            return (FunctionMark, *repr)
        elif type(x) in (tuple, set, list, frozenset):
            return (
                Marker(str(type(x))),
                *type(x)(map(go, x))
            )
        elif type(x) == dict:
            return (
                DictMark,
                *((go(k), go(v)) for k, v in sorted(x.items()))
            )
        elif isinstance(x, Path):
            return x.read_bytes()
        elif isinstance(x, Bo):
            return (Marker('Bo'), x.path)
        elif is_dataclass(x):
            return (
                Marker(x.__class__.__qualname__),
                *((f.name, go(getattr(x, f.name))) for f in fields(x))
            )
        else:
            return x
    return go(x), list(sorted(fns.items()))


def digest(x: Any) -> str:
    with Bo.timeit('replace'):
        repr = replace(x)
    with Bo.timeit('pkl'):
        pkl = pickle.dumps(repr)
    with Bo.timeit('hash'):
        return hex(cast(Any, metrohash).hash128_int(pkl))


schema = '''
    create table if not exists data(
        arg_digest text,
        res_pickle blob
    );
    create table if not exists atimes(
        arg_digest text,
        atime real
    );
    create view if not exists bo as
        select
            data.arg_digest                                  as arg_digest,
            datetime(atimes.atime, 'unixepoch', 'localtime') as atime,
            data.res_pickle                                  as res_pickle
        from
            data, atimes
        where
            data.arg_digest = atimes.arg_digest
        order by
            atimes.atime
    ;
    -- create index if not exists bo_arg_digest on bo(arg_digest);
    pragma journal_mode = WAL;
    pragma wal_autocheckpoint = 0;
    pragma synchronous = OFF;
'''


@dataclass(frozen=True)
class DB:
    conn: sqlite3.Connection
    def insert(self, table_name: str, data: dict[str, Any]):
        colnames = ','.join(data.keys())
        qstmarks = ','.join('?' for _ in data.keys())
        self.conn.execute(
            f'insert into {table_name}({colnames}) values ({qstmarks})', [*data.values()]
        )

    def select(self, table_name: str, exprs: str, **where: Any) -> sqlite3.Cursor:
        if not where:
            where = {'1': 1}
        return self.conn.execute(
            f'select {exprs} from {table_name} where '
            + ' and '.join(f'{k} = ?' for k in where.keys()),
            [*where.values()],
        )

    def update(self, table_name: str, data: dict[str, Any], **where: Any):
        if not where:
            where = {'1': 1}
        colnames = ','.join(data.keys())
        qstmarks = ','.join('?' for _ in data.keys())
        sql = f'update {table_name} set ({colnames}) = ({qstmarks}) where '
        sql = sql + ' and '.join(f'{k} = ?' for k in where.keys())
        return self.conn.execute(
            sql,
            [*data.values(), *where.values()],
        ).fetchall()


@dataclass
class Bo:
    path: str = 'bo.db'
    db: DB = cast(Any, None)

    def __post_init__(self):
        self.db = DB(sqlite3.connect(self.path, isolation_level=None))
        self.db.conn.executescript(schema)

    def __call__(self, f: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(f)
        def F(*args: P.args, **kwargs: P.kwargs):
            with self.timeit('call'):
                with self.timeit('digest'):
                    arg_digest = digest((f, args, kwargs))
                atime = datetime.now().timestamp()
                res_pickles = self.db.select('data', 'res_pickle', arg_digest=arg_digest).fetchall()
                if res_pickles:
                    (res_pickle,), *_ = res_pickles
                    with Bo.timeit('unpickle'):
                        res = pickle.loads(res_pickle)
                    with self.timeit('update db'):
                        self.db.update('atimes', dict(atime=atime), arg_digest=arg_digest)
                    return res
                else:
                    with self.timeit('get result'):
                        res = f(*args, **kwargs)
                    with self.timeit('insert into db'):
                        self.db.insert('data', dict(
                            arg_digest=arg_digest,
                            res_pickle=pickle.dumps(res),
                        ))
                        self.db.insert('atimes', dict(
                            arg_digest=arg_digest,
                            atime=atime,
                        ))
                    return res
        return F

    def dump(self):
        for row in self.db.select(
            'bo',
            'arg_digest, atime, length(res_pickle)',
        ):
            print(row)


    def stats(self):
        print(
            'rows:',
            len(list(self.db.select('data', 'rowid'))),
            'size:',
            self.db.select('data', 'sum(length(res_pickle))').fetchone()[0]
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
            if 0 or cls.depth == 0:
                print(' ' * cls.depth + f'{T/1e6:.1f}ms {desc}')

        return worker()



