from __future__ import annotations
from typing import *
from dataclasses import *
from types import FunctionType, CodeType
import pickle
import sqlite3
import functools
from datetime import datetime
from pathlib import Path
import time
from contextlib import contextmanager
import metrohash  # type: ignore

A = TypeVar('A')
P = ParamSpec('P')
R = TypeVar('R')

def is_function(f: Any) -> TypeGuard[FunctionType]:
    return isinstance(getattr(f, '__code__', None), CodeType)


def code_repr(f: CodeType) -> tuple[str, bytes]:
    return (f.co_name, f.co_code)


def normalize(x: Any) -> Any:
    fns: dict[Any, Any] = {}

    def go(x: Any) -> Any:
        if isinstance(x, CodeType):
            return code_repr(x)
        elif is_function(x):
            repr = code_repr(x.__code__)
            if repr not in fns:
                fns[repr] = ...
                fns[repr] = go(
                    (
                        x.__code__.co_consts,
                        [
                            getattr(c, 'cell_contents', None)
                            for c in x.__closure__ or []
                        ],
                        getattr(x, '__self__', None),
                        getattr(x, '__defaults__', None),
                    )
                )
            return ('fn', *repr)
        elif type(x) in (tuple, set, list, frozenset):
            return (
                f'{type(x)}',
                *type(x)(map(go, x)),
            )
        elif type(x) == dict:
            return (
                'dict',
                *((go(k), go(v)) for k, v in sorted(x.items())),
            )
        elif isinstance(x, Path):
            return x.read_bytes()
        elif isinstance(x, Bo):
            return ('bo', x.path)
        elif is_dataclass(x):
            return (
                f'data {x.__class__.__qualname__}',
                *((f.name, go(getattr(x, f.name))) for f in fields(x)),
            )
        else:
            return x

    return go(x), list(sorted(fns.items()))


def calculate_digest(x: Any) -> str:
    with Bo.timeit('normalize'):
        repr = normalize(x)
    with Bo.timeit('pkl'):
        try:
            pkl = pickle.dumps(repr)
        except:
            print('Cannot pickle:', repr)
            raise
    with Bo.timeit('hash'):
        return hex(cast(Any, metrohash).hash128_int(pkl))


schema = '''
    create table if not exists data(
        digest text,
        binary blob
    );
    create table if not exists atimes(
        digest text,
        atime real
    );
    create view if not exists bo as
        select
            data.digest                                      as digest,
            datetime(atimes.atime, 'unixepoch', 'localtime') as atime,
            data.binary                                      as binary
        from
            data, atimes
        where
            data.digest = atimes.digest
        order by
            atimes.atime
    ;
    -- create index if not exists bo_arg_digest on bo(digest);
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
        return self.conn.execute(
            f'insert into {table_name}({colnames}) values ({qstmarks})',
            [*data.values()],
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
        )

    def delete(self, table_name: str, **where: Any):
        if not where:
            where = {'1': 1}
        sql = f'delete from {table_name} where '
        sql = sql + ' and '.join(f'{k} = ?' for k in where.keys())
        return self.conn.execute(
            sql,
            [*where.values()],
        )

class Serializer:
    def dumps(self, object: Any) -> bytes:
        return pickle.dumps(object)

    def loads(self, binary: bytes) -> Any:
        return pickle.loads(binary)

@dataclass
class Bo:
    path: str = 'bo.db'
    _db: DB = cast(Any, None)
    extra_state: Callable[[], Any] = field(default_factory=lambda: lambda: None)
    serializer: Serializer = field(default_factory=lambda: Serializer())

    Serializer: ClassVar = Serializer

    @property
    def db(self):
        if cast(Any, self._db) is None:
            self._db = DB(sqlite3.connect(self.path, isolation_level=None, check_same_thread=False))
            self._db.conn.executescript(schema)
        return self._db

    def __enter__(self):
        return self

    def __exit__(self, *err: Any):
        self.close()

    def close(self):
        self.db.conn.close()

    def with_extra_state(self, k: Callable[[], Any]) -> Bo:
        return replace(self, extra_state=lambda: (self.extra_state(), k()))

    def with_serializer(self, serializer: Serializer) -> Bo:
        return replace(self, serializer=serializer)

    def __call__(self, f: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(f)
        def F(*args: P.args, **kwargs: P.kwargs):
            with self.timeit(f'call'):
                with self.timeit('digest'):
                    extra_state = (self.extra_state(), self.serializer.__class__.__qualname__)
                    digest = calculate_digest(((f, args, kwargs), extra_state))
                atime = datetime.now().timestamp()
                res_binaries = self.db.select(
                    'data', 'binary', digest=digest
                ).fetchall()
                if res_binaries:
                    (binary,), *_ = res_binaries
                    with Bo.timeit('loads'):
                        res = self.serializer.loads(binary)
                    with self.timeit('update db'):
                        self.db.update(
                            'atimes', dict(atime=atime), digest=digest
                        )
                    return res
                else:
                    with self.timeit('get result'):
                        res = f(*args, **kwargs)
                    with self.timeit('dumps'):
                        binary = self.serializer.dumps(res)
                    with self.timeit('insert into db'):
                        self.db.insert(
                            'data',
                            dict(
                                digest=digest,
                                binary=binary,
                            ),
                        )
                        self.db.insert(
                            'atimes',
                            dict(
                                digest=digest,
                                atime=atime,
                            ),
                        )
                    return res

        return F

    def dump(self):
        for row in self.db.select(
            'bo',
            'digest, atime, length(binary)',
        ):
            print(row)

    def stats(self):
        print(
            'rows:',
            len(list(self.db.select('data', 'rowid'))),
            'size:',
            self.db.select('data', 'sum(length(binary))').fetchone()[0],
        )

    def clear(self):
        self.db.delete('data')
        self.db.delete('atimes')

    depth: ClassVar[int] = 0

    @classmethod
    def timeit(cls, desc: str = ''):
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
