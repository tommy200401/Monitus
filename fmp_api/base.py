import json
import os
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from pathlib import Path

import requests


class AutoName(Enum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()

    def __str__(self):
        return self.value


class Quarter(AutoName):
    Q1 = auto()
    Q2 = auto()
    Q3 = auto()
    Q4 = auto()


class FMPApi:
    endpoint: str
    _cache: dict
    domain = 'https://financialmodelingprep.com'
    use_cache = True
    cache_base_path = Path.home().joinpath('.cache')
    _cache = None
    expire = dict(days=14)

    @staticmethod
    def now():
        return datetime.utcnow().replace(tzinfo=timezone.utc)

    @staticmethod
    def parse_timestamp(timestamp):
        return datetime.strptime(timestamp, r'%Y-%m-%dT%H:%M:%S%z')

    @classmethod
    def get_expire(cls):
        return timedelta(**cls.expire)

    @classmethod
    def get_cache_path(cls):
        return cls.cache_base_path.joinpath(f'financialmodelingprep-{cls.__name__}.json')

    @classmethod
    def _get(cls, key):
        if (
            not cls.use_cache
            or not cls.cache_base_path.is_dir()
        ):
            return None
        if cls._cache is None:
            if os.path.exists(cls.get_cache_path()):
                with open(cls.get_cache_path()) as f:
                    cls._cache = json.load(f)
            else:
                cls._cache = {}
        return cls._cache.get(key)

    @classmethod
    def _set(cls, key, value):
        if (
            not cls.use_cache
            or not cls.cache_base_path.is_dir()
        ):
            return
        if cls._cache is None:
            if os.path.exists(cls.get_cache_path()):
                with open(cls.get_cache_path()) as f:
                    cls._cache = json.load(f)
            else:
                cls._cache = {}
        cls._cache[key] = value
        with open(cls.get_cache_path(), 'w') as f:
            json.dump(cls._cache, f, ensure_ascii=False)

    @classmethod
    def process(cls, value):
        '''Should be overriden for dedicated processing, such as type casting'''
        return value

    @classmethod
    def get(cls, use_cache: bool = None, **kwargs):
        if use_cache is None:
            use_cache = cls.use_cache
        value = None
        kwargs['api_key'] = 'f33b3631d5140a4f1c87e7f2eafd8fdd'
        if 'ticker' in kwargs:
            kwargs['ticker'] = kwargs['ticker'].upper()
        if use_cache:
            value = cls._get(cls.endpoint % kwargs)
        if not value:
            value = requests.get(
                cls.domain + cls.endpoint % kwargs
            ).json()
            cls._set(cls.endpoint % kwargs, value)
        return cls.process(value)
