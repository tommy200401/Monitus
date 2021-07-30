# %%
import io
import zstd
import pandas as pd


def compress(df: pd.DataFrame) -> bytes:
    cctx = zstd.ZstdCompressor()
    return cctx.compress(df.to_csv(index=False).encode())


def decompress(data: bytes) -> pd.DataFrame:
    dctx = zstd.ZstdDecompressor()
    return pd.read_csv(io.BytesIO(dctx.decompress(data)))


def read_zst(path: str) -> pd.DataFrame:
    dctx = zstd.ZstdDecompressor()
    with open(path, 'rb') as fh:
        reader = dctx.stream_reader(fh)
        return pd.read_csv(io.TextIOWrapper(reader, encoding='utf-8'))
