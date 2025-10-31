import os
from pathlib import Path
import random
import string
import math

from typing import Tuple

import pandas as pd
import tempfile


def atomic_save(df: pd.DataFrame, target_path: Path) -> None:
    """
    Save a DataFrame to a temp file, flush OS and Python caches, then delete
    old file and renames original to correct filename

    :param df:              The DataFrame to save to the target_path
    :param target_path:     The ultimate location to save df to
    """
    target_path = Path(target_path)
    tmp_dir = target_path.parent

    # Create a temporary file in the same directory (so rename stays atomic)
    with tempfile.NamedTemporaryFile(dir=tmp_dir, delete=False, suffix=".tmp") as tmp_file:
        tmp_path = Path(tmp_file.name)
        df.to_parquet(tmp_path, index=False)

    # Flush + sync to ensure durability
    tmp_file = open(tmp_path, "rb")
    os.fsync(tmp_file.fileno())
    tmp_file.close()

    # Replace original file atomically
    os.replace(tmp_path, target_path)


def gen_random_string(length: int) -> str:
    characters = string.ascii_letters + string.digits  # Letters and digits
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return str(random_string)


def construct_filestore_safe_filepath(input_path: str, create_path=True, return_path_obj=False,
                                      filestore_path_override: str=None) -> str | Path:
    """
        If input_path starts with one of the recognized keywords
    ('bronze', 'silver', 'gold', 'platinum'), that portion is replaced by
    the corresponding environment variable. Then, if additional subfolders
    or a file name follow, they are appended to that root path.

    If input_path does not start with one of these keywords, the path remains
    as is (possibly absolute or relative). In either case, any missing
    directories in the path are created.

    Returns the final absolute path (including file name if present).

    :param input_path:
    :param create_path:
    :param return_path_obj:
    :param filestore_path_override:
    :return:
    """

    # Normalize path (remove leading/trailing slashes for splitting).
    # We also handle backslashes on Windows by converting them to forward slashes if desired,
    # or you can split by os.sep directly.
    cleaned_path = input_path.strip().replace('\\', '/').strip('/')
    parts = cleaned_path.split('/')

    # Possible special roots:
    valid_roots = ('bronze', 'silver', 'gold', 'platinum')

    if parts and parts[0].lower() in valid_roots:
        # The first segment is one of the special keywords.
        root_key = parts[0].lower()  # e.g. "bronze"

        if filestore_path_override:
            root_dir = Path(filestore_path_override)
        else:
            # Get the environment variable for this key
            root_dir = os.environ[root_key + "_path"]

        # Append any remaining subdirectories or file
        sub_path = os.path.join(*parts[1:]) if len(parts) > 1 else ""

        # Construct the final path
        final_path = os.path.join(root_dir, sub_path)
    else:
        # No recognized keyword -> use the path as given
        final_path = input_path

    # Convert final_path to an absolute path for clarity
    final_path = os.path.abspath(final_path)

    # Ensure the directory portion exists.
    # os.path.dirname(...) handles the case if final_path is a file or a folder.
    # If final_path might not have a file name, we can handle that as well.
    if create_path:
        directory_part = os.path.dirname(final_path)
        if directory_part:  # non-empty means there's a parent directory to create
            os.makedirs(directory_part, exist_ok=True)
        else:
            # If there's no directory part (unlikely if final_path isn't just ""),
            # you might want to handle that differently, e.g., create final_path itself.
            os.makedirs(final_path, exist_ok=True)

    if return_path_obj:
        return Path(final_path)
    else:
        return final_path


def bytes_to_human(num_bytes: int) -> str:
    """Convert a byte count into a human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} PB"


def suggested_threads(qps: float | None = None, p95_latency_s: float | None = None,
                      safety: float = 1.2, cap: int = 64) -> int:
    """Estimate a thread count for I/O-bound work from rate limit and latency.

    Computes threads ≈ qps × p95_latency_s × safety, capped at `cap`.
    If either `qps` or `p95_latency_s` is missing, falls back to
    `int(os.environ.get("PROM_MT_WORKERS", "32"))`.

    Args:
        qps: Max permitted requests per second (RPS). None to skip sizing.
        p95_latency_s: 95th percentile request latency in seconds. None to skip sizing.
        safety: Headroom multiplier
        cap: Upper bound on threads (must be ≥ 1).

    Returns:
        Suggested number of worker threads (≥ 1).
    """
    if qps and p95_latency_s:
        return max(1, min(int(qps * p95_latency_s * safety), cap))
    # fallback to env/constant if you don’t know limits yet
    return int(os.environ.get("PROM_MT_WORKERS", "32"))


def optimize_chunk_qty_and_size(workers: int, function_calls: int, min_chunk_size: int = 20) -> Tuple[int, int, int]:
    """
    This util helps size number and size of chunks based on the number of available threads, number of API calls,
    and the target chunks per thread.

    Returns:    (tuple) ideal_chunks, ideal_chunk_size, and workers.

    Variables:  workers (int): Total number of workers available to use for threads or processes

                function_calls (int) : The total number of items to be iterated through (eg. chunks * chunk_size)

                min_chunk_size (int)(dflt: 20): This variable controls the minimum size of chunks. Each chunk incurs
                    overhead from spinning up the process/thread. On macOS, thread spawning costs 20-100 micros, then
                    each thread incurs its setup time (retrieving data, any operations within the thread like
                    connecting to servers). Processes require 2-5 milliseconds (20,000 - 50,000 micros) to spawn, plus
                    per-chunk setup (retrieving data, connecting to servers, etc.).  These amounts must be amortized
                    over the course of the program lifetime while accounting for concurrency of multiple workers.

                    Recommended to keep default at 20, but for extra long io, reduce is OK

    Problem:    Too few chunks and threads may finish at different times with nothing to work on, too many chunks
                and the system spends too much time spinning up threads. Therefore, the goal is to have all threads
                complete at the same time to minimize wasted resources.

    Process:    The total population size is the number of function calls. The goal is to optimize the number of
                draws (chunks) and the size of the draw (chunk size) so that we can maximize the likelihood that we
                generate a series of chunks/sizes that minimize variance.

    Theory:     N = total number of tasks, W = number of workers, M = number of chunks, s = chunk size, k = safety factor

                M = k WlogW; chunks = safety factor * workers * log workers
                    W log(W) is the point at which all chunks are expected to be hit with high probability
                    (Assuming this follows a poisson distribution, this is the greatest binomial probability)
                    scaling by k improves liklihood of missing a chunk by 1/M^k
    """

    if workers < 1:
        raise ValueError("workers must be >= 1")
    if function_calls < 1:
        return 1, 1
    if min_chunk_size < 1:
        min_chunk_size = 1

    kappa = max(math.log(function_calls, 10), 1)

    def get_chunks(kappa, workers, function_calls):
        ideal_chunks = max(math.ceil(kappa * workers * math.log(workers)), workers) # k*WlogW
        ideal_chunk_size = max(1, math.ceil(function_calls/ideal_chunks))
        return ideal_chunks, ideal_chunk_size

    while True: # No binary search - juice not worth the squeeze
        ideal_chunks, ideal_chunk_size = get_chunks(kappa, workers, function_calls)
        if workers == 1:
            break
        elif ideal_chunk_size < min_chunk_size:
            workers -= 1
            continue
        break

    return ideal_chunks, ideal_chunk_size, workers


def df_upsert(
    base_dataframe: pd.DataFrame,
    incoming_dataframe: pd.DataFrame,
    keys: str | list[str],
    *,
    overwrite: bool = True,
    add_incoming_cols: bool = False,
) -> pd.DataFrame:
    """
    Upsert rows from `incoming_dataframe` into `base_dataframe` using `keys` as the unique key(s).

    Args:
        base_dataframe: Existing dataframe to be updated.
        incoming_dataframe: New dataframe containing rows/values to upsert.
        keys: Column name or list of column names that define uniqueness.
        overwrite: Passed to pandas.DataFrame.update(overwrite=...).
                   True  -> replace with non-NA values from incoming
                   False -> only fill NAs in base
        add_incoming_cols: If True, columns from incoming_dataframe not present in base will be added

    Returns:
        pd.DataFrame: The updated + inserted dataframe.
    """
    if isinstance(keys, str):
        keys = [keys]

    if base_dataframe is None:
        if add_incoming_cols:
            base_dataframe = incoming_dataframe.head(0).copy()
        else:
            base_dataframe = pd.DataFrame()

    incoming_dataframe = incoming_dataframe.drop_duplicates(subset=keys, keep="last") # keep=last to match w/ postgres

    # Index on keys (keep key columns as regular columns via drop=False)
    base_idx = base_dataframe.set_index(keys, drop=False).copy()
    incoming_idx = incoming_dataframe.set_index(keys, drop=False).copy()

    # Add missing columns if elected
    if add_incoming_cols:
        for col in (set(incoming_idx.columns) - set(base_idx.columns)) - set(keys):
            base_idx[col] = pd.NA

    # Update ONLY non-key columns
    incoming_nonkey = incoming_idx.drop(columns=keys, errors="ignore")
    base_idx.update(incoming_nonkey, overwrite=overwrite)

    # New rows: keys present in incoming but not in base
    new_rows = incoming_idx.loc[~incoming_idx.index.isin(base_idx.index)]

    # Concatenate and return
    result = pd.concat([base_idx, new_rows], axis=0, ignore_index=False)
    return result.reset_index(drop=True)


