import multiprocessing as mp
import shutil
import subprocess
import time
import traceback
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path

import polars as pl
from datasets import Dataset, get_dataset_config_names
from datasets.exceptions import DatasetNotFoundError
from huggingface_hub import login
from tqdm import tqdm

from rdfq.core.caching import make_cache_path, mktemp_cache_dir

# Authentication and dataset configuration
login(new_session=False)

username = "permutans"
result_dataset_name = "wdc-common-crawl-embedded-jsonld"
result_dataset_id = f"{username}/{result_dataset_name}"
repo_id = "wbsg-uni-mannheim/wdc-page"
REPO_URL = f"https://github.com/{repo_id}.git"
usb_hdd = "/media/louis/Passport"

n_cpus = mp.cpu_count()
# If set, use `non_tmp_cache_dir` instead of /tmp for repo and intermediate pq files
(non_tmp_cache_dir := Path.home() / ".cache" / "wdc-files").mkdir(exist_ok=True)
(external_cache_dir := Path(usb_hdd) / "data" / "wdc-files").mkdir(exist_ok=True)
# Make a specific dataset parquet cache dir to clear after an upload finishes (or fails)
dataset_pq_cache_dir = non_tmp_cache_dir / "ds-pq-store"
dataset_pq_cache_dir.mkdir(exist_ok=True)

# WDC use "a variation on the n-quads format" (.nq), like n-triples but with 4 fields
nq_pat = (
    r"(?P<subject><[^>]+>|_:[\w-]+)\s+"  # Subject: IRI or blank node
    r"(?P<predicate><[^>]+>)\s+"  # Predicate: IRI
    r"(?P<object>"  # Object can be one of:
    r'"[^"]*(?:\\.[^"]*)*"(?:@[a-zA-Z-]+|@\*|\^\^<[^>]+>)?|'  # Literal with optional language tag (including @*) or datatype
    r"<[^>]+>|"  # IRI
    r"_:[\w-]+"  # Blank node
    r")\s+"
    r"(?P<graph><[^>]+>|_:[\w-]+)"  # Graph: IRI or blank node
    r"\s*\."  # Full stop
)
parse_line = pl.col("*").str.extract_groups(nq_pat).struct.unnest()


def clone_or_pull_repo(repo_path: Path):
    git_dir = repo_path / ".git"
    if not git_dir.exists():
        subprocess.run(["git", "clone", REPO_URL, str(repo_path)], check=True)
    else:
        subprocess.run(["git", "pull"], cwd=str(repo_path), check=True)


def ds_subset_exists(dataset_id: str, subset_name: str) -> bool:
    try:
        return subset_name in get_dataset_config_names(dataset_id)
    except DatasetNotFoundError:
        return False
    except Exception as e:
        if "doesn't contain any data files" in str(e):
            # Dataset found but blank repo
            return False
        else:
            raise


def cap_nulls(df: pl.DataFrame, threshold=0) -> pl.DataFrame:
    """If any of the columns have nulls in over `threshold` rows, halt the program."""
    null_rows = df.filter(pl.any_horizontal(pl.all().is_null()))
    if total_nulls := len(null_rows):
        assert (
            total_nulls < threshold
        ), f"{total_nulls} nulls detected, regex may have a bug:\n{null_rows}"
        # print(f"Dropping {len(dropped)} null rows:")
        df = df.drop_nulls()
    return df


def process_all_years(repo_path: Path):
    ld_dir = repo_path / "structureddata"
    cache_dir = mktemp_cache_dir(id_path=repo_id, base_dir=non_tmp_cache_dir)
    # dataset_cache_path = partial(make_cache_path, cache_dir=cache_dir)

    wdc_releases = sorted(ld_dir.glob("**/html-embedded-jsonld.list"))
    for path in tqdm(wdc_releases):
        # print(f"Full path: {path}")
        rel = path.relative_to(ld_dir)
        # print(f"Parts: {rel.parts}")
        subset = rel.parts[0]
        print(f"Processing subset: {subset}")

        (subset_cache_dir := dataset_pq_cache_dir / subset).mkdir(exist_ok=True)
        (subset_parquet_cache_dir := subset_cache_dir / "parquet").mkdir(exist_ok=True)
        ss_pq_cache_path = partial(make_cache_path, cache_dir=subset_parquet_cache_dir)

        (subset_ext_c_dir := external_cache_dir / subset).mkdir(exist_ok=True)
        # (subset_arrow_cache_dir := subset_cache_dir / "arrow").mkdir(exist_ok=True)
        (subset_arrow_cache_dir := subset_ext_c_dir / "arrow").mkdir(exist_ok=True)

        try:
            if ds_subset_exists(result_dataset_id, subset):
                print(f"Skipping {subset}")
                shutil.rmtree(subset_parquet_cache_dir)
                shutil.rmtree(subset_arrow_cache_dir)
                continue

            urls_df = pl.read_csv(
                path, has_header=False, separator="\n", new_columns=["url"]
            )
            pq_caches = []

            def process_subset_chunk(source_url: str) -> Path:
                fname = Path(source_url).name
                parquet_cache_chunk = ss_pq_cache_path(fname)
                if parquet_cache_chunk.exists():
                    try:
                        # Verify we can read the cached file
                        df = pl.read_parquet(parquet_cache_chunk)
                        df = cap_nulls(df)
                    except Exception:
                        print(f"Failed to read {parquet_cache_chunk}")
                        raise
                else:
                    print(f"\nProcessing {source_url}")
                    df = pl.read_csv(
                        source_url,
                        separator="\n",
                        has_header=False,
                        comment_prefix="#",
                        new_columns=["line"],
                    ).select(parse_line)  # ).with_columns(parse_line) # for debugging
                    df = cap_nulls(df, threshold=100)  # 2-4 is acceptable (HTML junk)
                    df.write_parquet(parquet_cache_chunk)
                return parquet_cache_chunk

            for url in tqdm(list(urls_df.head()["url"])):
                parquet_cache_chunk = process_subset_chunk(url)
                pq_caches.append(parquet_cache_chunk)

            # Reload once all parts completed and upload
            # --!-- Cannot load all into RAM! --!--
            # aggregator = pl.read_parquet(pq_caches)
            print(
                f"Making Arrow dataset from [{str(pq_caches[0])}, ...,\n{str(pq_caches[-1])}] (x{len(pq_caches)})"
            )
            dataset = Dataset.from_parquet(
                list(map(str, pq_caches)),
                num_proc=n_cpus,
                cache_dir=subset_arrow_cache_dir,  # 300GB+ of Arrow files per subset
            )
            print(
                f"Made the dataset: {dataset}, deleting intermediate parquet files..."
            )
            push_start_t = time.time()
            dataset.push_to_hub(
                result_dataset_id,
                config_name=subset,
                private=False,
            )
            push_end_t = time.time()
            elapsed = timedelta(seconds=int(push_end_t - push_start_t))
            print(f"Successfully processed and uploaded {subset} in {elapsed}")
            # Ensure we are definitely only deleting the parquet directory and .lock files
            assert {"parquet"} == {
                f.name for f in subset_arrow_cache_dir.iterdir() if f.suffix != ".lock"
            }
            shutil.rmtree(subset_parquet_cache_dir)
            shutil.rmtree(subset_arrow_cache_dir)

        except KeyboardInterrupt:
            print("\nShutting down - current subset incomplete")
            return
        except Exception as e:
            subset_log = cache_dir / f"{subset}.log"
            print(
                f"\nError processing {subset}: {str(e)} (see {subset_log} for traceback)"
            )
            tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            formatted_entry = f"Date: {current_date}\n\n{tb}"
            if subset_log.exists():
                formatted_entry = subset_log.read_text() + "\n\n\n" + formatted_entry
            subset_log.write_text(formatted_entry)
            # Do not continue in case the cache needs to be manually cleared
            print("Halting to avoid creating multiple large Arrow caches")
            raise  # raise


if __name__ == "__main__":
    repo_path = mktemp_cache_dir(id_path=repo_id)
    try:
        clone_or_pull_repo(repo_path)
        process_all_years(repo_path)
    except KeyboardInterrupt:
        print("\nShutting down...")
