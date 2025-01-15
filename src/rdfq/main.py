import multiprocessing as mp
import shutil
import subprocess
import tempfile
import time
import traceback
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path

import polars as pl
from datasets import get_dataset_config_names
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

n_cpus = mp.cpu_count()
# If set, use `non_tmp_cache_dir` instead of /tmp for repo and intermediate pq files
(non_tmp_cache_dir := Path.home() / ".cache" / "wdc-files").mkdir(exist_ok=True)
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


def cap_nulls(df: pl.DataFrame, threshold=100) -> pl.DataFrame:
    """If any of the columns have nulls in over `threshold` rows, halt the program.
    Threshold is set to 10. Typically we see at most 2-4 is acceptable (HTML junk)
    """
    null_rows = df.filter(pl.any_horizontal(pl.all().is_null()))
    if total_nulls := len(null_rows):
        assert (
            total_nulls < threshold
        ), f"{total_nulls} nulls detected, regex may have a bug:\n{null_rows}"
        # print(f"Dropping {len(dropped)} null rows:")
        df = df.drop_nulls()
    return df


def ds_subset_complete(
    repo_id: str, config_name: str, urls: list[str], split="train"
) -> tuple[bool, int]:
    """Try to read the lengths of every file in the subset.

    Returns a boolean indicating whether or not the subset is complete, along with a
    count of how many values were successfully read (indicating where to resume from).
    """
    total = len(urls)
    repo_path_prefix = f"{config_name}/{split}-"
    for idx, url in enumerate(tqdm(urls, desc=f"Scanning {repo_path_prefix}*.parquet")):
        config_split_index = f"{idx:05d}-of-{total:05d}"
        repo_path = f"{repo_path_prefix}{config_split_index}.parquet"
        hf_url = f"hf://datasets/{repo_id}/{repo_path}"
        try:
            size = pl.scan_parquet(hf_url).select(pl.len()).collect().item()
            assert size > 0
        except Exception:
            return False, idx
        else:
            pass
    return True, total


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

        try:
            urls_df = pl.read_csv(
                path, has_header=False, separator="\n", new_columns=["url"]
            )
            urls = list(urls_df["url"])

            is_complete = ds_subset_exists(result_dataset_id, subset)
            if not is_complete:
                # May be complete but without README metadata
                is_complete, seen = ds_subset_complete(
                    result_dataset_id, config_name=subset, urls=urls
                )

            if is_complete:
                print(f"Skipping {subset}")
                shutil.rmtree(subset_cache_dir)  # subset_parquet_cache_dir
                continue

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
                    df = cap_nulls(df)
                    df.write_parquet(parquet_cache_chunk)
                return parquet_cache_chunk

            for idx, url in enumerate(tqdm(urls)):
                if idx < seen:
                    # Unpadded list (just skip the first `seen` entries)
                    continue
                parquet_cache_chunk = process_subset_chunk(url)
                pq_caches.append(parquet_cache_chunk)

            # Reload once all parts completed and upload
            # --!-- Cannot load all into RAM! --!--
            # aggregator = pl.read_parquet(pq_caches)
            upload_start_t = time.time()
            upload_dataset(
                pq_caches, repo_id=result_dataset_id, config_name=subset, resume=seen
            )
            # dataset = Dataset.from_parquet(
            #     list(map(str, pq_caches)),
            #     num_proc=n_cpus,
            #     cache_dir=subset_arrow_cache_dir,  # 300GB+ of Arrow files per subset
            # )
            # print(
            #     f"Made the dataset: {dataset}, deleting intermediate parquet files..."
            # )
            # dataset.push_to_hub(
            #     result_dataset_id,
            #     config_name=subset,
            #     private=False,
            # )
            upload_end_t = time.time()
            elapsed = timedelta(seconds=int(upload_end_t - upload_start_t))
            print(f"Successfully processed and uploaded {subset} in {elapsed}")
            shutil.rmtree(subset_cache_dir)  # subset_parquet_cache_dir

        except KeyboardInterrupt:
            print(
                "\nShutting down - current subset incomplete (please rewind the remote repo to re-upload it)"
            )
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
            print(
                "Halting to avoid a missed file in the config (please rewind the remote repo to re-upload it)"
            )
            raise  # raise


def create_dataset_symlinks(
    paths: list[Path],
    config_name: str,
    split: str = "train",
    resume: int = 0,
) -> Path:
    """
    Creates a temporary directory with symlinks to dataset files, organised in the
    proper structure for `huggingface-cli upload-large-folder`.

    Args:
        paths: List of paths to parquet files to upload
        config_name: Name of the config/subset within the dataset
        split: Split name (e.g. 'train', 'test')
        resume: Starting index for file numbering

    Returns:
        Path to the temporary directory containing the symlinks
    """
    temp_dir = Path(tempfile.mkdtemp())
    config_dir = temp_dir / config_name
    config_dir.mkdir(parents=True)
    total = len(paths) + resume
    for idx, source_path in enumerate(paths):
        target_name = f"{split}-{idx+resume:05d}-of-{total:05d}.parquet"
        target_path = config_dir / target_name
        target_path.symlink_to(source_path.resolve())
    return temp_dir


def upload_dataset(
    paths: list[Path],
    repo_id: str,
    config_name: str,
    split: str = "train",
    resume: int = 0,
) -> None:
    """Upload a dataset config via a temporary directory of properly named symlinks."""
    try:
        temp_dir = create_dataset_symlinks(paths, config_name, split, resume)
        proc = subprocess.run(
            [
                "huggingface-cli",
                "upload-large-folder",
                "--repo-type",
                "dataset",
                repo_id,
                str(temp_dir),
                "--include",
                f"{config_name}/*.parquet",
            ],
            check=True,
        )
        proc.check_returncode()
    except subprocess.CalledProcessError:
        print(f"Error during upload")
        raise
    finally:
        shutil.rmtree(temp_dir)
    return


if __name__ == "__main__":
    repo_path = mktemp_cache_dir(id_path=repo_id)
    try:
        clone_or_pull_repo(repo_path)
        process_all_years(repo_path)
    except KeyboardInterrupt:
        print("\nShutting down...")
