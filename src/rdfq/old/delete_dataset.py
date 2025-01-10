import huggingface_hub

huggingface_hub.delete_repo(
    repo_id="permutans/wdc-common-crawl-embedded-jsonld", repo_type="dataset"
)
