import sys
from huggingface_hub import snapshot_download
repo, out = sys.argv[1], sys.argv[2]
pats = sys.argv[3:] or None
p = snapshot_download(repo_id=repo, repo_type="dataset", local_dir=out, allow_patterns=pats, max_workers=8)
print("HF_DONE", p)
