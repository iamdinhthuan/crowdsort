"""LibriSpeech train-clean-100 with MFA word alignments as a lhotse CutSet for FastMSS."""
from lhotse import CutSet
from lhotse.recipes.librispeech import prepare_librispeech

R = "/mnt/ssd2/paper/paper_1/data/raw"
m = prepare_librispeech(f"{R}/librispeech/LibriSpeech", alignments_dir=f"{R}/librispeech_align/LibriSpeech",
                        dataset_parts=["train-clean-100"], num_jobs=8)["train-clean-100"]
cuts = CutSet.from_manifests(recordings=m["recordings"], supervisions=m["supervisions"])
n_al = sum(1 for c in cuts if c.supervisions[0].alignment and "word" in c.supervisions[0].alignment)
print(len(cuts), "cuts,", n_al, "with word alignments")
cuts.to_file(f"{R}/fastmss/manifests/librispeech_cutset_train-clean-100.jsonl.gz")
