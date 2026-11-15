"""Manifest for AliMeeting Train far-field channel 0 (after extract_ali_far.sh), forced-alignment references."""
import glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep_data import FA, ROOT, read_rttm, write_manifest, write_rttm

pairs = []
for ref in sorted(glob.glob(f"{FA}/AliMeeting/Train_Ali_far/*.rttm")):
    sess = os.path.basename(ref)[:-5]
    w = glob.glob(f"{ROOT}/wav/ali_train_far_ch0/{sess}_*.wav")
    if not w:
        print("missing", sess)
        continue
    rttm = f"{ROOT}/rttm/ali_train_far/{sess}.rttm"
    write_rttm(read_rttm(ref), sess, rttm)
    pairs.append((w[0], rttm))
write_manifest("ali_train_far", pairs)
