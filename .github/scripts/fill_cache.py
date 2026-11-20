# Apache-2.0
"""easydetect's cache for an offline bundle: everything the classroom path downloads
(run with EASYDETECT_HOME pointing into the bundle)."""

from easydetect import downloads as d
from easydetect.nn import EDNet

for name in ("ed-n", "ed-s"):
    d.download_checkpoint(name)   # training starts here (COCO weights)
    # 수업용 (320 px, 100 queries) starts from the 320 fine-tune
    assert d.download_sized_checkpoint(name, 320, 100), f"{name}-320.pt not on the mirror"
    d.download_ir(name)           # predicting with the COCO model; labels.txt
    d.download_onnx(name)         # the same without OpenVINO
# the ImageNet backbones training falls back to without a checkpoint
for variant in ("n", "s"):
    assert EDNet(variant, 2).backbone.imagenet_loaded, variant
# the lab offers outlines and keypoints too: about 75 MB
d.download_segmenter()            # outlines (MobileSAM)
d.download_sam_decoder()          # training on outlines
d.download_pose("s")              # keypoints
root = d.cache_dir()
for f in sorted(root.rglob("*")):
    if f.is_file():
        print(f"{f.stat().st_size / 1e6:8.1f} MB  {f.relative_to(root)}")
