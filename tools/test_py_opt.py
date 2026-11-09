#!/usr/bin/env python3
from paddlelite.lite import Opt
import os
import sys

pdmodel = sys.argv[1] if len(sys.argv) > 1 else "output/test_rec_inf/inference.pdmodel"
pdiparams = sys.argv[2] if len(sys.argv) > 2 else "output/test_rec_inf/inference.pdiparams"
out_prefix = sys.argv[3] if len(sys.argv) > 3 else "output/test_rec_inf/plate_rec_small"

print(f"Optimizing {pdmodel} via PaddleLite Python API...")
opt = Opt()
opt.set_model_file(pdmodel)
opt.set_param_file(pdiparams)
opt.set_valid_places("arm")
opt.set_model_type("naive_buffer")
opt.set_optimize_out(out_prefix)
opt.run()
print(f"Done! Check if {out_prefix}.nb exists.")
