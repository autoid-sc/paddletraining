#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix missing default attributes ('beta' for swish, 'slope' and 'offset' for hard_sigmoid)
in Paddle inference models for PaddleLite compatibility.
"""
import os
import sys

try:
    import paddle.base.proto.framework_pb2 as framework_pb2
except ImportError:
    try:
        import paddle.fluid.proto.framework_pb2 as framework_pb2
    except ImportError:
        import paddle.framework.proto.framework_pb2 as framework_pb2

def fix_pdmodel_attrs(pdmodel_path):
    if not os.path.exists(pdmodel_path):
        print(f"[ERROR] File not found: {pdmodel_path}")
        return False

    print(f"Reading protobuf model: {pdmodel_path}...")
    program = framework_pb2.ProgramDesc()
    with open(pdmodel_path, "rb") as f:
        program.ParseFromString(f.read())

    swish_count = 0
    hs_count = 0

    for block in program.blocks:
        for op in block.ops:
            if op.type == "swish":
                has_beta = any(attr.name == "beta" for attr in op.attrs)
                if not has_beta:
                    attr = op.attrs.add()
                    attr.name = "beta"
                    attr.type = 1 # FLOAT
                    attr.f = 1.0
                    swish_count += 1
            elif op.type == "hard_sigmoid":
                has_slope = any(attr.name == "slope" for attr in op.attrs)
                if not has_slope:
                    attr = op.attrs.add()
                    attr.name = "slope"
                    attr.type = 1 # FLOAT
                    attr.f = 0.2
                    hs_count += 1
                has_offset = any(attr.name == "offset" for attr in op.attrs)
                if not has_offset:
                    attr = op.attrs.add()
                    attr.name = "offset"
                    attr.type = 1 # FLOAT
                    attr.f = 0.5
                    hs_count += 1

    print(f"Patched {swish_count} 'swish' op(s) and {hs_count} 'hard_sigmoid' attribute(s).")
    with open(pdmodel_path, "wb") as f:
        f.write(program.SerializeToString())
        
    print(f"[SUCCESS] Saved patched model back to {pdmodel_path}")
    return True

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "output/test_rec_inf/inference.pdmodel"
    fix_pdmodel_attrs(path)
