#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fix pdmodel for PaddleLite compatibility (swish/hardsigmoid attributes).

Usage:
    python3 fix_pdmodel.py <path/to/inference.pdmodel>

PaddleLite v2.12 requires explicit beta/slope/offset attributes for
swish/hardsigmoid ops. PaddlePaddle 2.6+ export may omit them, causing
PaddleLite conversion to crash. This script patches the pdmodel file in-place.
"""
import sys
import paddle.base.proto.framework_pb2 as framework_pb2


def fix_pdmodel(pdmodel_path):
    program = framework_pb2.ProgramDesc()
    with open(pdmodel_path, 'rb') as f:
        program.ParseFromString(f.read())

    fixed = 0
    for block in program.blocks:
        for op in block.ops:
            if op.type == 'swish':
                if not any(a.name == 'beta' for a in op.attrs):
                    a = op.attrs.add()
                    a.name = 'beta'
                    a.type = 1  # FLOAT
                    a.f = 1.0
                    fixed += 1
            elif op.type in ['hard_sigmoid', 'hardsigmoid']:
                if not any(a.name == 'slope' for a in op.attrs):
                    a = op.attrs.add()
                    a.name = 'slope'
                    a.type = 1
                    a.f = 0.2
                    fixed += 1
                if not any(a.name == 'offset' for a in op.attrs):
                    a = op.attrs.add()
                    a.name = 'offset'
                    a.type = 1
                    a.f = 0.5
                    fixed += 1

    with open(pdmodel_path, 'wb') as f:
        f.write(program.SerializeToString())
    print(f"[OK] Fixed {fixed} ops in {pdmodel_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 fix_pdmodel.py <path/to/inference.pdmodel>")
        sys.exit(1)
    fix_pdmodel(sys.argv[1])
