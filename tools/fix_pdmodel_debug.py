#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import paddle.base.proto.framework_pb2 as framework_pb2

pdmodel_path = sys.argv[1] if len(sys.argv) > 1 else "output/test_rec_inf/inference.pdmodel"

program = framework_pb2.ProgramDesc()
with open(pdmodel_path, "rb") as f:
    program.ParseFromString(f.read())

swish_count = 0
hs_count = 0

for block in program.blocks:
    for op in block.ops:
        op_type = str(op.type)
        if "swish" in op_type:
            has_beta = any(attr.name == "beta" for attr in op.attrs)
            if not has_beta:
                attr = op.attrs.add()
                attr.name = "beta"
                attr.type = 1
                attr.f = 1.0
                swish_count += 1
        if "hard_sigmoid" in op_type or "hardsigmoid" in op_type:
            has_slope = any(attr.name == "slope" for attr in op.attrs)
            if not has_slope:
                attr = op.attrs.add()
                attr.name = "slope"
                attr.type = 1
                attr.f = 0.2
                hs_count += 1
            has_offset = any(attr.name == "offset" for attr in op.attrs)
            if not has_offset:
                attr = op.attrs.add()
                attr.name = "offset"
                attr.type = 1
                attr.f = 0.5
                hs_count += 1

print(f"成功补全了 {swish_count} 个 swish 属性与 {hs_count} 个 hard_sigmoid 属性。")
with open(pdmodel_path, "wb") as f:
    f.write(program.SerializeToString())
