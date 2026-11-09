#!/usr/bin/env python3
import paddle.base.proto.framework_pb2 as framework_pb2

program = framework_pb2.ProgramDesc()
with open("output/test_rec_inf/inference.pdmodel", "rb") as f:
    program.ParseFromString(f.read())

for block in program.blocks:
    for op in block.ops:
        attr_names = [a.name for a in op.attrs]
        print(f"OP: {op.type:25s} Attrs: {attr_names}")
