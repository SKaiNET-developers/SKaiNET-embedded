"""Dataclasses for parsed StableHLO MLIR IR."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TensorType:
    shape: list[int]
    element_type: str  # "f32", "f16", "i32", etc.

    @property
    def num_elements(self) -> int:
        result = 1
        for dim in self.shape:
            result *= dim
        return result

    def __str__(self) -> str:
        dims = "x".join(str(d) for d in self.shape)
        return f"tensor<{dims}x{self.element_type}>"


@dataclass
class ConstantOp:
    result_name: str
    values: list[float]
    result_type: TensorType


@dataclass
class ConvertOp:
    result_name: str
    operand: str  # SSA name
    input_type: TensorType
    result_type: TensorType


@dataclass
class ConvolutionOp:
    result_name: str
    lhs: str  # SSA name
    rhs: str  # SSA name
    lhs_type: TensorType
    rhs_type: TensorType
    result_type: TensorType
    strides: list[int]
    padding: list[list[int]]  # [[top, bottom], [left, right]]
    rhs_dilate: list[int]
    batch_group_count: int
    feature_group_count: int


@dataclass
class BinaryOp:
    """Element-wise binary operation (add, multiply, subtract, divide)."""

    op: str  # "add", "multiply", "subtract", "divide"
    result_name: str
    lhs: str  # SSA name
    rhs: str  # SSA name
    result_type: TensorType


@dataclass
class ReturnOp:
    values: list[str]  # SSA names
    types: list[TensorType]


Op = ConstantOp | ConvertOp | ConvolutionOp | BinaryOp | ReturnOp


@dataclass
class FuncDef:
    name: str
    args: list[tuple[str, TensorType]]  # [(name, type), ...]
    return_types: list[TensorType]
    body: list[Op]


@dataclass
class Module:
    functions: list[FuncDef]
