import json
import struct
import sys
from math import prod
from pathlib import Path

from emmy.compiler.graph import Graph
from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var
from emmy.compiler.ir.kernel.ir import KernelOp, Smem, Sync, Tile, TreeHalve, WarpShuffle
from emmy.compiler.ir.stmt import Accum, Cond, StridedLoop
from emmy.compiler.ir.stmt.leaves import Assign, Let, Load, Write


_DTYPE_BYTES = {"f16": 2, "f32": 4, "f64": 8, "i32": 4, "i64": 8}
_INTEGER_DTYPES = frozenset({"i32", "i64"})


class PtxBuilder:
    def __init__(self, shapes=None, buffer_dtypes=None):
        self.lines = []
        self.counts = {"r": 4, "rd": 0, "h": 0, "f": 0, "d": 0}
        self.predicates = 0
        self.labels = 0
        self.values = {}  # IR temporary name -> PTX float register
        self.value_dtypes = {}  # IR temporary name -> semantic dtype
        self.integers = {}  # IR integer/index binding -> PTX u32 operand
        self.axes = {}
        self.shapes = shapes or {}
        self.buffer_dtypes = buffer_dtypes or {}

    def register(self, kind):
        number = self.counts[kind]
        self.counts[kind] += 1
        return f"%{kind}{number}"

    def emit(self, instruction):
        self.lines.append(f"    {instruction}")

    def predicate(self):
        result = f"%p{self.predicates}"
        self.predicates += 1
        return result

    def label(self, prefix):
        result = f"{prefix}_{self.labels}"
        self.labels += 1
        return result

    @staticmethod
    def f32_literal(value):
        bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
        return f"0f{bits:08x}"

    @staticmethod
    def f64_literal(value):
        bits = struct.unpack("<Q", struct.pack("<d", float(value)))[0]
        return f"0d{bits:016x}"

    def round_f16(self, value):
        """Round an f32 physical register to Emmy's semantic f16 value."""
        half = self.register("h")
        result = self.register("f")
        self.emit(f"cvt.rn.f16.f32 {half}, {value};")
        self.emit(f"cvt.f32.f16 {result}, {half};")
        return result

    def convert_value(self, value, source, target):
        """Convert one physical scalar register without guessing unsupported casts."""
        if source == target or (source == "f16" and target == "f32"):
            return value
        if source == "f32" and target == "f16":
            return self.round_f16(value)
        result_kind = {"f32": "f", "f64": "d", "i32": "r", "i64": "rd"}.get(target)
        if result_kind is None:
            raise NotImplementedError(f"dtype conversion {source} -> {target}")
        result = self.register(result_kind)
        conversions = {
            ("f16", "f64"): "cvt.f64.f32",
            ("f32", "f64"): "cvt.f64.f32",
            ("f64", "f32"): "cvt.rn.f32.f64",
            ("i32", "f32"): "cvt.rn.f32.s32",
            ("i64", "f32"): "cvt.rn.f32.s64",
            ("i32", "f64"): "cvt.rn.f64.s32",
            ("i64", "f64"): "cvt.rn.f64.s64",
            ("f32", "i32"): "cvt.rzi.s32.f32",
            ("f32", "i64"): "cvt.rzi.s64.f32",
            ("f64", "i32"): "cvt.rzi.s32.f64",
            ("f64", "i64"): "cvt.rzi.s64.f64",
            ("i32", "i64"): "cvt.s64.s32",
            ("i64", "i32"): "cvt.s32.s64",
        }
        instruction = conversions.get((source, target))
        if instruction is None:
            raise NotImplementedError(f"dtype conversion {source} -> {target}")
        self.emit(f"{instruction} {result}, {value};")
        return result

    def typed_operation(self, operation, args, arg_dtypes, result_dtype):
        comparisons = {"equal": "eq", "not_equal": "ne", "greater": "gt",
                       "less": "lt", "greater_equal": "ge", "less_equal": "le"}
        if operation in comparisons and len(args) == 2:
            common = arg_dtypes[0] if len(set(arg_dtypes)) == 1 else result_dtype
            common = "f32" if common == "f16" else common
            converted = [self.convert_value(value, dtype, common)
                         for value, dtype in zip(args, arg_dtypes)]
            suffix = {"f32": "f32", "f64": "f64", "i32": "s32", "i64": "s64"}[common]
            predicate = self.predicate()
            result = self.register("f")
            self.emit(f"setp.{comparisons[operation]}.{suffix} {predicate}, {converted[0]}, {converted[1]};")
            self.emit(f"selp.f32 {result}, 0f3f800000, 0f00000000, {predicate};")
            return result

        if result_dtype in ("f16", "f32"):
            converted = [self.convert_value(value, dtype, "f32")
                         for value, dtype in zip(args, arg_dtypes)]
            result = self.f32_operation(operation, converted)
            return self.round_f16(result) if result_dtype == "f16" else result

        if result_dtype == "f64":
            converted = [self.convert_value(value, dtype, "f64")
                         for value, dtype in zip(args, arg_dtypes)]
            result = self.register("d")
            binary = {"add": "add.rn.f64", "sum": "add.rn.f64",
                      "subtract": "sub.rn.f64", "sub": "sub.rn.f64",
                      "multiply": "mul.rn.f64", "prod": "mul.rn.f64",
                      "divide": "div.rn.f64", "true_divide": "div.rn.f64"}
            unary = {"negative": "neg.f64", "abs": "abs.f64", "absolute": "abs.f64",
                     "sqrt": "sqrt.rn.f64", "copy": "mov.f64", "pad": "mov.f64"}
            if operation in binary and len(converted) == 2:
                self.emit(f"{binary[operation]} {result}, {converted[0]}, {converted[1]};")
            elif operation in ("maximum", "max", "minimum", "min") and len(converted) == 2:
                predicate = self.predicate()
                comparison = "gt" if operation in ("maximum", "max") else "lt"
                self.emit(f"setp.{comparison}.f64 {predicate}, {converted[0]}, {converted[1]};")
                self.emit(f"selp.f64 {result}, {converted[0]}, {converted[1]}, {predicate};")
            elif operation in unary and len(converted) == 1:
                self.emit(f"{unary[operation]} {result}, {converted[0]};")
            elif operation == "square" and len(converted) == 1:
                self.emit(f"mul.rn.f64 {result}, {converted[0]}, {converted[0]};")
            elif operation == "reciprocal" and len(converted) == 1:
                self.emit(f"div.rn.f64 {result}, {self.f64_literal(1.0)}, {converted[0]};")
            else:
                raise NotImplementedError(f"f64 operation {operation}/{len(converted)}")
            return result

        if result_dtype in _INTEGER_DTYPES:
            converted = [self.convert_value(value, dtype, result_dtype)
                         for value, dtype in zip(args, arg_dtypes)]
            width = "32" if result_dtype == "i32" else "64"
            result = self.register("r" if width == "32" else "rd")
            binary = {
                "add": f"add.s{width}", "sum": f"add.s{width}",
                "subtract": f"sub.s{width}", "sub": f"sub.s{width}",
                "multiply": f"mul.lo.s{width}", "prod": f"mul.lo.s{width}",
                "floor_divide": f"div.s{width}", "remainder": f"rem.s{width}",
                "mod": f"rem.s{width}", "left_shift": f"shl.b{width}",
                "right_shift": f"shr.s{width}", "bitwise_and": f"and.b{width}",
                "bitwise_or": f"or.b{width}", "bitwise_xor": f"xor.b{width}",
            }
            if operation in binary and len(converted) == 2:
                self.emit(f"{binary[operation]} {result}, {converted[0]}, {converted[1]};")
            elif operation in ("copy", "pad") and len(converted) == 1:
                self.emit(f"mov.b{width} {result}, {converted[0]};")
            elif operation == "bitwise_count" and len(converted) == 1:
                if result_dtype == "i64":
                    counted = self.register("r")
                    self.emit(f"popc.b64 {counted}, {converted[0]};")
                    self.emit(f"cvt.s64.s32 {result}, {counted};")
                else:
                    self.emit(f"popc.b32 {result}, {converted[0]};")
            else:
                raise NotImplementedError(f"{result_dtype} operation {operation}/{len(converted)}")
            return result
        raise NotImplementedError(f"Elementwise dtype {result_dtype}")

    def f32_operation(self, operation, args):
        """Emit one shared f32 scalar operation and return its result register.

        Pointwise, reduction, fused and structured-loop lowering all use this
        method.  Keeping the operation mapping here prevents the category
        backends from silently assigning different meanings to the same IR op.
        """
        result = self.register("f")
        binary = {
            "add": "add.f32", "subtract": "sub.f32",
            "multiply": "mul.f32", "maximum": "max.f32",
            "minimum": "min.f32", "divide": "div.rn.f32",
            "true_divide": "div.rn.f32",
            "sum": "add.f32", "sub": "sub.f32", "prod": "mul.f32",
            "amax": "max.f32", "fmax": "max.f32", "max": "max.f32",
            "fmin": "min.f32", "min": "min.f32",
        }
        unary = {
            "negative": "neg.f32", "absolute": "abs.f32",
            "sqrt": "sqrt.rn.f32", "reciprocal": "rcp.rn.f32",
            "copy": "mov.f32", "pad": "mov.f32", "abs": "abs.f32",
        }
        if operation in binary and len(args) == 2:
            self.emit(f"{binary[operation]} {result}, {args[0]}, {args[1]};")
        elif operation in unary and len(args) == 1:
            self.emit(f"{unary[operation]} {result}, {args[0]};")
        elif operation == "relu" and len(args) == 1:
            self.emit(f"max.f32 {result}, {args[0]}, 0f00000000;")
        elif operation in ("exp", "exp_fast") and len(args) == 1:
            scaled = self.register("f")
            self.emit(f"mul.f32 {scaled}, {args[0]}, 0f3fb8aa3b;")
            self.emit(f"ex2.approx.f32 {result}, {scaled};")
        elif operation == "rsqrt" and len(args) == 1:
            self.emit(f"rsqrt.approx.f32 {result}, {args[0]};")
        elif operation in ("log", "log2", "log10") and len(args) == 1:
            logarithm = result if operation == "log2" else self.register("f")
            self.emit(f"lg2.approx.f32 {logarithm}, {args[0]};")
            if operation == "log":
                self.emit(f"mul.f32 {result}, {logarithm}, 0f3f317218;")
            elif operation == "log10":
                self.emit(f"mul.f32 {result}, {logarithm}, 0f3e9a209b;")
        elif operation in ("sin", "cos", "tanh") and len(args) == 1:
            instruction = {"sin": "sin.approx.f32", "cos": "cos.approx.f32",
                           "tanh": "tanh.approx.f32"}[operation]
            self.emit(f"{instruction} {result}, {args[0]};")
        elif operation == "sigmoid" and len(args) == 1:
            negative = self.register("f")
            scaled = self.register("f")
            exponential = self.register("f")
            denominator = self.register("f")
            self.emit(f"neg.f32 {negative}, {args[0]};")
            self.emit(f"mul.f32 {scaled}, {negative}, 0f3fb8aa3b;")
            self.emit(f"ex2.approx.f32 {exponential}, {scaled};")
            self.emit(f"add.f32 {denominator}, {exponential}, 0f3f800000;")
            self.emit(f"rcp.rn.f32 {result}, {denominator};")
        elif operation == "square" and len(args) == 1:
            self.emit(f"mul.f32 {result}, {args[0]}, {args[0]};")
        elif operation == "erf" and len(args) == 1:
            absolute = self.register("f")
            t_denominator = self.register("f")
            t = self.register("f")
            polynomial = self.register("f")
            square = self.register("f")
            negative_square = self.register("f")
            scaled = self.register("f")
            exponential = self.register("f")
            tail = self.register("f")
            magnitude = self.register("f")
            negative_magnitude = self.register("f")
            negative_input = self.predicate()
            self.emit(f"abs.f32 {absolute}, {args[0]};")
            self.emit(f"fma.rn.f32 {t_denominator}, {absolute}, 0f3ea7ba06, 0f3f800000;")
            self.emit(f"rcp.rn.f32 {t}, {t_denominator};")
            self.emit(f"fma.rn.f32 {polynomial}, {t}, 0f3f87dc22, 0fbfba00e3;")
            self.emit(f"fma.rn.f32 {polynomial}, {polynomial}, {t}, 0f3fb5f0e3;")
            self.emit(f"fma.rn.f32 {polynomial}, {polynomial}, {t}, 0fbe91a98e;")
            self.emit(f"fma.rn.f32 {polynomial}, {polynomial}, {t}, 0f3e827906;")
            self.emit(f"mul.f32 {polynomial}, {polynomial}, {t};")
            self.emit(f"mul.f32 {square}, {absolute}, {absolute};")
            self.emit(f"neg.f32 {negative_square}, {square};")
            self.emit(f"mul.f32 {scaled}, {negative_square}, 0f3fb8aa3b;")
            self.emit(f"ex2.approx.f32 {exponential}, {scaled};")
            self.emit(f"mul.f32 {tail}, {polynomial}, {exponential};")
            self.emit(f"sub.f32 {magnitude}, 0f3f800000, {tail};")
            self.emit(f"neg.f32 {negative_magnitude}, {magnitude};")
            self.emit(f"setp.lt.f32 {negative_input}, {args[0]}, 0f00000000;")
            self.emit(f"selp.f32 {result}, {negative_magnitude}, {magnitude}, {negative_input};")
        elif operation == "pow" and len(args) == 2:
            absolute = self.register("f")
            logarithm = self.register("f")
            exponent = self.register("f")
            magnitude = self.register("f")
            integer_exponent = self.register("r")
            integer_as_float = self.register("f")
            odd_bits = self.register("r")
            negative_magnitude = self.register("f")
            signed_magnitude = self.register("f")
            negative_base = self.predicate()
            exponent_is_integer = self.predicate()
            exponent_is_odd = self.predicate()
            use_negative = self.predicate()
            non_integer = self.predicate()
            invalid = self.predicate()
            self.emit(f"abs.f32 {absolute}, {args[0]};")
            self.emit(f"lg2.approx.f32 {logarithm}, {absolute};")
            self.emit(f"mul.f32 {exponent}, {logarithm}, {args[1]};")
            self.emit(f"ex2.approx.f32 {magnitude}, {exponent};")
            self.emit(f"cvt.rzi.s32.f32 {integer_exponent}, {args[1]};")
            self.emit(f"cvt.rn.f32.s32 {integer_as_float}, {integer_exponent};")
            self.emit(f"setp.eq.f32 {exponent_is_integer}, {integer_as_float}, {args[1]};")
            self.emit(f"and.b32 {odd_bits}, {integer_exponent}, 1;")
            self.emit(f"setp.ne.s32 {exponent_is_odd}, {odd_bits}, 0;")
            self.emit(f"setp.lt.f32 {negative_base}, {args[0]}, 0f00000000;")
            self.emit(f"and.pred {use_negative}, {negative_base}, {exponent_is_odd};")
            self.emit(f"not.pred {non_integer}, {exponent_is_integer};")
            self.emit(f"and.pred {invalid}, {negative_base}, {non_integer};")
            self.emit(f"neg.f32 {negative_magnitude}, {magnitude};")
            self.emit(f"selp.f32 {signed_magnitude}, {negative_magnitude}, {magnitude}, {use_negative};")
            self.emit(f"selp.f32 {result}, 0f7fc00000, {signed_magnitude}, {invalid};")
        elif operation == "sign" and len(args) == 1:
            positive = self.predicate()
            negative = self.predicate()
            nonpositive = self.register("f")
            self.emit(f"setp.gt.f32 {positive}, {args[0]}, 0f00000000;")
            self.emit(f"setp.lt.f32 {negative}, {args[0]}, 0f00000000;")
            self.emit(f"selp.f32 {nonpositive}, 0fbf800000, 0f00000000, {negative};")
            self.emit(f"selp.f32 {result}, 0f3f800000, {nonpositive}, {positive};")
        elif operation in ("equal", "not_equal", "greater", "less",
                           "greater_equal", "less_equal") and len(args) == 2:
            predicate = self.predicate()
            comparison = {"equal": "eq", "not_equal": "ne", "greater": "gt",
                          "less": "lt", "greater_equal": "ge", "less_equal": "le"}[operation]
            self.emit(f"setp.{comparison}.f32 {predicate}, {args[0]}, {args[1]};")
            self.emit(f"selp.f32 {result}, 0f3f800000, 0f00000000, {predicate};")
        elif operation in ("bitwise_and", "bitwise_or") and len(args) == 2:
            left = self.predicate()
            right = self.predicate()
            combined = self.predicate()
            instruction = "and.pred" if operation == "bitwise_and" else "or.pred"
            self.emit(f"setp.ne.f32 {left}, {args[0]}, 0f00000000;")
            self.emit(f"setp.ne.f32 {right}, {args[1]}, 0f00000000;")
            self.emit(f"{instruction} {combined}, {left}, {right};")
            self.emit(f"selp.f32 {result}, 0f3f800000, 0f00000000, {combined};")
        elif operation == "bitwise_not" and len(args) == 1:
            predicate = self.predicate()
            self.emit(f"setp.eq.f32 {predicate}, {args[0]}, 0f00000000;")
            self.emit(f"selp.f32 {result}, 0f3f800000, 0f00000000, {predicate};")
        elif operation == "where" and len(args) == 3:
            predicate = self.predicate()
            self.emit(f"setp.ne.f32 {predicate}, {args[0]}, 0f00000000;")
            self.emit(f"selp.f32 {result}, {args[1]}, {args[2]}, {predicate};")
        elif operation in ("fma", "fused_multiply_add") and len(args) == 3:
            self.emit(f"fma.rn.f32 {result}, {args[0]}, {args[1]}, {args[2]};")
        else:
            raise NotImplementedError(f"f32 operation {operation}/{len(args)}")
        return result

    def index(self, expression):
        """Compile an IR element index into an unsigned 32-bit PTX operand."""
        if isinstance(expression, Var):
            if expression.name in self.axes:
                return self.axes[expression.name]
            if expression.name in self.integers:
                return self.integers[expression.name]
            raise NotImplementedError(f"Unknown index variable {expression.name}")
        if isinstance(expression, Literal):
            if isinstance(expression.value, bool):
                return "1" if expression.value else "0"
            if not isinstance(expression.value, int):
                raise NotImplementedError("Pointwise index must be an integer")
            return str(expression.value)
        if isinstance(expression, BinaryExpr):
            left = self.index(expression.left)
            right = self.index(expression.right)
            operations = {"+": "add.u32", "-": "sub.u32", "*": "mul.lo.u32",
                          "/": "div.u32", "//": "div.u32", "%": "rem.u32"}
            if expression.op not in operations:
                raise NotImplementedError(f"Index operation {expression.op}")
            result = self.register("r")
            self.emit(f"{operations[expression.op]} {result}, {left}, {right};")
            return result
        if isinstance(expression, TernaryExpr):
            condition = self.condition_predicate(expression.cond)
            false_label = self.label("INDEX_FALSE")
            end_label = self.label("INDEX_END")
            result = self.register("r")
            self.emit(f"@!{condition} bra {false_label};")
            true_value = self.index(expression.if_true)
            self.emit(f"mov.u32 {result}, {true_value};")
            self.emit(f"bra {end_label};")
            self.lines.append(f"{false_label}:")
            false_value = self.index(expression.if_false)
            self.emit(f"mov.u32 {result}, {false_value};")
            self.lines.append(f"{end_label}:")
            return result
        raise NotImplementedError(f"Index expression {type(expression).__name__}")

    def condition_operand(self, expression):
        if isinstance(expression, Var) and expression.name in self.values:
            return self.values[expression.name], "f32"
        if isinstance(expression, Literal) and isinstance(expression.value, float):
            return self.f32_literal(expression.value), "f32"
        return self.index(expression), "s32"

    def condition_predicate(self, expression):
        if isinstance(expression, BinaryExpr) and expression.op in ("&&", "||"):
            left = self.condition_predicate(expression.left)
            right = self.condition_predicate(expression.right)
            result = self.predicate()
            operation = "and.pred" if expression.op == "&&" else "or.pred"
            self.emit(f"{operation} {result}, {left}, {right};")
            return result
        comparisons = {"==": "eq", "!=": "ne", "<": "lt", "<=": "le",
                       ">": "gt", ">=": "ge"}
        if isinstance(expression, BinaryExpr) and expression.op in comparisons:
            left, left_type = self.condition_operand(expression.left)
            right, right_type = self.condition_operand(expression.right)
            operand_type = "f32" if "f32" in (left_type, right_type) else "s32"
            if operand_type == "f32":
                if (isinstance(expression.left, Literal)
                        and not isinstance(expression.left.value, float)):
                    left = self.f32_literal(expression.left.value)
                if (isinstance(expression.right, Literal)
                        and not isinstance(expression.right.value, float)):
                    right = self.f32_literal(expression.right.value)
                if left_type != "f32" and not isinstance(expression.left, Literal):
                    raise NotImplementedError("Mixed float/index condition")
                if right_type != "f32" and not isinstance(expression.right, Literal):
                    raise NotImplementedError("Mixed float/index condition")
            result = self.predicate()
            self.emit(
                f"setp.{comparisons[expression.op]}.{operand_type} {result}, {left}, {right};")
            return result
        value, value_type = self.condition_operand(expression)
        result = self.predicate()
        zero = "0f00000000" if value_type == "f32" else "0"
        self.emit(f"setp.ne.{value_type} {result}, {value}, {zero};")
        return result

    def address(self, buffer_name, coordinates, pointers):
        """Flatten a row-major coordinate and return its dtype-aware byte address."""
        shape = self.shapes.get(buffer_name)
        if shape is None or len(coordinates) != len(shape):
            raise NotImplementedError(f"Pointwise index rank for {buffer_name}")
        element_index = "0" if not coordinates else self.index(coordinates[0])
        for dimension, coordinate in zip(shape[1:], coordinates[1:]):
            scaled = self.register("r")
            combined = self.register("r")
            self.emit(f"mul.lo.u32 {scaled}, {element_index}, {dimension};")
            self.emit(f"add.u32 {combined}, {scaled}, {self.index(coordinate)};")
            element_index = combined
        byte_offset = self.register("rd")
        address = self.register("rd")
        dtype = self.buffer_dtypes.get(buffer_name, "f32")
        itemsize = _DTYPE_BYTES.get(dtype)
        if itemsize is None:
            raise NotImplementedError(f"Pointwise buffer dtype {dtype}")
        self.emit(f"mul.wide.u32 {byte_offset}, {element_index}, {itemsize};")
        self.emit(f"add.u64 {address}, {pointers[buffer_name]}, {byte_offset};")
        return address

    def statement(self, item, pointers):
        if isinstance(item, Let):
            if item.dtype is None or item.dtype.name in (
                    "i16", "i32", "i64", "u8", "u16", "u32", "u64"):
                value = self.index(item.value)
                self.integers[item.name] = value
                if item.dtype is not None and item.dtype.name in _INTEGER_DTYPES:
                    self.values[item.name] = value
                    self.value_dtypes[item.name] = item.dtype.name
            elif item.dtype.name in ("f16", "f32", "f64") and isinstance(item.value, Literal):
                physical_dtype = "f32" if item.dtype.name == "f16" else item.dtype.name
                result = self.register("f" if physical_dtype == "f32" else "d")
                literal = (self.f32_literal(item.value.value) if physical_dtype == "f32"
                           else self.f64_literal(item.value.value))
                self.emit(f"mov.{physical_dtype} {result}, {literal};")
                if item.dtype.name == "f16":
                    result = self.round_f16(result)
                self.values[item.name] = result
                self.value_dtypes[item.name] = item.dtype.name
            elif (item.dtype.name in _DTYPE_BYTES and isinstance(item.value, Var)
                  and item.value.name in self.values):
                source_dtype = self.value_dtypes[item.value.name]
                self.values[item.name] = self.convert_value(
                    self.values[item.value.name], source_dtype, item.dtype.name)
                self.value_dtypes[item.name] = item.dtype.name
            else:
                raise NotImplementedError("Pointwise binding expression")
        elif isinstance(item, Load):
            dtype = item.dtype.name
            if dtype not in _DTYPE_BYTES:
                raise NotImplementedError(f"Pointwise load dtype {dtype}")
            address = self.address(item.input, item.index, pointers)
            if dtype == "f32":
                registers = [self.register("f") for _ in item.names]
            elif dtype == "f64":
                registers = [self.register("d") for _ in item.names]
            elif dtype == "i32":
                registers = [self.register("r") for _ in item.names]
            elif dtype == "i64":
                registers = [self.register("rd") for _ in item.names]
            else:
                registers = [self.register("h") for _ in item.names]
            if dtype == "f32" and len(registers) in (2, 4):
                operands = "{" + ", ".join(registers) + "}"
                self.emit(f"ld.global.v{len(registers)}.f32 {operands}, [{address}];")
            else:
                for offset, register in enumerate(registers):
                    suffix = {"f16": "u16", "f32": "f32", "f64": "f64",
                              "i32": "s32", "i64": "s64"}[dtype]
                    self.emit(f"ld.global.{suffix} {register}, [{address}+{offset * _DTYPE_BYTES[dtype]}];")
            for name, register in zip(item.names, registers):
                if dtype == "f16":
                    converted = self.register("f")
                    self.emit(f"cvt.f32.f16 {converted}, {register};")
                    register = converted
                self.values[name] = register
                self.value_dtypes[name] = dtype
        elif isinstance(item, Assign):
            args = [self.values[name] for name in item.args]
            arg_dtypes = [self.value_dtypes.get(name, "f32") for name in item.args]
            dtype = item.dtype.name if item.dtype is not None else "f32"
            self.values[item.name] = self.typed_operation(
                item.op.name, args, arg_dtypes, dtype)
            self.value_dtypes[item.name] = dtype
        elif isinstance(item, Write):
            dtype = item.value_dtype.name
            if dtype not in _DTYPE_BYTES or item.atomic:
                raise NotImplementedError(f"Pointwise write dtype/atomic {dtype}/{item.atomic}")
            address = self.address(item.output, item.index, pointers)
            registers = [self.convert_value(self.values[name], self.value_dtypes.get(name, dtype), dtype)
                         for name in item.values]
            if dtype == "f32" and len(registers) in (2, 4):
                operands = "{" + ", ".join(registers) + "}"
                self.emit(f"st.global.v{len(registers)}.f32 [{address}], {operands};")
            else:
                for offset, register in enumerate(registers):
                    suffix = {"f16": "b16", "f32": "f32", "f64": "f64",
                              "i32": "s32", "i64": "s64"}[dtype]
                    if dtype == "f16":
                        half = self.register("h")
                        self.emit(f"cvt.rn.f16.f32 {half}, {register};")
                        register = half
                    self.emit(f"st.global.{suffix} [{address}+{offset * _DTYPE_BYTES[dtype]}], {register};")
        else:
            raise NotImplementedError(f"Kernel statement {type(item).__name__}")


def emit_launch_indices(ptx, launch, need_stride=True):
    """Populate %r0..%r3 with linear tid, block id, block size and global id.

    Returns an operand containing the total number of launch threads.  The
    one-dimensional spelling is intentionally unchanged for public-kernel
    regression and the general spelling follows CUDA's x-fastest ordering.
    """
    grid = tuple(launch["grid"])
    block = tuple(launch["block"])
    if grid[1:] == (1, 1) and block[1:] == (1, 1):
        ptx.emit("mov.u32 %r0, %tid.x;")
        ptx.emit("mov.u32 %r1, %ctaid.x;")
        ptx.emit("mov.u32 %r2, %ntid.x;")
        ptx.emit("mad.lo.u32 %r3, %r1, %r2, %r0;")
        if need_stride:
            stride = ptx.register("r")
            ptx.emit(f"mov.u32 {stride}, %nctaid.x;")
            ptx.emit(f"mul.lo.u32 {stride}, {stride}, %r2;")
            return stride
        return None

    temporary = ptx.register("r")
    ptx.emit("mov.u32 %r0, %tid.x;")
    ptx.emit(f"mov.u32 {temporary}, %tid.y;")
    ptx.emit(f"mad.lo.u32 %r0, {temporary}, {block[0]}, %r0;")
    ptx.emit(f"mov.u32 {temporary}, %tid.z;")
    ptx.emit(f"mad.lo.u32 %r0, {temporary}, {block[0] * block[1]}, %r0;")
    ptx.emit("mov.u32 %r1, %ctaid.x;")
    ptx.emit(f"mov.u32 {temporary}, %ctaid.y;")
    ptx.emit(f"mad.lo.u32 %r1, {temporary}, {grid[0]}, %r1;")
    ptx.emit(f"mov.u32 {temporary}, %ctaid.z;")
    ptx.emit(f"mad.lo.u32 %r1, {temporary}, {grid[0] * grid[1]}, %r1;")
    ptx.emit(f"mov.u32 %r2, {prod(block)};")
    ptx.emit("mad.lo.u32 %r3, %r1, %r2, %r0;")
    if need_stride:
        stride = ptx.register("r")
        ptx.emit(f"mov.u32 {stride}, {prod(grid) * prod(block)};")
        return stride
    return None


_SWIZZLE_MASKS = {"B128": 0x7, "B64": 0x3, "B32": 0x1}


def emit_smem_swizzle(ptx, element_index, mode):
    """Translate Emmy's software shared-memory XOR layout.

    B32/B64/B128 all relocate 8-element chunks; they differ only in
    how many row bits participate.  A spelling such as B128@7 overrides
    the default element shift while keeping the base mode's mask.
    """
    if mode in (None, "NONE"):
        return element_index
    base, separator, override = mode.partition("@")
    if base not in _SWIZZLE_MASKS:
        raise NotImplementedError(f"Shared-memory swizzle {mode}")
    try:
        shift = int(override) if separator else 6
    except ValueError as error:
        raise NotImplementedError(f"Shared-memory swizzle {mode}") from error
    if shift < 0:
        raise NotImplementedError(f"Shared-memory swizzle {mode}")
    row_bits = ptx.register("r")
    delta = ptx.register("r")
    result = ptx.register("r")
    ptx.emit(f"shr.u32 {row_bits}, {element_index}, {shift};")
    ptx.emit(f"and.b32 {row_bits}, {row_bits}, {_SWIZZLE_MASKS[base]};")
    ptx.emit(f"shl.b32 {delta}, {row_bits}, 3;")
    ptx.emit(f"xor.b32 {result}, {element_index}, {delta};")
    return result


class ReductionBuilder(PtxBuilder):
    """Lower a Tile containing a strided f32/f16 reduction."""

    def __init__(self, shapes, buffer_dtypes=None, zero_outputs=(), shared_layout=None):
        super().__init__(shapes, buffer_dtypes)
        self.zero_outputs = set(zero_outputs)
        self.axes = {}
        self.pointers = {}
        # name -> (shape, dtype, PTX symbol, byte offset, swizzle)
        self.shared = dict(shared_layout or {})
        self.predicates = 0
        self.labels = 0

    def predicate(self):
        result = f"%p{self.predicates}"
        self.predicates += 1
        return result

    def label(self, prefix):
        result = f"{prefix}_{self.labels}"
        self.labels += 1
        return result

    @staticmethod
    def combine_instruction(operation):
        instructions = {
            "add": "add.f32",
            "sum": "add.f32",
            "multiply": "mul.f32",
            "prod": "mul.f32",
            "maximum": "max.f32",
            "amax": "max.f32",
            "minimum": "min.f32",
        }
        if operation not in instructions:
            raise NotImplementedError(f"Reduction combine {operation}")
        return instructions[operation]

    @staticmethod
    def f32_literal(value):
        bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
        return f"0f{bits:08x}"

    def index(self, expression):
        if isinstance(expression, Var):
            if expression.name in self.axes:
                return self.axes[expression.name]
            if expression.name in self.integers:
                return self.integers[expression.name]
            if expression.name in ("lane", "warp"):
                register = self.register("r")
                operation = "and.b32" if expression.name == "lane" else "shr.u32"
                argument = 31 if expression.name == "lane" else 5
                self.emit(f"{operation} {register}, %r0, {argument};")
                self.axes[expression.name] = register
                return register
            raise NotImplementedError(f"Index variable {expression.name}")
        if isinstance(expression, Literal):
            if isinstance(expression.value, bool):
                return "1" if expression.value else "0"
            if not isinstance(expression.value, int):
                raise NotImplementedError("Reduction index must be an integer")
            return str(expression.value)
        if isinstance(expression, BinaryExpr):
            left = self.index(expression.left)
            right = self.index(expression.right)
            operation = {"+": "add.u32", "-": "sub.u32", "*": "mul.lo.u32",
                         "/": "div.u32", "//": "div.u32", "%": "rem.u32"}.get(expression.op)
            if operation is None:
                raise NotImplementedError(f"Index operation {expression.op}")
            result = self.register("r")
            self.emit(f"{operation} {result}, {left}, {right};")
            return result
        if isinstance(expression, TernaryExpr):
            condition = self.condition_predicate(expression.cond)
            false_label = self.label("INDEX_FALSE")
            end_label = self.label("INDEX_END")
            result = self.register("r")
            self.emit(f"@!{condition} bra {false_label};")
            true_value = self.index(expression.if_true)
            self.emit(f"mov.u32 {result}, {true_value};")
            self.emit(f"bra {end_label};")
            self.lines.append(f"{false_label}:")
            false_value = self.index(expression.if_false)
            self.emit(f"mov.u32 {result}, {false_value};")
            self.lines.append(f"{end_label}:")
            return result
        raise NotImplementedError(f"Index expression {type(expression).__name__}")

    def address(self, buffer_name, coordinates):
        shape = self.shapes[buffer_name]
        if len(coordinates) != len(shape):
            raise NotImplementedError(f"Index rank for {buffer_name}")
        if not coordinates:
            element_index = "0"
        else:
            element_index = self.index(coordinates[0])
            for dimension, coordinate in zip(shape[1:], coordinates[1:]):
                scaled = self.register("r")
                combined = self.register("r")
                self.emit(f"mul.lo.u32 {scaled}, {element_index}, {dimension};")
                self.emit(f"add.u32 {combined}, {scaled}, {self.index(coordinate)};")
                element_index = combined
        offset = self.register("rd")
        address = self.register("rd")
        dtype = self.buffer_dtypes.get(buffer_name, "f32")
        itemsize = _DTYPE_BYTES.get(dtype)
        if itemsize is None:
            raise NotImplementedError(f"Reduction buffer dtype {dtype}")
        self.emit(f"mul.wide.u32 {offset}, {element_index}, {itemsize};")
        self.emit(f"add.u64 {address}, {self.pointers[buffer_name]}, {offset};")
        return address

    def shared_base(self, buffer_name):
        if buffer_name not in self.shared:
            raise NotImplementedError(f"Unknown shared buffer {buffer_name}")
        _, _, symbol, byte_offset, _ = self.shared[buffer_name]
        base = self.register("r")
        self.emit(f"mov.u32 {base}, {symbol};")
        if byte_offset:
            adjusted = self.register("r")
            self.emit(f"add.u32 {adjusted}, {base}, {byte_offset};")
            return adjusted
        return base

    def shared_address(self, buffer_name, coordinates):
        if buffer_name not in self.shared:
            raise NotImplementedError(f"Unknown shared buffer {buffer_name}")
        shape, dtype, _, _, swizzle = self.shared[buffer_name]
        if dtype not in ("f16", "f32") or len(coordinates) != len(shape):
            raise NotImplementedError("Shared-memory dtype or index rank")
        element = "0" if not coordinates else self.index(coordinates[0])
        for dimension, coordinate in zip(shape[1:], coordinates[1:]):
            scaled = self.register("r")
            combined = self.register("r")
            self.emit(f"mul.lo.u32 {scaled}, {element}, {dimension};")
            self.emit(f"add.u32 {combined}, {scaled}, {self.index(coordinate)};")
            element = combined
        element = emit_smem_swizzle(self, element, swizzle)
        base = self.shared_base(buffer_name)
        byte_offset = self.register("r")
        address = self.register("r")
        self.emit(f"mul.lo.u32 {byte_offset}, {element}, {_DTYPE_BYTES[dtype]};")
        self.emit(f"add.u32 {address}, {base}, {byte_offset};")
        return address

    def load(self, item):
        dtype = item.dtype.name
        if dtype not in ("f16", "f32"):
            raise NotImplementedError(f"Reduction load dtype {dtype}")
        shared = item.input in self.shared
        address = (self.shared_address(item.input, item.index) if shared
                   else self.address(item.input, item.index))
        for position, name in enumerate(item.names):
            result = self.register("f" if dtype == "f32" else "h")
            suffix = f"+{position * _DTYPE_BYTES[dtype]}" if position else ""
            space = "shared" if shared else "global"
            load_suffix = "f32" if dtype == "f32" else "u16"
            self.emit(f"ld.{space}.{load_suffix} {result}, [{address}{suffix}];")
            if dtype == "f16":
                converted = self.register("f")
                self.emit(f"cvt.f32.f16 {converted}, {result};")
                result = converted
            self.values[name] = result
            self.value_dtypes[name] = dtype

    def write(self, item):
        dtype = item.value_dtype.name
        if dtype not in ("f16", "f32"):
            raise NotImplementedError(f"Reduction write dtype {dtype}")
        if item.output in self.shared:
            if item.atomic or item.swizzle != self.shared[item.output][4]:
                raise NotImplementedError("Shared-memory reduction write layout")
            address = self.shared_address(item.output, item.index)
            for position, name in enumerate(item.values):
                suffix = f"+{position * _DTYPE_BYTES[dtype]}" if position else ""
                value = self.convert_value(
                    self.values[name], self.value_dtypes.get(name, dtype), dtype)
                store_suffix = "f32"
                if dtype == "f16":
                    half = self.register("h")
                    self.emit(f"cvt.rn.f16.f32 {half}, {value};")
                    value = half
                    store_suffix = "b16"
                self.emit(f"st.shared.{store_suffix} [{address}{suffix}], {value};")
        else:
            address = self.address(item.output, item.index)
            if item.atomic:
                if (dtype != "f32" or len(item.values) != 1
                        or item.output not in self.zero_outputs
                        or item.swizzle != "NONE"):
                    raise NotImplementedError("Atomic reduction supports zero-initialized f32 outputs")
                self.emit(f"red.global.add.f32 [{address}], {self.values[item.values[0]]};")
            else:
                for position, name in enumerate(item.values):
                    suffix = f"+{position * _DTYPE_BYTES[dtype]}" if position else ""
                    value = self.convert_value(
                        self.values[name], self.value_dtypes.get(name, dtype), dtype)
                    store_suffix = "f32"
                    if dtype == "f16":
                        half = self.register("h")
                        self.emit(f"cvt.rn.f16.f32 {half}, {value};")
                        value = half
                        store_suffix = "b16"
                    self.emit(f"st.global.{store_suffix} [{address}{suffix}], {value};")

    def assign(self, item):
        args = [self.values[name] for name in item.args]
        arg_dtypes = [self.value_dtypes.get(name, "f32") for name in item.args]
        dtype = item.dtype.name if item.dtype is not None else "f32"
        if dtype not in ("f16", "f32"):
            raise NotImplementedError(f"Reduction assignment dtype {dtype}")
        self.values[item.name] = self.typed_operation(item.op.name, args, arg_dtypes, dtype)
        self.value_dtypes[item.name] = dtype

    def bind(self, item):
        if item.dtype is None or item.dtype.name in (
                "i16", "i32", "i64", "u8", "u16", "u32", "u64"):
            self.integers[item.name] = self.index(item.value)
        elif item.dtype.name in ("f16", "f32") and isinstance(item.value, Literal):
            result = self.register("f")
            self.emit(f"mov.f32 {result}, {self.f32_literal(item.value.value)};")
            if item.dtype.name == "f16":
                result = self.round_f16(result)
            self.values[item.name] = result
            self.value_dtypes[item.name] = item.dtype.name
        elif (item.dtype.name in ("f16", "f32") and isinstance(item.value, Var)
              and item.value.name in self.values):
            source_dtype = self.value_dtypes.get(item.value.name, "f32")
            self.values[item.name] = self.convert_value(
                self.values[item.value.name], source_dtype, item.dtype.name)
            self.value_dtypes[item.name] = item.dtype.name
        else:
            raise NotImplementedError("Reduction binding expression")

    def initialize(self, item):
        if item.dtype.name not in ("f16", "f32"):
            raise NotImplementedError(f"Reduction Init dtype {item.dtype.name}")
        register = self.values.get(item.name)
        if register is None:
            register = self.register("f")
            self.values[item.name] = register
        identity = self.f32_literal(item.identity)
        self.emit(f"mov.f32 {register}, {identity};")
        if item.dtype.name == "f16":
            rounded = self.round_f16(register)
            self.values[item.name] = rounded
        self.value_dtypes[item.name] = item.dtype.name

    @staticmethod
    def loop_accumulators(statements):
        """Yield accumulators owned by this loop, including conditional ones.

        Do not descend into a nested loop: its reduction states belong to that
        loop's own seed boundary.
        """
        for statement in statements:
            if isinstance(statement, Accum):
                yield statement
            elif isinstance(statement, Cond):
                yield from ReductionBuilder.loop_accumulators(statement.body)
                yield from ReductionBuilder.loop_accumulators(statement.else_body)

    def accumulate(self, item):
        if item.dtype.name not in ("f16", "f32") or not item.op.has_identity:
            raise NotImplementedError("Reduction accumulator operation")
        if item.name not in self.values:
            raise NotImplementedError("Reduction accumulator is not initialized")
        accumulator = self.values[item.name]
        if item.base is None or item.base == item.name:
            left = accumulator
        elif item.base in self.values:
            left = self.values[item.base]
        else:
            raise NotImplementedError("Reduction accumulator base is undefined")
        value = self.values[item.value]
        if item.dtype.name == "f32":
            operation = self.combine_instruction(item.op.name)
            self.emit(f"{operation} {accumulator}, {left}, {value};")
            self.value_dtypes[item.name] = "f32"
            return
        result = self.typed_operation(
            item.op.name, [left, value],
            [self.value_dtypes.get(item.base or item.name, item.dtype.name),
             self.value_dtypes.get(item.value, item.dtype.name)], item.dtype.name)
        self.values[item.name] = result
        self.value_dtypes[item.name] = item.dtype.name

    def loop(self, item):
        counter = self.register("r")
        is_strided = isinstance(item, StridedLoop)
        start_expression = item.start if is_strided else Literal(0, "int")
        step_expression = item.step if is_strided else Literal(1, "int")
        end_expression = item.end if is_strided else None
        self.emit(f"mov.u32 {counter}, {self.index(start_expression)};")
        self.axes[item.axis.name] = counter
        accumulator_specs = {}
        for statement in self.loop_accumulators(item.body):
            existing = accumulator_specs.get(statement.name)
            specification = (statement.dtype.name, statement.op.name,
                             statement.op.identity)
            if existing is not None and existing != specification:
                raise NotImplementedError("Conflicting accumulator definitions in one loop")
            accumulator_specs.setdefault(statement.name, specification)
        for name, (dtype_name, operation_name, identity) in accumulator_specs.items():
            if dtype_name not in ("f16", "f32"):
                raise NotImplementedError("Reduction accumulator operation")
            self.combine_instruction(operation_name)
            accumulator = self.values.get(name)
            if accumulator is None:
                if not item.seed:
                    raise NotImplementedError("Unseeded reduction state is undefined")
                accumulator = self.register("f")
                self.values[name] = accumulator
            if item.seed:
                self.emit(f"mov.f32 {accumulator}, {self.f32_literal(identity)};")
                if dtype_name == "f16":
                    accumulator = self.round_f16(accumulator)
                    self.values[name] = accumulator
            self.value_dtypes[name] = dtype_name
        loop_start = self.label("LOOP")
        loop_end = self.label("LOOP_END")
        self.lines.append(f"{loop_start}:")
        predicate = self.predicate()
        end = (self.index(end_expression) if end_expression is not None
               else str(item.axis.extent.as_static()))
        self.emit(f"setp.ge.u32 {predicate}, {counter}, {end};")
        self.emit(f"@{predicate} bra {loop_end};")
        for statement in item.body:
            self.statement(statement)
        self.emit(f"add.u32 {counter}, {counter}, {self.index(step_expression)};")
        self.emit(f"bra {loop_start};")
        self.lines.append(f"{loop_end}:")

    def shuffle(self, item):
        if (item.dtype.name not in ("f16", "f32") or not item.state
                or len(item.state) != len(item.state_b)
                or not item.combine_states
                or any(not isinstance(statement, Assign)
                       for statement in item.combine_states)
                or item.length < 1 or item.length > 32
                or item.length & (item.length - 1)):
            raise NotImplementedError("Warp shuffle combine")
        if any(name not in self.values for name in item.state):
            raise NotImplementedError("Warp shuffle state is undefined")
        active = self.register("r")
        self.emit(f"activemask.b32 {active};")
        distance = item.length // 2
        while distance:
            for state, other_name in zip(item.state, item.state_b):
                source_bits = self.register("r")
                other_bits = self.register("r")
                other = self.register("f")
                self.emit(f"mov.b32 {source_bits}, {self.values[state]};")
                self.emit(
                    f"shfl.sync.bfly.b32 {other_bits}, {source_bits}, {distance}, 0x1f, {active};")
                self.emit(f"mov.b32 {other}, {other_bits};")
                self.values[other_name] = other
                self.value_dtypes[other_name] = item.dtype.name
            for statement in item.combine_states:
                self.assign(statement)
            distance //= 2

    def sync(self, item):
        if item.warp:
            if item.barrier_id != 0 or item.count is not None:
                raise NotImplementedError("Warp sync cannot use a named barrier")
            active = self.register("r")
            self.emit(f"activemask.b32 {active};")
            self.emit(f"bar.warp.sync {active};")
        elif item.barrier_id == 0:
            self.emit("bar.sync 0;")
        elif item.count is not None and 0 < item.barrier_id < 16 and item.count > 0:
            self.emit(f"bar.sync {item.barrier_id}, {item.count};")
        else:
            raise NotImplementedError("Named barrier requires an id and thread count")

    def tree(self, item):
        if (item.dtype.name != "f32" or not item.state
                or len(item.bufs) != len(item.state)
                or len(item.state_b) != len(item.state)
                or not item.combine_states
                or any(not isinstance(statement, Assign)
                       for statement in item.combine_states)
                or item.length < 1 or item.length & (item.length - 1)
                or any(name not in self.shared for name in item.bufs)):
            raise NotImplementedError("Tree combine operation")
        tid = self.index(Var(item.tid_var))
        if item.inner is None:
            slot = tid
            root = "0"
            scale = 1
        else:
            inner_name, scale = item.inner
            if not isinstance(scale, int) or scale <= 0:
                raise NotImplementedError("Tree inner stride")
            inner = self.index(Var(inner_name))
            slot = self.register("r")
            self.emit(f"mad.lo.u32 {slot}, {tid}, {scale}, {inner};")
            root = inner
        bases = {}
        for buffer_name in item.bufs:
            bases[buffer_name] = self.shared_base(buffer_name)

        distance = item.length // 2
        while distance:
            skip = self.label("TREE_SKIP")
            inactive = self.predicate()
            self.emit(f"setp.ge.u32 {inactive}, {tid}, {distance};")
            self.emit(f"@{inactive} bra {skip};")
            other_slot = self.register("r")
            self.emit(f"add.u32 {other_slot}, {slot}, {distance * scale};")
            for buffer_name, state, other_name in zip(
                    item.bufs, item.state, item.state_b):
                left_offset = self.register("r")
                left_address = self.register("r")
                right_offset = self.register("r")
                right_address = self.register("r")
                left = self.register("f")
                right = self.register("f")
                self.emit(f"mul.lo.u32 {left_offset}, {slot}, 4;")
                self.emit(f"add.u32 {left_address}, {bases[buffer_name]}, {left_offset};")
                self.emit(f"mul.lo.u32 {right_offset}, {other_slot}, 4;")
                self.emit(f"add.u32 {right_address}, {bases[buffer_name]}, {right_offset};")
                self.emit(f"ld.shared.f32 {left}, [{left_address}];")
                self.emit(f"ld.shared.f32 {right}, [{right_address}];")
                self.values[state] = left
                self.values[other_name] = right
            for statement in item.combine_states:
                self.assign(statement)
            for buffer_name, state in zip(item.bufs, item.state):
                offset = self.register("r")
                address = self.register("r")
                self.emit(f"mul.lo.u32 {offset}, {slot}, 4;")
                self.emit(f"add.u32 {address}, {bases[buffer_name]}, {offset};")
                self.emit(f"st.shared.f32 [{address}], {self.values[state]};")
            self.lines.append(f"{skip}:")
            self.sync(Sync(barrier_id=item.barrier_id,
                           count=item.barrier_count, warp=False))
            distance //= 2

        for buffer_name, state in zip(item.bufs, item.state):
            offset = self.register("r")
            address = self.register("r")
            result = self.register("f")
            self.emit(f"mul.lo.u32 {offset}, {root}, 4;")
            self.emit(f"add.u32 {address}, {bases[buffer_name]}, {offset};")
            self.emit(f"ld.shared.f32 {result}, [{address}];")
            self.values[state] = result

    def condition_predicate(self, expression):
        return super().condition_predicate(expression)

    def condition(self, item):
        predicate = self.condition_predicate(item.cond)
        after = self.label("AFTER")
        alternative = self.label("ELSE") if item.else_body else after
        saved_values = self.values.copy()
        saved_value_dtypes = self.value_dtypes.copy()
        saved_integers = self.integers.copy()
        self.emit(f"@!{predicate} bra {alternative};")
        for statement in item.body:
            self.statement(statement)
        if item.else_body:
            self.emit(f"bra {after};")
            self.lines.append(f"{alternative}:")
            self.values = saved_values.copy()
            self.value_dtypes = saved_value_dtypes.copy()
            self.integers = saved_integers.copy()
            for statement in item.else_body:
                self.statement(statement)
        self.values = saved_values
        self.value_dtypes = saved_value_dtypes
        self.integers = saved_integers
        self.lines.append(f"{after}:")

    def statement(self, item):
        if type(item).__name__ == "Init":
            self.initialize(item)
        elif isinstance(item, StridedLoop) or type(item).__name__ == "Loop":
            self.loop(item)
        elif isinstance(item, Accum):
            self.accumulate(item)
        elif isinstance(item, WarpShuffle):
            self.shuffle(item)
        elif isinstance(item, Smem):
            if item.name not in self.shared:
                raise NotImplementedError(f"Unplanned shared buffer {item.name}")
        elif isinstance(item, Cond):
            self.condition(item)
        elif isinstance(item, Sync):
            self.sync(item)
        elif isinstance(item, TreeHalve):
            self.tree(item)
        elif isinstance(item, Load):
            self.load(item)
        elif isinstance(item, Assign):
            self.assign(item)
        elif isinstance(item, Let):
            self.bind(item)
        elif isinstance(item, Write):
            self.write(item)
        else:
            raise NotImplementedError(f"Reduction statement {type(item).__name__}")


def compile_reduction(kernel, tile, arg_order, graph, launch):
    shared_dtype_names = {"float": "f32", "__half": "f16", "half": "f16"}
    def walk(statements):
        for statement in statements:
            yield statement
            if hasattr(statement, "body"):
                yield from walk(statement.body)
            if isinstance(statement, Cond):
                yield from walk(statement.else_body)

    shapes = {
        tensor.name: tuple(dim.as_static() for dim in tensor.shape)
        for node in graph.nodes.values()
        for tensor in node.outputs
    }
    buffer_dtypes = {
        tensor.name: tensor.dtype.name
        for node in graph.nodes.values()
        for tensor in node.outputs
    }
    zero_outputs = set(launch["zero_outputs"]) | set(kernel.zero_delegated)
    declarations = []
    declaration_names = set()
    for statement in walk(tile.body):
        if not isinstance(statement, Smem):
            continue
        if statement.name in declaration_names:
            raise NotImplementedError(f"Duplicate shared buffer {statement.name}")
        if statement.dtype not in shared_dtype_names or statement.align < 0:
            raise NotImplementedError("Reduction shared-memory type or alignment")
        declaration_names.add(statement.name)
        declarations.append(statement)

    shared_swizzles = {name: "NONE" for name in declaration_names}
    for statement in walk(tile.body):
        if not isinstance(statement, Write) or statement.output not in declaration_names:
            continue
        previous = shared_swizzles[statement.output]
        if previous != "NONE" and previous != statement.swizzle:
            raise NotImplementedError("Inconsistent reduction shared-memory swizzle")
        shared_swizzles[statement.output] = statement.swizzle

    dynamic_bytes = launch.get("smem_bytes", 0)
    shared_layout = {}
    shared_declarations = []
    if dynamic_bytes:
        total_bytes = 0
        maximum_alignment = 4
        offsets = {}
        for declaration in declarations:
            dtype = shared_dtype_names[declaration.dtype]
            itemsize = _DTYPE_BYTES[dtype]
            alignment = declaration.align or itemsize
            total_bytes = ((total_bytes + alignment - 1) // alignment) * alignment
            offsets[declaration.name] = total_bytes
            total_bytes += prod(declaration.extents) * itemsize
            maximum_alignment = max(maximum_alignment, alignment)
        if dynamic_bytes < total_bytes:
            raise NotImplementedError(
                "Launch dynamic shared memory is smaller than the reduction IR layout")
        if declarations:
            shared_declarations.append(
                f".extern .shared .align {maximum_alignment} .b8 __emmy_dynamic_smem[];")
        for declaration in declarations:
            dtype = shared_dtype_names[declaration.dtype]
            shared_layout[declaration.name] = (
                tuple(declaration.extents), dtype, "__emmy_dynamic_smem",
                offsets[declaration.name], shared_swizzles[declaration.name])
    else:
        for declaration in declarations:
            dtype = shared_dtype_names[declaration.dtype]
            itemsize = _DTYPE_BYTES[dtype]
            size = prod(declaration.extents) * itemsize
            alignment = declaration.align or itemsize
            shared_declarations.append(
                f".shared .align {alignment} .b8 {declaration.name}[{size}];")
            shared_layout[declaration.name] = (
                tuple(declaration.extents), dtype, declaration.name, 0,
                shared_swizzles[declaration.name])

    ptx = ReductionBuilder(shapes, buffer_dtypes, zero_outputs, shared_layout)
    for name in arg_order:
        pointer = ptx.register("rd")
        ptx.pointers[name] = pointer
        ptx.emit(f"ld.param.u64 {pointer}, [param_{name}];")
        ptx.emit(f"cvta.to.global.u64 {pointer}, {pointer};")
    emit_launch_indices(ptx, launch, need_stride=False)
    outside = ptx.predicate()
    ptx.emit(f"setp.ge.u32 {outside}, %r3, {tile.n_elements};")
    ptx.emit(f"@{outside} bra DONE;")
    for position, (axis, extent) in enumerate(zip(tile.axes, tile.extents)):
        stride = prod(tile.extents[position + 1:])
        coordinate = "%r3"
        if stride > 1:
            coordinate = ptx.register("r")
            ptx.emit(f"div.u32 {coordinate}, %r3, {stride};")
        if position > 0:
            reduced = ptx.register("r")
            ptx.emit(f"rem.u32 {reduced}, {coordinate}, {extent};")
            coordinate = reduced
        ptx.axes[axis.name] = coordinate
    for statement in tile.body:
        ptx.statement(statement)
    signature = ",\n".join(f"    .param .u64 param_{name}" for name in arg_order)
    return [
        ".version 8.0", ".target sm_89", ".address_size 64", "",
        *shared_declarations, "",
        f".visible .entry {kernel.name}(\n{signature}\n)", "{",
        f"    .reg .b32 %r<{ptx.counts['r']}>;",
        f"    .reg .b64 %rd<{ptx.counts['rd']}>;",
        *([f"    .reg .b16 %h<{ptx.counts['h']}>;"] if ptx.counts["h"] else []),
        f"    .reg .f32 %f<{ptx.counts['f']}>;",
        f"    .reg .pred %p<{ptx.predicates}>;",
        *ptx.lines, "DONE:", "    ret;", "}",
    ]



class FusedRowBuilder(ReductionBuilder):
    """Evaluate each whole row in one thread, eliminating cross-thread folds."""

    def load(self, item):
        if item.input not in self.pointers:
            raise NotImplementedError("Fused row input buffer")
        address = self.address(item.input, item.index)
        for position, name in enumerate(item.names):
            result = self.register("f")
            suffix = f"+{position * 4}" if position else ""
            self.emit(f"ld.global.f32 {result}, [{address}{suffix}];")
            self.values[name] = result

    def assign(self, item):
        if item.dtype is not None and item.dtype.name != "f32":
            raise NotImplementedError("Fused row assignment requires f32")
        args = [self.values[name] for name in item.args]
        self.values[item.name] = self.f32_operation(item.op.name, args)

    def scalar_statement(self, item):
        if isinstance(item, Load):
            self.load(item)
        elif isinstance(item, Let):
            self.bind(item)
        elif isinstance(item, Assign):
            self.assign(item)
        elif isinstance(item, Accum):
            if item.name not in self.values or item.op.name not in (
                    "add", "maximum", "minimum", "multiply"):
                raise NotImplementedError("Fused row accumulator")
            left = self.values[item.base] if item.base is not None else self.values[item.name]
            right = self.values[item.value]
            operation = self.combine_instruction(item.op.name)
            self.emit(f"{operation} {self.values[item.name]}, {left}, {right};")
        elif isinstance(item, Write) and not item.atomic:
            address = self.address(item.output, item.index)
            for position, name in enumerate(item.values):
                suffix = f"+{position * 4}" if position else ""
                self.emit(f"st.global.f32 [{address}{suffix}], {self.values[name]};")
        else:
            raise NotImplementedError(f"Fused row statement {type(item).__name__}")

    def row_loop(self, item, end_value, start_value="0", step=1):
        counter = self.register("r")
        self.axes[item.axis.name] = counter
        self.emit(f"mov.u32 {counter}, {start_value};")
        start = self.label("ROW_COL")
        end = self.label("ROW_END")
        predicate = self.predicate()
        self.lines.append(f"{start}:")
        self.emit(f"setp.ge.u32 {predicate}, {counter}, {end_value};")
        self.emit(f"@{predicate} bra {end};")
        for statement in item.body:
            self.scalar_statement(statement)
        self.emit(f"add.u32 {counter}, {counter}, {step};")
        self.emit(f"bra {start};")
        self.lines.append(f"{end}:")

    def warp_combine(self, item):
        """Apply the IR's associative fold to the 32 partial row states."""
        if item.length != 32 or len(item.state) != len(item.state_b):
            raise NotImplementedError("Fused row warp state")
        for distance in (16, 8, 4, 2, 1):
            for state, other_name in zip(item.state, item.state_b):
                bits = self.register("r")
                other_bits = self.register("r")
                other = self.register("f")
                self.emit(f"mov.b32 {bits}, {self.values[state]};")
                self.emit(f"shfl.sync.bfly.b32 {other_bits}, {bits}, {distance}, 0x1f, 0xffffffff;")
                self.emit(f"mov.b32 {other}, {other_bits};")
                self.values[other_name] = other
            for combine in item.combine_states:
                if not isinstance(combine, Assign):
                    raise NotImplementedError("Fused row warp combine statement")
                self.assign(combine)


class StructuredLoopBuilder(FusedRowBuilder):
    """Lower scalar Loop IR by executing its statements in their original order.

    Unlike the category-specific attention recognizer, this builder does not
    reconstruct a softmax formula.  Accumulator scope comes from ``Accum.axes``:
    a state is initialized immediately before the first loop axis it reduces.
    """

    def __init__(self, shapes, buffer_dtypes=None, zero_outputs=()):
        super().__init__(shapes, buffer_dtypes, zero_outputs)
        self.explicit_inits = set()

    @staticmethod
    def nested_accumulators(statements):
        for statement in statements:
            if isinstance(statement, Accum):
                yield statement
            if hasattr(statement, "body"):
                yield from StructuredLoopBuilder.nested_accumulators(statement.body)
            if isinstance(statement, Cond):
                yield from StructuredLoopBuilder.nested_accumulators(statement.else_body)

    def initialize_axis_accumulators(self, axis_name, statements):
        initialized = set()
        for accumulator in self.nested_accumulators(statements):
            if (not accumulator.axes or accumulator.axes[0] != axis_name
                    or accumulator.name in initialized
                    or accumulator.name in self.explicit_inits):
                continue
            if (accumulator.dtype is not None and accumulator.dtype.name != "f32"
                    or not accumulator.op.has_identity):
                raise NotImplementedError("Structured accumulator type or identity")
            self.combine_instruction(accumulator.op.name)
            register = self.values.get(accumulator.name)
            if register is None:
                register = self.register("f")
                self.values[accumulator.name] = register
            self.emit(f"mov.f32 {register}, {self.f32_literal(accumulator.op.identity)};")
            initialized.add(accumulator.name)

    def initialize(self, item):
        if item.dtype.name != "f32":
            raise NotImplementedError("Structured Init requires f32")
        register = self.values.get(item.name)
        if register is None:
            register = self.register("f")
            self.values[item.name] = register
        self.emit(f"mov.f32 {register}, {self.f32_literal(item.identity)};")
        self.explicit_inits.add(item.name)

    def accumulate(self, item):
        if item.name not in self.values or item.value not in self.values:
            raise NotImplementedError("Structured accumulator value is undefined")
        if item.dtype is not None and item.dtype.name != "f32":
            raise NotImplementedError("Structured accumulator requires f32")
        left = self.values[item.name]
        if item.base is not None:
            if item.base not in self.values:
                raise NotImplementedError("Structured accumulator base is undefined")
            left = self.values[item.base]
        operation = self.combine_instruction(item.op.name)
        self.emit(
            f"{operation} {self.values[item.name]}, {left}, {self.values[item.value]};")

    def regular_loop(self, item):
        if item.seed:
            self.initialize_axis_accumulators(item.axis.name, item.body)
        counter = self.register("r")
        previous_axis = self.axes.get(item.axis.name)
        self.axes[item.axis.name] = counter
        self.emit(f"mov.u32 {counter}, 0;")
        start = self.label("LOOP")
        end = self.label("LOOP_END")
        predicate = self.predicate()
        self.lines.append(f"{start}:")
        self.emit(f"setp.ge.u32 {predicate}, {counter}, {item.axis.extent.as_static()};")
        self.emit(f"@{predicate} bra {end};")
        for statement in item.body:
            self.statement(statement)
        self.emit(f"add.u32 {counter}, {counter}, 1;")
        self.emit(f"bra {start};")
        self.lines.append(f"{end}:")
        if previous_axis is None:
            del self.axes[item.axis.name]
        else:
            self.axes[item.axis.name] = previous_axis

    def strided_loop(self, item):
        if item.seed:
            self.initialize_axis_accumulators(item.axis.name, item.body)
        counter = self.register("r")
        previous_axis = self.axes.get(item.axis.name)
        start_value = self.index(item.start)
        step_value = self.index(item.step)
        end_value = (self.index(item.end) if item.end is not None
                     else str(item.axis.extent.as_static()))
        self.axes[item.axis.name] = counter
        self.emit(f"mov.u32 {counter}, {start_value};")
        start = self.label("STRIDED_LOOP")
        end = self.label("STRIDED_END")
        predicate = self.predicate()
        self.lines.append(f"{start}:")
        self.emit(f"setp.ge.u32 {predicate}, {counter}, {end_value};")
        self.emit(f"@{predicate} bra {end};")
        for statement in item.body:
            self.statement(statement)
        self.emit(f"add.u32 {counter}, {counter}, {step_value};")
        self.emit(f"bra {start};")
        self.lines.append(f"{end}:")
        if previous_axis is None:
            del self.axes[item.axis.name]
        else:
            self.axes[item.axis.name] = previous_axis

    def statement(self, item):
        if type(item).__name__ == "Init":
            self.initialize(item)
        elif type(item).__name__ == "Loop":
            self.regular_loop(item)
        elif isinstance(item, StridedLoop):
            self.strided_loop(item)
        elif isinstance(item, Load):
            self.load(item)
        elif isinstance(item, Let):
            self.bind(item)
        elif isinstance(item, Assign):
            self.assign(item)
        elif isinstance(item, Accum):
            self.accumulate(item)
        elif isinstance(item, Cond):
            self.condition(item)
        elif isinstance(item, Write):
            self.write(item)
        else:
            raise NotImplementedError(
                f"Structured loop statement {type(item).__name__}")


def compile_structured_tile(kernel, tile, arg_order, graph, launch):
    """Translate a barrier-free scalar Tile directly instead of guessing its algorithm."""
    def walk(statements):
        for statement in statements:
            yield statement
            if hasattr(statement, "body"):
                yield from walk(statement.body)
            if isinstance(statement, Cond):
                yield from walk(statement.else_body)

    supported = {"Init", "Loop", "StridedLoop", "Load", "Let", "Assign", "Accum",
                 "Cond", "Write"}
    statements = list(walk(tile.body))
    if any(type(statement).__name__ not in supported for statement in statements):
        raise NotImplementedError("Structured Tile contains a collective or memory pipeline")
    shapes = {tensor.name: tuple(dim.as_static() for dim in tensor.shape)
              for node in graph.nodes.values() for tensor in node.outputs}
    if any(name not in shapes for statement in statements
           for name in ((statement.input,) if isinstance(statement, Load)
                        else (statement.output,) if isinstance(statement, Write) else ())):
        raise NotImplementedError("Structured Tile references an unknown buffer")
    if any(tensor.dtype.name != "f32" for node in graph.nodes.values()
           for tensor in node.outputs if tensor.name in arg_order):
        raise NotImplementedError("Structured Tile currently supports f32 buffers")

    zero_outputs = set(launch["zero_outputs"]) | set(kernel.zero_delegated)
    ptx = StructuredLoopBuilder(shapes, zero_outputs=zero_outputs)
    ptx.predicates = 1  # %p0 is the outer Tile bounds predicate.
    for name in arg_order:
        pointer = ptx.register("rd")
        ptx.pointers[name] = pointer
        ptx.emit(f"ld.param.u64 {pointer}, [param_{name}];")
        ptx.emit(f"cvta.to.global.u64 {pointer}, {pointer};")
    tile_stride = emit_launch_indices(ptx, launch)
    ptx.lines.append("TILE_LOOP:")
    ptx.emit(f"setp.ge.u32 %p0, %r3, {tile.n_elements};")
    ptx.emit("@%p0 bra DONE;")

    for position, (axis, extent) in enumerate(zip(tile.axes, tile.extents)):
        inner_stride = prod(tile.extents[position + 1:])
        coordinate = "%r3"
        if inner_stride > 1:
            coordinate = ptx.register("r")
            ptx.emit(f"div.u32 {coordinate}, %r3, {inner_stride};")
        if position > 0:
            reduced = ptx.register("r")
            ptx.emit(f"rem.u32 {reduced}, {coordinate}, {extent};")
            coordinate = reduced
        ptx.axes[axis.name] = coordinate
    for statement in tile.body:
        ptx.statement(statement)
    ptx.emit(f"add.u32 %r3, %r3, {tile_stride};")
    ptx.emit("bra TILE_LOOP;")

    signature = ",\n".join(f"    .param .u64 param_{name}" for name in arg_order)
    return [
        ".version 8.0", ".target sm_89", ".address_size 64", "",
        f".visible .entry {kernel.name}(\n{signature}\n)", "{",
        f"    .reg .b32 %r<{ptx.counts['r']}>;",
        f"    .reg .b64 %rd<{ptx.counts['rd']}>;",
        f"    .reg .f32 %f<{ptx.counts['f']}>;",
        f"    .reg .pred %p<{ptx.predicates}>;",
        *ptx.lines, "DONE:", "    ret;", "}",
    ]


class MatrixScheduleBuilder(StructuredLoopBuilder):
    """Translate Emmy's scalar cp.async/shared-memory schedule statement by statement."""

    def __init__(self, shapes, zero_outputs, shared_layout):
        super().__init__(shapes, zero_outputs=zero_outputs)
        self.matrix_shared = shared_layout

    def initialize_axis_accumulators(self, axis_name, statements):
        # Matmul loop-carried states are declared explicitly by Init outside
        # the pipeline loops, so entering a nested loop must not reset them.
        return

    def shared_address(self, buffer_name, coordinates):
        if buffer_name not in self.matrix_shared:
            raise NotImplementedError("Unknown matrix shared buffer")
        shape, symbol, base_offset, swizzle = self.matrix_shared[buffer_name]
        if len(coordinates) != len(shape):
            raise NotImplementedError("Matrix shared-memory index rank")
        element = "0" if not coordinates else self.index(coordinates[0])
        for dimension, coordinate in zip(shape[1:], coordinates[1:]):
            scaled = self.register("r")
            combined = self.register("r")
            self.emit(f"mul.lo.u32 {scaled}, {element}, {dimension};")
            self.emit(f"add.u32 {combined}, {scaled}, {self.index(coordinate)};")
            element = combined
        element = emit_smem_swizzle(self, element, swizzle)
        byte_offset = self.register("r")
        base = self.register("r")
        address = self.register("r")
        self.emit(f"mul.lo.u32 {byte_offset}, {element}, 4;")
        self.emit(f"mov.u32 {base}, {symbol};")
        if base_offset:
            adjusted = self.register("r")
            self.emit(f"add.u32 {adjusted}, {base}, {base_offset};")
            base = adjusted
        self.emit(f"add.u32 {address}, {base}, {byte_offset};")
        return address

    def load(self, item):
        if item.dtype is not None and item.dtype.name != "f32":
            raise NotImplementedError("Matrix schedule load requires f32")
        if item.input not in self.matrix_shared:
            return super().load(item)
        address = self.shared_address(item.input, item.index)
        for position, name in enumerate(item.names):
            result = self.register("f")
            suffix = f"+{position * 4}" if position else ""
            self.emit(f"ld.shared.f32 {result}, [{address}{suffix}];")
            self.values[name] = result

    def initialize(self, item):
        if item.dtype.name != "f32":
            raise NotImplementedError("Matrix Init requires f32")
        register = self.values.get(item.name)
        if register is None:
            register = self.register("f")
            self.values[item.name] = register
        self.emit(f"mov.f32 {register}, {self.f32_literal(item.identity)};")

    def cp_async_copy(self, item):
        if item.nbytes not in (4, 8, 16) or item.src not in self.pointers:
            raise NotImplementedError("Unsupported cp.async copy")
        if item.smem not in self.matrix_shared:
            raise NotImplementedError("cp.async destination shared buffer")
        if item.swizzle != self.matrix_shared[item.smem][3]:
            raise NotImplementedError("Inconsistent shared-memory swizzle")
        coordinates = item.smem_index
        if item.lane_index is not None:
            if len(item.lane_index) != len(coordinates):
                raise NotImplementedError("cp.async lane index rank")
            coordinates = tuple(BinaryExpr("+", base, lane)
                                for base, lane in zip(coordinates, item.lane_index))
        shared = self.shared_address(item.smem, coordinates)
        global_address = self.address(item.src, item.src_index)
        cache = "cg" if item.nbytes == 16 else "ca"
        self.emit(
            f"cp.async.{cache}.shared.global [{shared}], [{global_address}], {item.nbytes};")

    def statement(self, item):
        kind = type(item).__name__
        if kind == "Init":
            self.initialize(item)
        elif isinstance(item, Smem):
            return  # Declarations and pool offsets are prepared before the body.
        elif kind == "CpAsyncCopy":
            self.cp_async_copy(item)
        elif kind == "CpAsyncCommit":
            self.emit("cp.async.commit_group;")
        elif kind == "CpAsyncWait":
            if not isinstance(item.group, int) or item.group < 0:
                raise NotImplementedError("cp.async wait group")
            self.emit(f"cp.async.wait_group {item.group};")
        elif isinstance(item, Sync):
            self.sync(item)
        else:
            super().statement(item)


def compile_scheduled_matmul(kernel, tile, arg_order, graph, launch):
    """Emit the supplied shared-memory pipeline instead of reconstructing a GEMM."""
    if prod(launch["grid"]) * prod(launch["block"]) != tile.n_elements:
        raise NotImplementedError(
            "Scheduled matrix Tile requires exactly one launch thread per cell")
    shapes = {tensor.name: tuple(dim.as_static() for dim in tensor.shape)
              for node in graph.nodes.values() for tensor in node.outputs}
    if any(tensor.dtype.name != "f32" for node in graph.nodes.values()
           for tensor in node.outputs if tensor.name in arg_order):
        raise NotImplementedError("Scheduled matrix buffers require f32")

    def walk(statements):
        for statement in statements:
            yield statement
            if hasattr(statement, "body"):
                yield from walk(statement.body)
            if isinstance(statement, Cond):
                yield from walk(statement.else_body)

    statements = list(walk(tile.body))
    declarations = [item for item in statements if isinstance(item, Smem)]
    if not declarations:
        raise NotImplementedError("Scheduled matrix kernel has no shared memory")
    if len({item.name for item in declarations}) != len(declarations):
        raise NotImplementedError("Duplicate matrix shared buffer")
    copy_swizzles = {}
    for statement in statements:
        if type(statement).__name__ != "CpAsyncCopy":
            continue
        previous = copy_swizzles.setdefault(statement.smem, statement.swizzle)
        if previous != statement.swizzle:
            raise NotImplementedError("Inconsistent shared-memory swizzle")
    offsets = {}
    total_bytes = 0
    maximum_alignment = 1
    for declaration in declarations:
        if declaration.dtype != "float" or declaration.align <= 0:
            raise NotImplementedError("Scheduled matrix shared-memory type")
        alignment = declaration.align
        total_bytes = ((total_bytes + alignment - 1) // alignment) * alignment
        offsets[declaration.name] = total_bytes
        total_bytes += prod(declaration.extents) * 4
        maximum_alignment = max(maximum_alignment, alignment)

    dynamic_bytes = launch.get("smem_bytes", 0)
    use_dynamic = dynamic_bytes > 0
    if use_dynamic and dynamic_bytes < total_bytes:
        raise NotImplementedError("Launch dynamic shared memory is smaller than the IR layout")
    if use_dynamic:
        shared_declarations = [
            f".extern .shared .align {maximum_alignment} .b8 __emmy_dynamic_smem[];"
        ]
        shared_layout = {
            item.name: (tuple(item.extents), "__emmy_dynamic_smem", offsets[item.name],
                        copy_swizzles.get(item.name, "NONE"))
            for item in declarations
        }
    else:
        shared_declarations = [
            f".shared .align {item.align} .b8 {item.name}[{prod(item.extents) * 4}];"
            for item in declarations
        ]
        shared_layout = {
            item.name: (tuple(item.extents), item.name, 0,
                        copy_swizzles.get(item.name, "NONE"))
            for item in declarations
        }

    zero_outputs = set(launch["zero_outputs"]) | set(kernel.zero_delegated)
    ptx = MatrixScheduleBuilder(
        shapes, zero_outputs=zero_outputs, shared_layout=shared_layout)
    for name in arg_order:
        if name not in shapes:
            raise NotImplementedError("Scheduled matrix argument shape")
        pointer = ptx.register("rd")
        ptx.pointers[name] = pointer
        ptx.emit(f"ld.param.u64 {pointer}, [param_{name}];")
        ptx.emit(f"cvta.to.global.u64 {pointer}, {pointer};")
    emit_launch_indices(ptx, launch, need_stride=False)
    for position, (axis, extent) in enumerate(zip(tile.axes, tile.extents)):
        stride = prod(tile.extents[position + 1:])
        coordinate = "%r3"
        if stride > 1:
            coordinate = ptx.register("r")
            ptx.emit(f"div.u32 {coordinate}, %r3, {stride};")
        if position > 0:
            reduced = ptx.register("r")
            ptx.emit(f"rem.u32 {reduced}, {coordinate}, {extent};")
            coordinate = reduced
        ptx.axes[axis.name] = coordinate
    for statement in tile.body:
        ptx.statement(statement)

    signature = ",\n".join(f"    .param .u64 param_{name}" for name in arg_order)
    return [
        ".version 8.0", ".target sm_89", ".address_size 64", "",
        *shared_declarations, "",
        f".visible .entry {kernel.name}(\n{signature}\n)", "{",
        f"    .reg .b32 %r<{ptx.counts['r']}>;",
        f"    .reg .b64 %rd<{ptx.counts['rd']}>;",
        f"    .reg .f32 %f<{ptx.counts['f']}>;",
        f"    .reg .pred %p<{max(1, ptx.predicates)}>;",
        *ptx.lines, "    ret;", "}",
    ]


def compile_matmul(kernel, tile, arg_order, graph, launch):
    return compile_scheduled_matmul(kernel, tile, arg_order, graph, launch)




def classify_tile(tile):
    """Choose a backend by required IR mechanisms, not by public-problem shape."""
    def walk(statements):
        for statement in statements:
            yield statement
            if hasattr(statement, "body"):
                yield from walk(statement.body)
            if isinstance(statement, Cond):
                yield from walk(statement.else_body)

    statements = list(walk(tile.body))
    kinds = {type(statement).__name__ for statement in statements}
    if kinds & {"CpAsyncCopy", "CpAsyncCommit", "CpAsyncWait"}:
        return "matmul"
    if any(isinstance(statement, (WarpShuffle, TreeHalve, Smem, Sync))
           for statement in statements):
        return "cooperative"
    if (tile.axes
            and all(isinstance(item, (Let, Load, Assign, Write))
                    for item in tile.body)
            and not any(isinstance(item, Write) and item.atomic
                        for item in tile.body)):
        return "pointwise"
    structured = {"Init", "Loop", "StridedLoop", "Load", "Let", "Assign",
                  "Accum", "Cond", "Write"}
    if tile.axes and kinds <= structured:
        return "structured"
    raise NotImplementedError("Unsupported Tile structure")


def main():
    ir_path = Path(sys.argv[1])
    with ir_path.open(encoding="utf-8") as file:
        graph = Graph.from_dict(json.load(file))
    kernels = [node.op for node in graph.nodes.values() if isinstance(node.op, KernelOp)]
    if len(kernels) != 1:
        raise ValueError(f"Expected one KernelOp, found {len(kernels)}")
    kernel = kernels[0]
    tiles = [item for item in kernel.body if isinstance(item, Tile)]
    if len(tiles) != 1 or len(kernel.body) != 1:
        raise NotImplementedError("Only a single Tile is supported")
    tile = tiles[0]

    with ir_path.with_name("meta.json").open(encoding="utf-8") as file:
        meta = json.load(file)
    matches = [item for item in meta["kernels"] if item["entry"] == kernel.name]
    if len(matches) != 1:
        raise ValueError(f"Expected one meta entry for {kernel.name}")
    launch = matches[0]
    arg_order = launch["arg_order"]
    if (len(launch["grid"]) != 3 or len(launch["block"]) != 3
            or any(dimension <= 0 for dimension in (*launch["grid"], *launch["block"]))):
        raise ValueError("Grid and block must each have three positive dimensions")
    kind = classify_tile(tile)
    if kind == "cooperative":
        launch_threads = prod(launch["grid"]) * prod(launch["block"])
        block_threads = prod(launch["block"])
        if (launch_threads < tile.n_elements
                or (launch_threads != tile.n_elements
                    and tile.n_elements % block_threads)):
            raise NotImplementedError(
                "Cooperative Tile needs enough threads and may only guard whole excess blocks")
    if kind == "structured":
        print("\n".join(compile_structured_tile(kernel, tile, arg_order, graph, launch)))
        return
    if kind == "matmul":
        print("\n".join(compile_matmul(kernel, tile, arg_order, graph, matches[0])))
        return
    if kind == "cooperative":
        print("\n".join(compile_reduction(kernel, tile, arg_order, graph, launch)))
        return

    shapes = {tensor.name: tuple(dim.as_static() for dim in tensor.shape)
              for node in graph.nodes.values() for tensor in node.outputs}
    buffer_dtypes = {tensor.name: tensor.dtype.name
                     for node in graph.nodes.values() for tensor in node.outputs}
    ptx = PtxBuilder(shapes, buffer_dtypes)
    ptx.predicates = 1  # %p0 is reserved for the outer Tile bounds guard.
    pointers = {}
    for name in arg_order:
        pointer = ptx.register("rd")
        pointers[name] = pointer
        ptx.emit(f"ld.param.u64 {pointer}, [param_{name}];")
        ptx.emit(f"cvta.to.global.u64 {pointer}, {pointer};")

    tile_stride = emit_launch_indices(ptx, launch)
    ptx.lines.append("TILE_LOOP:")
    ptx.emit(f"setp.ge.u32 %p0, %r3, {tile.n_elements};")
    ptx.emit("@%p0 bra DONE;")

    for position, (axis, extent) in enumerate(zip(tile.axes, tile.extents)):
        inner_stride = prod(tile.extents[position + 1:])
        coordinate = "%r3"
        if inner_stride > 1:
            coordinate = ptx.register("r")
            ptx.emit(f"div.u32 {coordinate}, %r3, {inner_stride};")
        if position > 0:
            reduced = ptx.register("r")
            ptx.emit(f"rem.u32 {reduced}, {coordinate}, {extent};")
            coordinate = reduced
        ptx.axes[axis.name] = coordinate
    for item in tile.body:
        ptx.statement(item, pointers)
    ptx.emit(f"add.u32 %r3, %r3, {tile_stride};")
    ptx.emit("bra TILE_LOOP;")

    parameters = [f"    .param .u64 param_{name}" for name in arg_order]
    signature = ",\n".join(parameters)
    output = [
        ".version 8.0",
        ".target sm_89",
        ".address_size 64",
        "",
        f".visible .entry {kernel.name}(\n{signature}\n)",
        "{",
        f"    .reg .b32 %r<{ptx.counts['r']}>;",
        f"    .reg .b64 %rd<{ptx.counts['rd']}>;",
        *([f"    .reg .b16 %h<{ptx.counts['h']}>;"] if ptx.counts["h"] else []),
        f"    .reg .f32 %f<{ptx.counts['f']}>;",
        *([f"    .reg .f64 %d<{ptx.counts['d']}>;"] if ptx.counts["d"] else []),
        f"    .reg .pred %p<{ptx.predicates}>;",
        *ptx.lines,
        "DONE:",
        "    ret;",
        "}",
    ]
    print("\n".join(output))


if __name__ == "__main__":
    main()
