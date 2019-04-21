import numba
from numba.types import float64
from numba.extending import intrinsic
from llvmlite import ir


@intrinsic
def _fma(typing_context, x, y, z):
    """Compute x * y + z using a fused multiply add."""
    sig = float64(float64, float64, float64)

    def codegen(context, builder, signature, args):
        ty = args[0].type
        mod = builder.module
        fnty = ir.types.FunctionType(ty, [ty, ty, ty])
        fn = mod.declare_intrinsic('llvm.fma', [ty], fnty)
        ret = builder.call(fn, args)
        return ret

    return sig, codegen
