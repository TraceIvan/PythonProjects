from cpuid import *

def _is_set(id, reg_idx, bit):
    regs = cpuid(id)

    if (1 << bit) & regs[reg_idx]:
        return "Yes"
    else:
        return "--"

print("Vendor ID         : %s" % cpu_vendor())
print("CPU name          : %s" % cpu_name())
print("Microarchitecture : %s%s" % cpu_microarchitecture())
print("Vector instructions supported:")
print("SSE       : %s" % _is_set(1, 3, 25))
print("SSE2      : %s" % _is_set(1, 3, 26))
print("SSE3      : %s" % _is_set(1, 2, 0))
print("SSSE3     : %s" % _is_set(1, 2, 9))
print("SSE4.1    : %s" % _is_set(1, 2, 19))
print("SSE4.2    : %s" % _is_set(1, 2, 20))
print("SSE4a     : %s" % _is_set(0x80000001, 2, 6))
print("AVX       : %s" % _is_set(1, 2, 28))
print("AVX2      : %s" % _is_set(7, 1, 5))
print("BMI1      : %s" % _is_set(7, 1, 3))
print("BMI2      : %s" % _is_set(7, 1, 8))