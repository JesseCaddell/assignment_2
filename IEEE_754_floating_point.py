import struct
import math
from decimal import Decimal, getcontext, ROUND_HALF_UP, ROUND_DOWN, ROUND_CEILING
import numpy as np
import matplotlib.pyplot as plt

### 1. Convert decimal to IEEE 754 format

def float_to_ieee754(num):
    #pack the float
    packed = struct.pack('>f', num)
    #unpack the bytes to an unsigned int
    integers = struct.unpack('>I', packed)[0]
    #format the int to a 32 bit
    binary_str = f'{integers:032b}'
    return binary_str

number = 0.15625
binary_repr = float_to_ieee754(number)
print('Part 1')
print(f'Decimal: {number}')
print(f'IEEE 754 32-bit binary: {binary_repr}')
print('\n----------------------------------------\n')
print('Part 2')


### 2. Arithmetic Operations

# 0.1 + 0.2
result1 = 0.1 + 0.2
binary_result1 = float_to_ieee754(result1)
print(f'0.1 + 0.2 = {result1}')
print(f'IEEE 754 binary: {binary_result1}')

# 1.0 / 3.0
result2 = 1.0 / 3.0
binary_result2 = float_to_ieee754(result2)
print(f'1.0 / 3.0 = {result2}')
print(f'IEEE 754 binary: {binary_result2}')
print('\n----------------------------------------\n')
print('Part 3')


### 3. Special Values Handling

# Generate positive infinity using arithmetic operation
pos_inf = 1e308 * 1e10  # This will cause an overflow to positive infinity
print(f'Positive infinity via overflow: {pos_inf}')

# Generate negative infinity using arithmetic operation
neg_inf = -1e308 * 1e10  # Overflow to negative infinity
print(f'Negative infinity via overflow: {neg_inf}')

# Generate NaN using arithmetic operations
nan_value = pos_inf - pos_inf  # inf - inf results in NaN
print(f'NaN via invalid operation (inf - inf): {nan_value}')

# Alternative method: 0 * infinity
nan_alternative = pos_inf * 0  # 0 * inf results in NaN
print(f'NaN via invalid operation (inf * 0): {nan_alternative}')

# Verify properties
print(f'Is NaN not equal to itself? {nan_value != nan_value}')
print(f'Is positive infinity greater than any number? {pos_inf > 1e308}')
print(f'Is negative infinity less than any number? {neg_inf < -1e308}')
print(f'Is negative infinity less than any number? {neg_inf < -1e308}')
print('\n----------------------------------------\n')
print('Part 4')

### 4. Rounding Modes

# Set Precision
getcontext().prec = 5

# Test Values
a = Decimal('1.23456')
b = Decimal('1.23454')

# Rounding HALF_UP
getcontext().rounding = ROUND_HALF_UP
a_half_up = a.quantize(Decimal('1.0000'))
b_half_up = b.quantize(Decimal('1.0000'))
print(f'ROUND_UP_HALF:')
print(f'{a} rounded: {a_half_up}')
print(f'{b} rounded: {b_half_up}')

# Rounding DOWN
getcontext().rounding = ROUND_DOWN
a_down = a.quantize(Decimal('1.0000'))
b_down = b.quantize(Decimal('1.0000'))
print(f'ROUND_DOWN:')
print(f'{a} rounded: {a_down}')
print(f'{b} rounded: {b_down}')

# Rounding CEILING
getcontext().rounding = ROUND_CEILING
a_ceiling = a.quantize(Decimal('1.0000'))
b_ceiling = b.quantize(Decimal('1.0000'))
print(f'ROUND_CEILING:')
print(f'{a} rounded: {a_ceiling}')
print(f'{b} rounded: {b_ceiling}')
print('\n----------------------------------------\n')
print('Part 5')

### 5. Underflow and Overflow

# Overflow example
large_number = 1e308 * 1e10
print(f'Overflow example: {large_number}')

# Underflow example
small_number = 1e-308 / 1e10
print(f'Underflow example: {small_number}')

# Underflow leading to zero
tiny_number = 1e-324 / 1e10
print(f'Underflow to zero: {tiny_number}')

### 6. Visualizing Precision Loss

# Generate a range of small numbers (including subnormals)
exponents = np.arange(-45, -37, 0.1)
x = 10 ** exponents
y = np.float32(x)
precision_loss = np.abs(x - y)

plt.figure(figsize=(10, 6))
plt.plot(x, precision_loss, label='Precision Loss')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Value')
plt.ylabel('Precision Loss')
plt.title('Precision Loss for Very Small Numbers (Subnormals)')
plt.legend()
plt.grid(True)
plt.show()


### 7. Comparative Study

# Python:
# Uses double-precision (64-bit) floats by default.
# Follows IEEE 754 standards for floating-point arithmetic.
# Special values like inf and nan are supported.
# Limited control over rounding modes at the hardware level.

# C/C++:
# Provides more direct control over floating-point settings.
# Can change rounding modes using fesetround() from <fenv.h>.
# Allows handling of floating-point exceptions (overflow, underflow, divide-by-zero) using 