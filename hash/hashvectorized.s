#include "textflag.h"

// Data section for the shuffle constant lookup table
DATA ·LCPI0_0+0(SB)/64, ZCONSTANT64 $0x0, $0x1, $0x2, $0x3, $0x4, $0x5, $0x6, $0x7

// func hashVectorizedAVX512(out *uint32, n *uint32, s *uint32, max uint32, length uint32)
TEXT ·hashVectorizedAVX512(SB), NOSPLIT, $0-40
    MOVQ out+0(FP), DI
    MOVQ n+8(FP), SI
    MOVQ s+16(FP), DX
    MOVL max+24(FP), R8
    MOVL len+28(FP), CX

    // Preserve length for bounds checking
    MOVL CX, R9

    // Broadcast max to Z31
    VPBROADCASTD R8, Z31

    // Prepare to use mask: move immediate 21845 into eax (k1 corresponds to 0b0101010101010101)
    MOVL $21845, AX
    KMOVW K1, AX

    // Check if we have at least 16 elements
    CMPQ R9, $16
    JL remainder_loop

    // Process 16 elements at a time
    SHRQ $4, CX
    JZ remainder_loop

loop:
    // Load 16 elements from n and s
    VMOVDQU32 (SI), Z0
    VMOVDQU32 (DX), Z1

    // m = n - s
    VPSUBD Z1, Z0, Z2

    // Hashing stage
    VPSLLD $2, Z2, Z3
    VPXORD Z3, Z2, Z2
    VPSLLD $3, Z2, Z3
    VPXORD Z3, Z2, Z2
    VPSRLD $5, Z2, Z3
    VPXORD Z3, Z2, Z2
    VPSRLD $7, Z2, Z3
    VPXORD Z3, Z2, Z2
    VPSLLD $11, Z2, Z3
    VPXORD Z3, Z2, Z2
    VPSLLD $13, Z2, Z3
    VPXORD Z3, Z2, Z2
    VPSRLD $17, Z2, Z3
    VPXORD Z3, Z2, Z2
    VPSLLD $19, Z2, Z3
    VPXORD Z3, Z2, Z2

    // m += s
    VPADDD Z1, Z2, Z2

    // Modular reduction: (uint64(m) * uint64(max)) >> 32
    // First multiply (even lanes)
    VPMULUDQ Z31, Z2, Z3  // Z3 = Z1 * Z2 (even lanes)

    // Shift right by 32 bits to handle the odd lanes
    VPSRLQ $32, Z2, Z2   // Z2 >>= 32
    VPSRLQ $32, Z1, Z1   // Z1 >>= 32

    // Second multiply (odd lanes)
    VPMULUDQ Z31, Z2, Z2  // Z2 = Z1 * Z2 (odd lanes)

    // Load constant lookup table for permutation
    // The constant table will be placed later in the data section
    VMOVDQA64 ·LCPI0_0(SB), Z0  // Z0 = constant shuffle table

    // Use vpermi2d to interleave the results from Z3 and Z2 based on the lookup table in Z0
    VPERMI2D Z3, Z0, Z2  // Z2 = interleave(Z3, Z0, Z2)

    // Store result back
    VMOVDQU32 Z2, (DI)

    ADDQ $64, SI
    ADDQ $64, DX
    ADDQ $64, DI
    SUBQ $16, R9
    DECQ CX
    JNZ loop

remainder_loop:
    CMPQ R9, $0
    JE end_loop                // Exit if no elements left

    MOVL (SI), AX              // Load n (scalar)
    MOVL (DX), BX              // Load s (scalar)
    SUBL BX, AX                // m = n - s

    // Hashing stage: XOR shifts
    MOVL AX, R10
    SHLL $2, R10
    XORL R10, AX
    MOVL AX, R10
    SHLL $3, R10
    XORL R10, AX
    MOVL AX, R10
    SHRL $5, R10
    XORL R10, AX
    MOVL AX, R10
    SHRL $7, R10
    XORL R10, AX
    MOVL AX, R10
    SHLL $11, R10
    XORL R10, AX
    MOVL AX, R10
    SHLL $13, R10
    XORL R10, AX
    MOVL AX, R10
    SHRL $17, R10
    XORL R10, AX
    MOVL AX, R10
    SHLL $19, R10
    XORL R10, AX

    // Second mixing stage: Add s
    ADDL BX, AX                // m += s

    // Modular reduction using multiply-shift method
    MOVL AX, R11               // Save m in R11
    MOVL $0, R10               // Clear upper 32 bits of R10:R11
    MOVL R8, AX                // Move max to AX
    MULL R11                   // Multiply m by max, result in EDX:EAX
    MOVL DX, (DI)             // Store high 32 bits (EDX) to output

    ADDQ $4, SI                // Move to next n (advance pointer)
    ADDQ $4, DX                // Move to next s (advance pointer)
    ADDQ $4, DI                // Move to next out (advance pointer)
    DECQ R9                    // Decrease remaining element count
    JNZ remainder_loop         // Continue if remaining elements

end_loop:
    //VZEROUPPER                 // Clear upper parts of YMM registers - not needed if we don't use it
    RET
