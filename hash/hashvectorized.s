//go:build !noasm && amd64

#include "textflag.h"


// func hashVectorizedAVX512(out *uint32, n *uint32, s *uint32, max uint32, length uint32)
TEXT ·hashVectorizedAVX512(SB), NOSPLIT, $0-32
    MOVQ out+0(FP), DI
    MOVQ n+8(FP), SI
    MOVQ s+16(FP), CX
    MOVL max+24(FP), R8
    MOVL length+28(FP), DX

    // Preserve length for bounds checking
    MOVL DX, R9

    // Prepare to use mask: move immediate 21845 into eax (k1 corresponds to 0b0101010101010101)
    MOVL $21845, AX
    KMOVW K1, AX

    // Check if we have at least 16 elements
    CMPL R9, $16
    JL remainder_loop

loop:
    // Broadcast max to Z31
    VPBROADCASTD R8, Z31
    // Load 16 elements from n and s
    VMOVDQU32 (SI), Z0
    VMOVDQU32 (CX), Z1

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
    VPADDD Z1, Z2, Z1  // Z2 = Z1 + Z2

    VPMULUDQ  Z31, Z1, Z2
    VPSRLQ $32, Z31, Z31
    VPSRLQ $32, Z1, Z1
    VPMULUDQ Z31, Z1, Z1

    // Load permutation table
    VMOVDQU32 ·lCPI0_0(SB), Z0
    // Permute the result
    VPERMI2D Z1, Z2, Z0

    VMOVDQU32 Z0, (DI)   // Store result


    ADDQ $64, SI
    ADDQ $64, CX
    ADDQ $64, DI
    SUBL $16, R9
    // Check if we have at least 16 elements
    CMPL R9, $16
    JAE loop

remainder_loop:
    CMPL R9, $0
    JE end_loop                // Exit if no elements left

    MOVL (SI), AX              // Load n (scalar)
    MOVL (CX), BX              // Load s (scalar)
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
    ADDQ $4, CX                // Move to next s (advance pointer)
    ADDQ $4, DI                // Move to next out (advance pointer)
    DECL R9                    // Decrease remaining element count
    CMPL R9, $0
    JNZ remainder_loop         // Continue if remaining elements

end_loop:
    VZEROUPPER                 // Clear upper parts of YMM registers - not needed if we don't use it
    RET


// func hashVectorizedDistinctAVX512(out *uint32, n *uint32, s *uint32, max *uint32, length uint32)
TEXT ·hashVectorizedDistinctAVX512(SB), NOSPLIT, $0-36
    MOVQ out+0(FP), DI
    MOVQ n+8(FP), SI
    MOVQ s+16(FP), CX
    MOVQ max+24(FP), R8  // Load max array pointer
    MOVL length+32(FP), DX   // Load length from correct offset

    // Preserve length for bounds checking
    MOVL DX, R9

    // Prepare to use mask: move immediate 21845 into eax (k1 corresponds to 0b0101010101010101)
    MOVL $21845, AX
    KMOVW K1, AX

    // Check if we have at least 16 elements
    CMPL R9, $16
    JL remainder_loop

loop:
    // Load 16 max values into Z31
    VMOVDQU32 (R8), Z31
    // Load 16 elements from n and s
    VMOVDQU32 (SI), Z0
    VMOVDQU32 (CX), Z1

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
    VPADDD Z1, Z2, Z1  // Z2 = Z1 + Z2

    VPMULUDQ  Z31, Z1, Z2
    VPSRLQ $32, Z31, Z31
    VPSRLQ $32, Z1, Z1
    VPMULUDQ Z31, Z1, Z1

    // Load permutation table
    VMOVDQU32 ·lCPI0_0(SB), Z0
    // Permute the result
    VPERMI2D Z1, Z2, Z0

    VMOVDQU32 Z0, (DI)   // Store result

    // Advance pointers by 16 elements (64 bytes)
    ADDQ $64, SI
    ADDQ $64, CX
    ADDQ $64, DI
    ADDQ $64, R8         // Advance max pointer
    SUBL $16, R9
    // Check if remaining elements >=16
    CMPL R9, $16
    JAE loop

remainder_loop:
    CMPL R9, $0
    JE end_loop

    MOVL (SI), AX        // Load n[i]
    MOVL (CX), BX        // Load s[i]
    SUBL BX, AX          // m = n - s

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

    ADDL BX, AX          // m += s

    // Load current max from array and compute
    MOVL (R8), R11       // max[i]
    MOVL AX, R10         // m
    MOVL R11, AX         // Move max to AX for MUL
    MULL R10             // EDX:EAX = max * m
    MOVL DX, (DI)        // Store result

    // Advance pointers by 1 element (4 bytes)
    ADDQ $4, SI
    ADDQ $4, CX
    ADDQ $4, DI
    ADDQ $4, R8          // Next max element
    DECL R9
    JNZ remainder_loop

end_loop:
    VZEROUPPER
    RET
