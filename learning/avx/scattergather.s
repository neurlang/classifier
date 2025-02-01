//go:build !noasm && amd64

#include "textflag.h"

// Function definition
TEXT ·scatterGatherAVX512Vectorized(SB), NOSPLIT, $0-65
    // Load arguments
    MOVQ    outs+0(FP), R10           // R10 = &outs[0]
    MOVQ    outs+8(FP), DI            // DI = len(outs)
    MOVQ    buf+24(FP), CX            // CX = &buf[0]
    MOVQ    size+48(FP), R11          // R11 = size pointer
    MOVL    wh+56(FP), R8             // R8 = wh

    // Calculate constants
    MOVL    $2, R9
    SUBL    R8, R9                    // R9 = 2 - wh
    MOVL    R9, R12                   // Save 2 - wh in R12 for scalar loop
    INCL    R8                        // R8 = 1 + wh

    XORQ    AX, AX                    // i = 0

    // Check if vector processing is possible
    CMPQ    DI, $16
    JL      scalar_loop                // Jump to scalar if len < 16

    // Vectorized setup
    VPBROADCASTD  R12, Z2             // Z2 = broadcast(2 - wh)
    VPBROADCASTD  R8, Z3              // Z3 = broadcast(1 + wh)


    // Increment index by 16
    ADDQ    $16, AX

vector_loop:

    // Decrement index by 16
    SUBQ    $16, AX

    // Load 16 indices from outs[AX]
    VMOVDQU32 (R10)(AX*4), Z1

    // Check for conflicts in Z0
    VPCONFLICTD Z1, Z4                // Z4 will contain conflict detection results
    VPTESTMD Z4, Z4, K2               // K2 = mask where conflicts are detected
    KORTESTW K2, K2
    JNE     scalar_loop               // If conflicts detected, switch to scalar loop


    // Increment index by 16
    ADDQ    $16, AX

    // Gather buf values into Z0
    KXNORW  K1, K1, K1                // K1 = all ones
    VPGATHERDD (CX)(Z1*4), K1, Z0

    // Check if any element equals 2 - wh
    VPCMPD  $0, Z2, Z0, K0            // EQ comparison
    KORTESTW K0, K0
    JNE     return_true               // If any match, return true

    // Count elements where buf[v] == 0
    VPTESTNMD Z0, Z0, K4              // K4 = mask where elements are zero
    KMOVW   K4, DX                    // Move mask to DX
    POPCNTW DX, DX                    // Count zeros
    ANDL    $31, DX
    ADDL    DX, (R11)                 // Update size

    // scatter back
    KXNORW  K1, K1, K1                // K1 = all ones
    VPSCATTERDD Z3, K1, (CX)(Z1*4)


    // Check if exactly 0 elements
    CMPQ    AX, DI
    JZ      return_false

    // Increment index by 16
    ADDQ    $16, AX
    
    // Check if more elements
    CMPQ    AX, DI
    JL      vector_loop


    // Decrement index by 16
    SUBQ    $16, AX
    



scalar_loop:
    XORQ    R9, R9                    // R9 = 0 (scalar loop index)

scalar_loop_start:

    // Calculate the address of outs[AX + R9]
    MOVQ    R9, R13                   // R13 = AX (base index from vectorized loop)
    ADDQ    AX, R13                   // R13 = AX + R9 (current index)
    CMPQ    R13, DI
    JGE     return_false

    MOVL    (R10)(R13*4), R14         // R14 = outs[AX + R9]

    // Calculate the address of buf[R14]
    LEAQ    (CX)(R14*4), R15          // R15 = &buf[R14]
    MOVL    (R15), R13                // R13 = buf[R14]

    TESTL   R13, R13
    JNE     scalar_check_2wh
    INCL    (R11)                     // Increment size

scalar_check_2wh:
    CMPL    R13, R12                  // Compare with 2 - wh
    JE      return_true_scalar

    MOVL    R8, (R15)                // Store back

    INCL    R9                        // Increment scalar index
    JMP     scalar_loop_start

return_true_scalar:
    VZEROUPPER
    MOVB    $1, ret+64(FP)
    RET

return_true:
    VZEROUPPER
    MOVB    $1, ret+64(FP)
    RET

return_false:
    VZEROUPPER
    MOVB    $0, ret+64(FP)
    RET
