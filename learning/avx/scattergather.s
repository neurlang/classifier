//go:build !noasm && amd64

#include "textflag.h"

// Function definition
TEXT ·scatterGatherAVX512Vectorized(SB), NOSPLIT, $0-65
    // Load arguments
    MOVQ    outs+0(FP), R10           // R10 = &outs[0]
    MOVQ    outs+8(FP), DI            // DI = len(outs)
    MOVQ    buf+24(FP), BX            // BX = &buf[0]
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
    MOVL $3, DX
    VPBROADCASTD  DX, Z13              // Z3 = broadcast(3)
    MOVL $15, DX
    VPBROADCASTD  DX, Z12              // Z3 = broadcast(15)
    
    // Increment index by 16
    ADDQ    $16, AX

vector_loop:

    // Decrement index by 16
    SUBQ    $16, AX

    // Load 16 indices from outs[AX]
    VMOVDQU32 (R10)(AX*4), Z1

    VPANDD Z12, Z1, Z15
    VPSRLD $4, Z1, Z1
    VPSLLD $1, Z15, Z15

    // Check for conflicts in Z1
    VPCONFLICTD Z1, Z4                // Z4 will contain conflict detection results
    VPTESTMD Z4, Z4, K2               // K2 = mask where conflicts are detected
    KORTESTW K2, K2
    JNE     scalar_loop               // If conflicts detected, switch to scalar loop

    // Increment index by 16
    ADDQ    $16, AX

    // Gather buf values into Z0
    KXNORW  K1, K1, K1                // K1 = all ones
    VPGATHERDD (BX)(Z1*4), K1, Z0


    VPSRLVD Z15, Z0, Z14
    VPANDD Z13, Z14, Z14

    // Check if any element equals 2 - wh
    VPCMPD  $0, Z2, Z14, K0            // EQ comparison
    KORTESTW K0, K0
    JNE     return_true               // If any match, return true

    // Count elements where buf[v] == 0
    VPTESTNMD Z14, Z14, K4              // K4 = mask where elements are zero
    KMOVW   K4, DX                    // Move mask to DX
    POPCNTW DX, DX                    // Count zeros
    ANDL    $31, DX
    ADDL    DX, (R11)                 // Update size

    // or it to original
    VPSLLVD Z15, Z3, Z14
    VPORD   Z14, Z0, Z0
    
    // scatter back
    KXNORW  K1, K1, K1                // K1 = all ones
    VPSCATTERDD Z0, K1, (BX)(Z1*4)


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

    MOVQ    AX, R9                    // R9 = 0 (scalar loop index)

scalar_loop_start:

    // Calculate the address of outs[R9]
    CMPQ    R9, DI
    JGE     return_false

    MOVL    (R10)(R9*4), R14         // R14 = outs[R9]
    
    MOVL    R14, CX
    SHRL    $4, R14
    ANDL    $15, CX
    SHLL    $1, CX

    // Calculate the address of buf[R14]
    MOVL    (BX)(R14*4), AX                // AX = buf[R14]
    SHRL    CX, AX

    ANDL    $3, AX


    CMPL    AX, R12                  // Compare with 2 - wh
    JE      return_true

    TESTL   AX, AX
    JNE     scalar_check_2wh
    INCL    (R11)                     // Increment size

scalar_check_2wh:

    SHLL    CX, R8

    MOVL    (BX)(R14*4), AX                // AX = buf[R14]
    ORL     R8, AX
    MOVL    AX, (BX)(R14*4)                // Store back

    SHRL    CX, R8

    INCL    R9                        // Increment scalar index
    JMP     scalar_loop_start

return_true:
    VZEROUPPER
    MOVB    $1, ret+64(FP)
    RET

return_false:
    VZEROUPPER
    MOVB    $0, ret+64(FP)
    RET
