#include "textflag.h"

// func setorallocAVX512Vectorized(sets **byte, imodmax *uint32, j uint32, len uint32)
TEXT ·setorallocAVX512Vectorized(SB), NOSPLIT, $0-24
    MOVQ sets+0(FP), SI     // SI = pointer to sets array
    MOVQ imodmax+8(FP), CX  // CX = imodmax array
    MOVL j+16(FP), DX       // DX = j
    MOVL len+20(FP), R9     // R9 = len (number of elements)
    
    ANDL $1, DX
    INCB DX                 // DX = (j&1) + 1
    VPBROADCASTD DX, Z11    // Z11 = repeated (j&1) + 1 across all lanes
    
    CMPL R9, $16
    JB   scalar_loop        // Jump to scalar loop if len < 16
    

avx512_loop:

    
    MOVQ (SI), AX           // AX = lowest sets pointer
    VPBROADCASTQ AX, Z31    // Broadcast lowest sets pointer to Z31
    
    // Load permutation table
    VMOVDQU32 ·lCPI0_0(SB), Z0

    VMOVDQU32 (CX), Z1      // Z1 = imodmax[ii:ii+16]
    VMOVDQU64 (SI), Z2      // Z2 = pointers to sets[ii:ii+8]
    ADDQ $64, SI
    VMOVDQU64 (SI), Z3      // Z3 = pointers to sets[ii+8:ii+16]
    
    VPSUBQ Z31, Z2, Z2      // Z2 now contains relative addresses
    VPSUBQ Z31, Z3, Z3      // Z3 now contains relative addresses
    
    // Permute the result
    VPERMI2D Z3, Z2, Z0     // Z0 now contains all 16 relative addresses
    
    // Divide Z0 by 4 to convert byte offsets to 32-bit element offsets
    //VPSRLD $32, Z0, Z0
    
        // Calculate indices and shifts
    VPSRLD $2, Z1, Z10      // Z10 = imodmax[ii:ii+16] >> 2
    //VPSLLD $32, Z10, Z10
    VPSLLD $30, Z1, Z4      // Shift right by 30 instead of VPANDD $3
    VPSRLD $29, Z4, Z4      // Shift left by 29 to get (imodmax[ii] & 3) << 1
    VPADDD Z0, Z10, Z5      // Z5 now contains relative addresses + (imodmax[ii] >> 2)

    
    // Prepare values to OR: (j&1 + 1) << ((imodmax[ii] & 3) << 1)
    VPSLLVD Z4, Z11, Z7
    
    MOVL $0xFFFF, BX        // Mask for 16 lanes
    KMOVW BX, K1
    // Gather current values
    VPGATHERDD (AX)(Z5*1), K1, Z6  // Use AX (which is lowest pointer) as base, scale by 4 
   
    // OR gathered values with shifted (j&1)+1
    VPORD Z7, Z6, Z8
    
    MOVL $0xFFFF, BX        // Mask for 16 lanes
    KMOVW BX, K1
    // Scatter results back
    VPSCATTERDD Z8, K1, (AX)(Z5*1)  // Scale by 4
    
    ADDQ $64, CX            // Move to next 16 imodmax values
    ADDQ $64, SI           // Move to next 16 set pointers
    SUBL $16, R9
    CMPL R9, $16
    JAE  avx512_loop
    
    CMPL R9, $0
    JE   done

scalar_loop:
    MOVQ (SI), BX           // Load pointer to current set
    MOVL (CX), AX           // Load current imodmax value
    MOVL AX, DI
    SHRL $2, DI             // DI = imodmax[ii] >> 2
    ANDL $3, AX
    SHLL $1, AX             // AX = (imodmax[ii] & 3) << 1
    MOVB DX, R8B
    SHLB CL, R8B            // R8B = (j&1 + 1) << ((imodmax[ii] & 3) << 1)
    ORB  R8B, (BX)(DI*1)    // OR with sets[ii][imodmax[ii]>>2]
    ADDQ $4, CX             // Move to next imodmax
    ADDQ $8, SI             // Move to next set pointer
    DECL R9
    JNZ  scalar_loop

done:
    VZEROUPPER
    RET
