; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: sed 's/iXLen2/i64/g' %s | llc -mtriple=riscv32 -mattr=+m | \
; RUN:   FileCheck %s --check-prefix=RV32
; RUN: sed 's/iXLen2/i128/g' %s | llc -mtriple=riscv64 -mattr=+m | \
; RUN:   FileCheck %s --check-prefix=RV64

define iXLen2 @test_urem_3(iXLen2 %x) nounwind {
; RV32-LABEL: test_urem_3:
; RV32:       # %bb.0:
; RV32-NEXT:    add a1, a0, a1
; RV32-NEXT:    lui a2, 699051
; RV32-NEXT:    sltu a0, a1, a0
; RV32-NEXT:    addi a2, a2, -1365
; RV32-NEXT:    add a0, a1, a0
; RV32-NEXT:    mulhu a1, a0, a2
; RV32-NEXT:    srli a2, a1, 1
; RV32-NEXT:    andi a1, a1, -2
; RV32-NEXT:    add a1, a1, a2
; RV32-NEXT:    sub a0, a0, a1
; RV32-NEXT:    li a1, 0
; RV32-NEXT:    ret
;
; RV64-LABEL: test_urem_3:
; RV64:       # %bb.0:
; RV64-NEXT:    add a1, a0, a1
; RV64-NEXT:    lui a2, 699051
; RV64-NEXT:    sltu a0, a1, a0
; RV64-NEXT:    addi a2, a2, -1365
; RV64-NEXT:    add a0, a1, a0
; RV64-NEXT:    slli a1, a2, 32
; RV64-NEXT:    add a1, a2, a1
; RV64-NEXT:    mulhu a1, a0, a1
; RV64-NEXT:    srli a2, a1, 1
; RV64-NEXT:    andi a1, a1, -2
; RV64-NEXT:    add a1, a1, a2
; RV64-NEXT:    sub a0, a0, a1
; RV64-NEXT:    li a1, 0
; RV64-NEXT:    ret
  %a = urem iXLen2 %x, 3
  ret iXLen2 %a
}

define iXLen2 @test_urem_5(iXLen2 %x) nounwind {
; RV32-LABEL: test_urem_5:
; RV32:       # %bb.0:
; RV32-NEXT:    add a1, a0, a1
; RV32-NEXT:    lui a2, 838861
; RV32-NEXT:    sltu a0, a1, a0
; RV32-NEXT:    addi a2, a2, -819
; RV32-NEXT:    add a0, a1, a0
; RV32-NEXT:    mulhu a1, a0, a2
; RV32-NEXT:    srli a2, a1, 2
; RV32-NEXT:    andi a1, a1, -4
; RV32-NEXT:    add a1, a1, a2
; RV32-NEXT:    sub a0, a0, a1
; RV32-NEXT:    li a1, 0
; RV32-NEXT:    ret
;
; RV64-LABEL: test_urem_5:
; RV64:       # %bb.0:
; RV64-NEXT:    add a1, a0, a1
; RV64-NEXT:    lui a2, 838861
; RV64-NEXT:    sltu a0, a1, a0
; RV64-NEXT:    addi a2, a2, -819
; RV64-NEXT:    add a0, a1, a0
; RV64-NEXT:    slli a1, a2, 32
; RV64-NEXT:    add a1, a2, a1
; RV64-NEXT:    mulhu a1, a0, a1
; RV64-NEXT:    srli a2, a1, 2
; RV64-NEXT:    andi a1, a1, -4
; RV64-NEXT:    add a1, a1, a2
; RV64-NEXT:    sub a0, a0, a1
; RV64-NEXT:    li a1, 0
; RV64-NEXT:    ret
  %a = urem iXLen2 %x, 5
  ret iXLen2 %a
}

define iXLen2 @test_urem_7(iXLen2 %x) nounwind {
; RV32-LABEL: test_urem_7:
; RV32:       # %bb.0:
; RV32-NEXT:    addi sp, sp, -16
; RV32-NEXT:    sw ra, 12(sp) # 4-byte Folded Spill
; RV32-NEXT:    li a2, 7
; RV32-NEXT:    li a3, 0
; RV32-NEXT:    call __umoddi3
; RV32-NEXT:    lw ra, 12(sp) # 4-byte Folded Reload
; RV32-NEXT:    addi sp, sp, 16
; RV32-NEXT:    ret
;
; RV64-LABEL: test_urem_7:
; RV64:       # %bb.0:
; RV64-NEXT:    addi sp, sp, -16
; RV64-NEXT:    sd ra, 8(sp) # 8-byte Folded Spill
; RV64-NEXT:    li a2, 7
; RV64-NEXT:    li a3, 0
; RV64-NEXT:    call __umodti3
; RV64-NEXT:    ld ra, 8(sp) # 8-byte Folded Reload
; RV64-NEXT:    addi sp, sp, 16
; RV64-NEXT:    ret
  %a = urem iXLen2 %x, 7
  ret iXLen2 %a
}

define iXLen2 @test_urem_9(iXLen2 %x) nounwind {
; RV32-LABEL: test_urem_9:
; RV32:       # %bb.0:
; RV32-NEXT:    addi sp, sp, -16
; RV32-NEXT:    sw ra, 12(sp) # 4-byte Folded Spill
; RV32-NEXT:    li a2, 9
; RV32-NEXT:    li a3, 0
; RV32-NEXT:    call __umoddi3
; RV32-NEXT:    lw ra, 12(sp) # 4-byte Folded Reload
; RV32-NEXT:    addi sp, sp, 16
; RV32-NEXT:    ret
;
; RV64-LABEL: test_urem_9:
; RV64:       # %bb.0:
; RV64-NEXT:    addi sp, sp, -16
; RV64-NEXT:    sd ra, 8(sp) # 8-byte Folded Spill
; RV64-NEXT:    li a2, 9
; RV64-NEXT:    li a3, 0
; RV64-NEXT:    call __umodti3
; RV64-NEXT:    ld ra, 8(sp) # 8-byte Folded Reload
; RV64-NEXT:    addi sp, sp, 16
; RV64-NEXT:    ret
  %a = urem iXLen2 %x, 9
  ret iXLen2 %a
}

define iXLen2 @test_urem_15(iXLen2 %x) nounwind {
; RV32-LABEL: test_urem_15:
; RV32:       # %bb.0:
; RV32-NEXT:    add a1, a0, a1
; RV32-NEXT:    lui a2, 559241
; RV32-NEXT:    sltu a0, a1, a0
; RV32-NEXT:    add a0, a1, a0
; RV32-NEXT:    addi a1, a2, -1911
; RV32-NEXT:    mulhu a1, a0, a1
; RV32-NEXT:    srli a1, a1, 3
; RV32-NEXT:    slli a2, a1, 4
; RV32-NEXT:    sub a1, a1, a2
; RV32-NEXT:    add a0, a0, a1
; RV32-NEXT:    li a1, 0
; RV32-NEXT:    ret
;
; RV64-LABEL: test_urem_15:
; RV64:       # %bb.0:
; RV64-NEXT:    add a1, a0, a1
; RV64-NEXT:    lui a2, 559241
; RV64-NEXT:    sltu a0, a1, a0
; RV64-NEXT:    addi a2, a2, -1911
; RV64-NEXT:    add a0, a1, a0
; RV64-NEXT:    slli a1, a2, 32
; RV64-NEXT:    add a1, a2, a1
; RV64-NEXT:    mulhu a1, a0, a1
; RV64-NEXT:    srli a1, a1, 3
; RV64-NEXT:    slli a2, a1, 4
; RV64-NEXT:    sub a1, a1, a2
; RV64-NEXT:    add a0, a0, a1
; RV64-NEXT:    li a1, 0
; RV64-NEXT:    ret
  %a = urem iXLen2 %x, 15
  ret iXLen2 %a
}

define iXLen2 @test_urem_17(iXLen2 %x) nounwind {
; RV32-LABEL: test_urem_17:
; RV32:       # %bb.0:
; RV32-NEXT:    add a1, a0, a1
; RV32-NEXT:    lui a2, 986895
; RV32-NEXT:    sltu a0, a1, a0
; RV32-NEXT:    addi a2, a2, 241
; RV32-NEXT:    add a0, a1, a0
; RV32-NEXT:    mulhu a1, a0, a2
; RV32-NEXT:    srli a2, a1, 4
; RV32-NEXT:    andi a1, a1, -16
; RV32-NEXT:    add a1, a1, a2
; RV32-NEXT:    sub a0, a0, a1
; RV32-NEXT:    li a1, 0
; RV32-NEXT:    ret
;
; RV64-LABEL: test_urem_17:
; RV64:       # %bb.0:
; RV64-NEXT:    add a1, a0, a1
; RV64-NEXT:    lui a2, 986895
; RV64-NEXT:    sltu a0, a1, a0
; RV64-NEXT:    addi a2, a2, 241
; RV64-NEXT:    add a0, a1, a0
; RV64-NEXT:    slli a1, a2, 32
; RV64-NEXT:    add a1, a2, a1
; RV64-NEXT:    mulhu a1, a0, a1
; RV64-NEXT:    srli a2, a1, 4
; RV64-NEXT:    andi a1, a1, -16
; RV64-NEXT:    add a1, a1, a2
; RV64-NEXT:    sub a0, a0, a1
; RV64-NEXT:    li a1, 0
; RV64-NEXT:    ret
  %a = urem iXLen2 %x, 17
  ret iXLen2 %a
}

define iXLen2 @test_urem_255(iXLen2 %x) nounwind {
; RV32-LABEL: test_urem_255:
; RV32:       # %bb.0:
; RV32-NEXT:    add a1, a0, a1
; RV32-NEXT:    lui a2, 526344
; RV32-NEXT:    sltu a0, a1, a0
; RV32-NEXT:    add a0, a1, a0
; RV32-NEXT:    addi a1, a2, 129
; RV32-NEXT:    mulhu a1, a0, a1
; RV32-NEXT:    srli a1, a1, 7
; RV32-NEXT:    slli a2, a1, 8
; RV32-NEXT:    sub a1, a1, a2
; RV32-NEXT:    add a0, a0, a1
; RV32-NEXT:    li a1, 0
; RV32-NEXT:    ret
;
; RV64-LABEL: test_urem_255:
; RV64:       # %bb.0:
; RV64-NEXT:    add a1, a0, a1
; RV64-NEXT:    lui a2, 526344
; RV64-NEXT:    sltu a0, a1, a0
; RV64-NEXT:    addi a2, a2, 129
; RV64-NEXT:    add a0, a1, a0
; RV64-NEXT:    slli a1, a2, 32
; RV64-NEXT:    add a1, a2, a1
; RV64-NEXT:    mulhu a1, a0, a1
; RV64-NEXT:    srli a1, a1, 7
; RV64-NEXT:    slli a2, a1, 8
; RV64-NEXT:    sub a1, a1, a2
; RV64-NEXT:    add a0, a0, a1
; RV64-NEXT:    li a1, 0
; RV64-NEXT:    ret
  %a = urem iXLen2 %x, 255
  ret iXLen2 %a
}

define iXLen2 @test_urem_257(iXLen2 %x) nounwind {
; RV32-LABEL: test_urem_257:
; RV32:       # %bb.0:
; RV32-NEXT:    add a1, a0, a1
; RV32-NEXT:    lui a2, 1044496
; RV32-NEXT:    sltu a0, a1, a0
; RV32-NEXT:    addi a2, a2, -255
; RV32-NEXT:    add a0, a1, a0
; RV32-NEXT:    mulhu a1, a0, a2
; RV32-NEXT:    srli a2, a1, 8
; RV32-NEXT:    andi a1, a1, -256
; RV32-NEXT:    add a1, a1, a2
; RV32-NEXT:    sub a0, a0, a1
; RV32-NEXT:    li a1, 0
; RV32-NEXT:    ret
;
; RV64-LABEL: test_urem_257:
; RV64:       # %bb.0:
; RV64-NEXT:    add a1, a0, a1
; RV64-NEXT:    lui a2, 1044496
; RV64-NEXT:    sltu a0, a1, a0
; RV64-NEXT:    addi a2, a2, -255
; RV64-NEXT:    add a0, a1, a0
; RV64-NEXT:    slli a1, a2, 32
; RV64-NEXT:    add a1, a2, a1
; RV64-NEXT:    mulhu a1, a0, a1
; RV64-NEXT:    srli a2, a1, 8
; RV64-NEXT:    andi a1, a1, -256
; RV64-NEXT:    add a1, a1, a2
; RV64-NEXT:    sub a0, a0, a1
; RV64-NEXT:    li a1, 0
; RV64-NEXT:    ret
  %a = urem iXLen2 %x, 257
  ret iXLen2 %a
}

define iXLen2 @test_urem_65535(iXLen2 %x) nounwind {
; RV32-LABEL: test_urem_65535:
; RV32:       # %bb.0:
; RV32-NEXT:    add a1, a0, a1
; RV32-NEXT:    lui a2, 524296
; RV32-NEXT:    sltu a0, a1, a0
; RV32-NEXT:    add a0, a1, a0
; RV32-NEXT:    addi a2, a2, 1
; RV32-NEXT:    mulhu a1, a0, a2
; RV32-NEXT:    srli a1, a1, 15
; RV32-NEXT:    slli a2, a1, 16
; RV32-NEXT:    sub a1, a1, a2
; RV32-NEXT:    add a0, a0, a1
; RV32-NEXT:    li a1, 0
; RV32-NEXT:    ret
;
; RV64-LABEL: test_urem_65535:
; RV64:       # %bb.0:
; RV64-NEXT:    add a1, a0, a1
; RV64-NEXT:    lui a2, 524296
; RV64-NEXT:    sltu a0, a1, a0
; RV64-NEXT:    addi a2, a2, 1
; RV64-NEXT:    add a0, a1, a0
; RV64-NEXT:    slli a1, a2, 32
; RV64-NEXT:    add a1, a2, a1
; RV64-NEXT:    mulhu a1, a0, a1
; RV64-NEXT:    srli a1, a1, 15
; RV64-NEXT:    slli a2, a1, 16
; RV64-NEXT:    sub a1, a1, a2
; RV64-NEXT:    add a0, a0, a1
; RV64-NEXT:    li a1, 0
; RV64-NEXT:    ret
  %a = urem iXLen2 %x, 65535
  ret iXLen2 %a
}

define iXLen2 @test_urem_65537(iXLen2 %x) nounwind {
; RV32-LABEL: test_urem_65537:
; RV32:       # %bb.0:
; RV32-NEXT:    add a1, a0, a1
; RV32-NEXT:    lui a2, 1048560
; RV32-NEXT:    sltu a0, a1, a0
; RV32-NEXT:    add a0, a1, a0
; RV32-NEXT:    addi a1, a2, 1
; RV32-NEXT:    mulhu a1, a0, a1
; RV32-NEXT:    and a2, a1, a2
; RV32-NEXT:    srli a1, a1, 16
; RV32-NEXT:    or a1, a2, a1
; RV32-NEXT:    sub a0, a0, a1
; RV32-NEXT:    li a1, 0
; RV32-NEXT:    ret
;
; RV64-LABEL: test_urem_65537:
; RV64:       # %bb.0:
; RV64-NEXT:    add a1, a0, a1
; RV64-NEXT:    lui a2, 1048560
; RV64-NEXT:    sltu a0, a1, a0
; RV64-NEXT:    addi a3, a2, 1
; RV64-NEXT:    add a0, a1, a0
; RV64-NEXT:    slli a1, a3, 32
; RV64-NEXT:    add a1, a3, a1
; RV64-NEXT:    mulhu a1, a0, a1
; RV64-NEXT:    and a2, a1, a2
; RV64-NEXT:    srli a1, a1, 16
; RV64-NEXT:    add a1, a2, a1
; RV64-NEXT:    sub a0, a0, a1
; RV64-NEXT:    li a1, 0
; RV64-NEXT:    ret
  %a = urem iXLen2 %x, 65537
  ret iXLen2 %a
}

define iXLen2 @test_urem_12(iXLen2 %x) nounwind {
; RV32-LABEL: test_urem_12:
; RV32:       # %bb.0:
; RV32-NEXT:    slli a2, a1, 30
; RV32-NEXT:    srli a3, a0, 2
; RV32-NEXT:    srli a1, a1, 2
; RV32-NEXT:    or a2, a3, a2
; RV32-NEXT:    lui a3, 699051
; RV32-NEXT:    addi a3, a3, -1365
; RV32-NEXT:    add a1, a2, a1
; RV32-NEXT:    sltu a2, a1, a2
; RV32-NEXT:    add a1, a1, a2
; RV32-NEXT:    mulhu a2, a1, a3
; RV32-NEXT:    srli a3, a2, 1
; RV32-NEXT:    andi a2, a2, -2
; RV32-NEXT:    add a2, a2, a3
; RV32-NEXT:    sub a1, a1, a2
; RV32-NEXT:    slli a1, a1, 2
; RV32-NEXT:    andi a0, a0, 3
; RV32-NEXT:    or a0, a1, a0
; RV32-NEXT:    li a1, 0
; RV32-NEXT:    ret
;
; RV64-LABEL: test_urem_12:
; RV64:       # %bb.0:
; RV64-NEXT:    slli a2, a1, 62
; RV64-NEXT:    srli a3, a0, 2
; RV64-NEXT:    lui a4, 699051
; RV64-NEXT:    or a2, a3, a2
; RV64-NEXT:    addi a3, a4, -1365
; RV64-NEXT:    slli a4, a3, 32
; RV64-NEXT:    add a3, a3, a4
; RV64-NEXT:    srli a1, a1, 2
; RV64-NEXT:    add a1, a2, a1
; RV64-NEXT:    sltu a2, a1, a2
; RV64-NEXT:    add a1, a1, a2
; RV64-NEXT:    mulhu a2, a1, a3
; RV64-NEXT:    srli a3, a2, 1
; RV64-NEXT:    andi a2, a2, -2
; RV64-NEXT:    add a2, a2, a3
; RV64-NEXT:    sub a1, a1, a2
; RV64-NEXT:    slli a1, a1, 2
; RV64-NEXT:    andi a0, a0, 3
; RV64-NEXT:    or a0, a1, a0
; RV64-NEXT:    li a1, 0
; RV64-NEXT:    ret
  %a = urem iXLen2 %x, 12
  ret iXLen2 %a
}
