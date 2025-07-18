; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py UTC_ARGS: --version 2
; RUN: llc -mcpu=pwr9 -mtriple=powerpc64le-unknown-unknown \
; RUN:   -ppc-vsr-nums-as-vr -ppc-asm-full-reg-names < %s | FileCheck %s

define { half, i32 } @test_frexp_f16_i32(half %a) nounwind {
; CHECK-LABEL: test_frexp_f16_i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    mflr r0
; CHECK-NEXT:    stdu r1, -48(r1)
; CHECK-NEXT:    std r0, 64(r1)
; CHECK-NEXT:    xscvdphp f0, f1
; CHECK-NEXT:    addi r4, r1, 44
; CHECK-NEXT:    mffprwz r3, f0
; CHECK-NEXT:    clrlwi r3, r3, 16
; CHECK-NEXT:    mtfprwz f0, r3
; CHECK-NEXT:    xscvhpdp f1, f0
; CHECK-NEXT:    bl frexpf
; CHECK-NEXT:    nop
; CHECK-NEXT:    lwz r3, 44(r1)
; CHECK-NEXT:    addi r1, r1, 48
; CHECK-NEXT:    ld r0, 16(r1)
; CHECK-NEXT:    mtlr r0
; CHECK-NEXT:    blr
  %result = call { half, i32 } @llvm.frexp.f16.i32(half %a)
  ret { half, i32 } %result
}

define half @test_frexp_f16_i32_only_use_fract(half %a) nounwind {
; CHECK-LABEL: test_frexp_f16_i32_only_use_fract:
; CHECK:       # %bb.0:
; CHECK-NEXT:    mflr r0
; CHECK-NEXT:    stdu r1, -48(r1)
; CHECK-NEXT:    std r0, 64(r1)
; CHECK-NEXT:    xscvdphp f0, f1
; CHECK-NEXT:    addi r4, r1, 44
; CHECK-NEXT:    mffprwz r3, f0
; CHECK-NEXT:    clrlwi r3, r3, 16
; CHECK-NEXT:    mtfprwz f0, r3
; CHECK-NEXT:    xscvhpdp f1, f0
; CHECK-NEXT:    bl frexpf
; CHECK-NEXT:    nop
; CHECK-NEXT:    addi r1, r1, 48
; CHECK-NEXT:    ld r0, 16(r1)
; CHECK-NEXT:    mtlr r0
; CHECK-NEXT:    blr
  %result = call { half, i32 } @llvm.frexp.f16.i32(half %a)
  %result.0 = extractvalue { half, i32 } %result, 0
  ret half %result.0
}

define i32 @test_frexp_f16_i32_only_use_exp(half %a) nounwind {
; CHECK-LABEL: test_frexp_f16_i32_only_use_exp:
; CHECK:       # %bb.0:
; CHECK-NEXT:    mflr r0
; CHECK-NEXT:    stdu r1, -48(r1)
; CHECK-NEXT:    std r0, 64(r1)
; CHECK-NEXT:    xscvdphp f0, f1
; CHECK-NEXT:    addi r4, r1, 44
; CHECK-NEXT:    mffprwz r3, f0
; CHECK-NEXT:    clrlwi r3, r3, 16
; CHECK-NEXT:    mtfprwz f0, r3
; CHECK-NEXT:    xscvhpdp f1, f0
; CHECK-NEXT:    bl frexpf
; CHECK-NEXT:    nop
; CHECK-NEXT:    lwz r3, 44(r1)
; CHECK-NEXT:    addi r1, r1, 48
; CHECK-NEXT:    ld r0, 16(r1)
; CHECK-NEXT:    mtlr r0
; CHECK-NEXT:    blr
  %result = call { half, i32 } @llvm.frexp.f16.i32(half %a)
  %result.0 = extractvalue { half, i32 } %result, 1
  ret i32 %result.0
}

define { <2 x half>, <2 x i32> } @test_frexp_v2f16_v2i32(<2 x half> %a) nounwind {
; CHECK-LABEL: test_frexp_v2f16_v2i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    mflr r0
; CHECK-NEXT:    std r29, -40(r1) # 8-byte Folded Spill
; CHECK-NEXT:    std r30, -32(r1) # 8-byte Folded Spill
; CHECK-NEXT:    stfd f30, -16(r1) # 8-byte Folded Spill
; CHECK-NEXT:    stfd f31, -8(r1) # 8-byte Folded Spill
; CHECK-NEXT:    stdu r1, -80(r1)
; CHECK-NEXT:    std r0, 96(r1)
; CHECK-NEXT:    xscvdphp f0, f2
; CHECK-NEXT:    addi r30, r1, 32
; CHECK-NEXT:    mr r4, r30
; CHECK-NEXT:    mffprwz r3, f0
; CHECK-NEXT:    clrlwi r3, r3, 16
; CHECK-NEXT:    mtfprwz f0, r3
; CHECK-NEXT:    xscvhpdp f31, f0
; CHECK-NEXT:    xscvdphp f0, f1
; CHECK-NEXT:    mffprwz r3, f0
; CHECK-NEXT:    clrlwi r3, r3, 16
; CHECK-NEXT:    mtfprwz f0, r3
; CHECK-NEXT:    xscvhpdp f1, f0
; CHECK-NEXT:    bl frexpf
; CHECK-NEXT:    nop
; CHECK-NEXT:    addi r29, r1, 36
; CHECK-NEXT:    fmr f30, f1
; CHECK-NEXT:    fmr f1, f31
; CHECK-NEXT:    mr r4, r29
; CHECK-NEXT:    bl frexpf
; CHECK-NEXT:    nop
; CHECK-NEXT:    fmr f2, f1
; CHECK-NEXT:    lfiwzx f0, 0, r30
; CHECK-NEXT:    lfiwzx f1, 0, r29
; CHECK-NEXT:    xxmrghw v2, vs1, vs0
; CHECK-NEXT:    fmr f1, f30
; CHECK-NEXT:    addi r1, r1, 80
; CHECK-NEXT:    ld r0, 16(r1)
; CHECK-NEXT:    lfd f31, -8(r1) # 8-byte Folded Reload
; CHECK-NEXT:    lfd f30, -16(r1) # 8-byte Folded Reload
; CHECK-NEXT:    ld r30, -32(r1) # 8-byte Folded Reload
; CHECK-NEXT:    ld r29, -40(r1) # 8-byte Folded Reload
; CHECK-NEXT:    mtlr r0
; CHECK-NEXT:    blr
  %result = call { <2 x half>, <2 x i32> } @llvm.frexp.v2f16.v2i32(<2 x half> %a)
  ret { <2 x half>, <2 x i32> } %result
}

define <2 x half> @test_frexp_v2f16_v2i32_only_use_fract(<2 x half> %a) nounwind {
; CHECK-LABEL: test_frexp_v2f16_v2i32_only_use_fract:
; CHECK:       # %bb.0:
; CHECK-NEXT:    mflr r0
; CHECK-NEXT:    stfd f30, -16(r1) # 8-byte Folded Spill
; CHECK-NEXT:    stfd f31, -8(r1) # 8-byte Folded Spill
; CHECK-NEXT:    stdu r1, -64(r1)
; CHECK-NEXT:    std r0, 80(r1)
; CHECK-NEXT:    xscvdphp f0, f2
; CHECK-NEXT:    addi r4, r1, 40
; CHECK-NEXT:    mffprwz r3, f0
; CHECK-NEXT:    clrlwi r3, r3, 16
; CHECK-NEXT:    mtfprwz f0, r3
; CHECK-NEXT:    xscvhpdp f31, f0
; CHECK-NEXT:    xscvdphp f0, f1
; CHECK-NEXT:    mffprwz r3, f0
; CHECK-NEXT:    clrlwi r3, r3, 16
; CHECK-NEXT:    mtfprwz f0, r3
; CHECK-NEXT:    xscvhpdp f1, f0
; CHECK-NEXT:    bl frexpf
; CHECK-NEXT:    nop
; CHECK-NEXT:    addi r4, r1, 44
; CHECK-NEXT:    fmr f30, f1
; CHECK-NEXT:    fmr f1, f31
; CHECK-NEXT:    bl frexpf
; CHECK-NEXT:    nop
; CHECK-NEXT:    fmr f2, f1
; CHECK-NEXT:    fmr f1, f30
; CHECK-NEXT:    addi r1, r1, 64
; CHECK-NEXT:    ld r0, 16(r1)
; CHECK-NEXT:    lfd f31, -8(r1) # 8-byte Folded Reload
; CHECK-NEXT:    lfd f30, -16(r1) # 8-byte Folded Reload
; CHECK-NEXT:    mtlr r0
; CHECK-NEXT:    blr
  %result = call { <2 x half>, <2 x i32> } @llvm.frexp.v2f16.v2i32(<2 x half> %a)
  %result.0 = extractvalue { <2 x half>, <2 x i32> } %result, 0
  ret <2 x half> %result.0
}

define <2 x i32> @test_frexp_v2f16_v2i32_only_use_exp(<2 x half> %a) nounwind {
; CHECK-LABEL: test_frexp_v2f16_v2i32_only_use_exp:
; CHECK:       # %bb.0:
; CHECK-NEXT:    mflr r0
; CHECK-NEXT:    std r29, -32(r1) # 8-byte Folded Spill
; CHECK-NEXT:    std r30, -24(r1) # 8-byte Folded Spill
; CHECK-NEXT:    stfd f31, -8(r1) # 8-byte Folded Spill
; CHECK-NEXT:    stdu r1, -80(r1)
; CHECK-NEXT:    std r0, 96(r1)
; CHECK-NEXT:    xscvdphp f0, f2
; CHECK-NEXT:    addi r30, r1, 40
; CHECK-NEXT:    mr r4, r30
; CHECK-NEXT:    mffprwz r3, f0
; CHECK-NEXT:    clrlwi r3, r3, 16
; CHECK-NEXT:    mtfprwz f0, r3
; CHECK-NEXT:    xscvhpdp f31, f0
; CHECK-NEXT:    xscvdphp f0, f1
; CHECK-NEXT:    mffprwz r3, f0
; CHECK-NEXT:    clrlwi r3, r3, 16
; CHECK-NEXT:    mtfprwz f0, r3
; CHECK-NEXT:    xscvhpdp f1, f0
; CHECK-NEXT:    bl frexpf
; CHECK-NEXT:    nop
; CHECK-NEXT:    addi r29, r1, 44
; CHECK-NEXT:    fmr f1, f31
; CHECK-NEXT:    mr r4, r29
; CHECK-NEXT:    bl frexpf
; CHECK-NEXT:    nop
; CHECK-NEXT:    lfiwzx f0, 0, r30
; CHECK-NEXT:    lfiwzx f1, 0, r29
; CHECK-NEXT:    xxmrghw v2, vs1, vs0
; CHECK-NEXT:    addi r1, r1, 80
; CHECK-NEXT:    ld r0, 16(r1)
; CHECK-NEXT:    lfd f31, -8(r1) # 8-byte Folded Reload
; CHECK-NEXT:    ld r30, -24(r1) # 8-byte Folded Reload
; CHECK-NEXT:    ld r29, -32(r1) # 8-byte Folded Reload
; CHECK-NEXT:    mtlr r0
; CHECK-NEXT:    blr
  %result = call { <2 x half>, <2 x i32> } @llvm.frexp.v2f16.v2i32(<2 x half> %a)
  %result.1 = extractvalue { <2 x half>, <2 x i32> } %result, 1
  ret <2 x i32> %result.1
}

define { float, i32 } @test_frexp_f32_i32(float %a) nounwind {
; CHECK-LABEL: test_frexp_f32_i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    mflr r0
; CHECK-NEXT:    stdu r1, -48(r1)
; CHECK-NEXT:    addi r4, r1, 44
; CHECK-NEXT:    std r0, 64(r1)
; CHECK-NEXT:    bl frexpf
; CHECK-NEXT:    nop
; CHECK-NEXT:    lwz r3, 44(r1)
; CHECK-NEXT:    addi r1, r1, 48
; CHECK-NEXT:    ld r0, 16(r1)
; CHECK-NEXT:    mtlr r0
; CHECK-NEXT:    blr
  %result = call { float, i32 } @llvm.frexp.f32.i32(float %a)
  ret { float, i32 } %result
}

define float @test_frexp_f32_i32_only_use_fract(float %a) nounwind {
; CHECK-LABEL: test_frexp_f32_i32_only_use_fract:
; CHECK:       # %bb.0:
; CHECK-NEXT:    mflr r0
; CHECK-NEXT:    stdu r1, -48(r1)
; CHECK-NEXT:    addi r4, r1, 44
; CHECK-NEXT:    std r0, 64(r1)
; CHECK-NEXT:    bl frexpf
; CHECK-NEXT:    nop
; CHECK-NEXT:    addi r1, r1, 48
; CHECK-NEXT:    ld r0, 16(r1)
; CHECK-NEXT:    mtlr r0
; CHECK-NEXT:    blr
  %result = call { float, i32 } @llvm.frexp.f32.i32(float %a)
  %result.0 = extractvalue { float, i32 } %result, 0
  ret float %result.0
}

define i32 @test_frexp_f32_i32_only_use_exp(float %a) nounwind {
; CHECK-LABEL: test_frexp_f32_i32_only_use_exp:
; CHECK:       # %bb.0:
; CHECK-NEXT:    mflr r0
; CHECK-NEXT:    stdu r1, -48(r1)
; CHECK-NEXT:    addi r4, r1, 44
; CHECK-NEXT:    std r0, 64(r1)
; CHECK-NEXT:    bl frexpf
; CHECK-NEXT:    nop
; CHECK-NEXT:    lwz r3, 44(r1)
; CHECK-NEXT:    addi r1, r1, 48
; CHECK-NEXT:    ld r0, 16(r1)
; CHECK-NEXT:    mtlr r0
; CHECK-NEXT:    blr
  %result = call { float, i32 } @llvm.frexp.f32.i32(float %a)
  %result.0 = extractvalue { float, i32 } %result, 1
  ret i32 %result.0
}

; FIXME
; define { <2 x float>, <2 x i32> } @test_frexp_v2f32_v2i32(<2 x float> %a) nounwind {
;   %result = call { <2 x float>, <2 x i32> } @llvm.frexp.v2f32.v2i32(<2 x float> %a)
;   ret { <2 x float>, <2 x i32> } %result
; }

; define <2 x float> @test_frexp_v2f32_v2i32_only_use_fract(<2 x float> %a) nounwind {
;   %result = call { <2 x float>, <2 x i32> } @llvm.frexp.v2f32.v2i32(<2 x float> %a)
;   %result.0 = extractvalue { <2 x float>, <2 x i32> } %result, 0
;   ret <2 x float> %result.0
; }

; define <2 x i32> @test_frexp_v2f32_v2i32_only_use_exp(<2 x float> %a) nounwind {
;   %result = call { <2 x float>, <2 x i32> } @llvm.frexp.v2f32.v2i32(<2 x float> %a)
;   %result.1 = extractvalue { <2 x float>, <2 x i32> } %result, 1
;   ret <2 x i32> %result.1
; }

define { double, i32 } @test_frexp_f64_i32(double %a) nounwind {
; CHECK-LABEL: test_frexp_f64_i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    mflr r0
; CHECK-NEXT:    stdu r1, -48(r1)
; CHECK-NEXT:    addi r4, r1, 44
; CHECK-NEXT:    std r0, 64(r1)
; CHECK-NEXT:    bl frexp
; CHECK-NEXT:    nop
; CHECK-NEXT:    lwz r3, 44(r1)
; CHECK-NEXT:    addi r1, r1, 48
; CHECK-NEXT:    ld r0, 16(r1)
; CHECK-NEXT:    mtlr r0
; CHECK-NEXT:    blr
  %result = call { double, i32 } @llvm.frexp.f64.i32(double %a)
  ret { double, i32 } %result
}

define double @test_frexp_f64_i32_only_use_fract(double %a) nounwind {
; CHECK-LABEL: test_frexp_f64_i32_only_use_fract:
; CHECK:       # %bb.0:
; CHECK-NEXT:    mflr r0
; CHECK-NEXT:    stdu r1, -48(r1)
; CHECK-NEXT:    addi r4, r1, 44
; CHECK-NEXT:    std r0, 64(r1)
; CHECK-NEXT:    bl frexp
; CHECK-NEXT:    nop
; CHECK-NEXT:    addi r1, r1, 48
; CHECK-NEXT:    ld r0, 16(r1)
; CHECK-NEXT:    mtlr r0
; CHECK-NEXT:    blr
  %result = call { double, i32 } @llvm.frexp.f64.i32(double %a)
  %result.0 = extractvalue { double, i32 } %result, 0
  ret double %result.0
}

define i32 @test_frexp_f64_i32_only_use_exp(double %a) nounwind {
; CHECK-LABEL: test_frexp_f64_i32_only_use_exp:
; CHECK:       # %bb.0:
; CHECK-NEXT:    mflr r0
; CHECK-NEXT:    stdu r1, -48(r1)
; CHECK-NEXT:    addi r4, r1, 44
; CHECK-NEXT:    std r0, 64(r1)
; CHECK-NEXT:    bl frexp
; CHECK-NEXT:    nop
; CHECK-NEXT:    lwz r3, 44(r1)
; CHECK-NEXT:    addi r1, r1, 48
; CHECK-NEXT:    ld r0, 16(r1)
; CHECK-NEXT:    mtlr r0
; CHECK-NEXT:    blr
  %result = call { double, i32 } @llvm.frexp.f64.i32(double %a)
  %result.0 = extractvalue { double, i32 } %result, 1
  ret i32 %result.0
}

; FIXME: Widen vector result
; define { <2 x double>, <2 x i32> } @test_frexp_v2f64_v2i32(<2 x double> %a) nounwind {
;   %result = call { <2 x double>, <2 x i32> } @llvm.frexp.v2f64.v2i32(<2 x double> %a)
;   ret { <2 x double>, <2 x i32> } %result
; }

; define <2 x double> @test_frexp_v2f64_v2i32_only_use_fract(<2 x double> %a) nounwind {
;   %result = call { <2 x double>, <2 x i32> } @llvm.frexp.v2f64.v2i32(<2 x double> %a)
;   %result.0 = extractvalue { <2 x double>, <2 x i32> } %result, 0
;   ret <2 x double> %result.0
; }

; define <2 x i32> @test_frexp_v2f64_v2i32_only_use_exp(<2 x double> %a) nounwind {
;   %result = call { <2 x double>, <2 x i32> } @llvm.frexp.v2f64.v2i32(<2 x double> %a)
;   %result.1 = extractvalue { <2 x double>, <2 x i32> } %result, 1
;   ret <2 x i32> %result.1
; }

; FIXME: f128 ExpandFloatResult
; define { ppc_fp128, i32 } @test_frexp_f128_i32(ppc_fp128 %a) nounwind {
;   %result = call { ppc_fp128, i32 } @llvm.frexp.f128.i32(ppc_fp128 %a)
;   ret { ppc_fp128, i32 } %result
; }

; define ppc_fp128 @test_frexp_f128_i32_only_use_fract(ppc_fp128 %a) nounwind {
;   %result = call { ppc_fp128, i32 } @llvm.frexp.f128.i32(ppc_fp128 %a)
;   %result.0 = extractvalue { ppc_fp128, i32 } %result, 0
;   ret ppc_fp128 %result.0
; }

; define i32 @test_frexp_f128_i32_only_use_exp(ppc_fp128 %a) nounwind {
;   %result = call { ppc_fp128, i32 } @llvm.frexp.f128.i32(ppc_fp128 %a)
;   %result.0 = extractvalue { ppc_fp128, i32 } %result, 1
;   ret i32 %result.0
; }

declare { float, i32 } @llvm.frexp.f32.i32(float) #0
declare { <2 x float>, <2 x i32> } @llvm.frexp.v2f32.v2i32(<2 x float>) #0

declare { half, i32 } @llvm.frexp.f16.i32(half) #0
declare { <2 x half>, <2 x i32> } @llvm.frexp.v2f16.v2i32(<2 x half>) #0

declare { double, i32 } @llvm.frexp.f64.i32(double) #0
declare { <2 x double>, <2 x i32> } @llvm.frexp.v2f64.v2i32(<2 x double>) #0

declare { half, i16 } @llvm.frexp.f16.i16(half) #0
declare { <2 x half>, <2 x i16> } @llvm.frexp.v2f16.v2i16(<2 x half>) #0

declare { ppc_fp128, i32 } @llvm.frexp.f128.i32(ppc_fp128) #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
