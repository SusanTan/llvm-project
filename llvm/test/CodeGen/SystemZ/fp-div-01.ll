; Test 32-bit floating-point division.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 \
; RUN:   | FileCheck -check-prefix=CHECK -check-prefix=CHECK-SCALAR %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

declare float @foo()

; Check register division.
define half @f0(half %f1, half %f2) {
; CHECK-LABEL: f0:
; CHECK: brasl %r14, __extendhfsf2@PLT
; CHECK: brasl %r14, __extendhfsf2@PLT
; CHECK: debr %f0, %f9
; CHECK: brasl %r14, __truncsfhf2@PLT
; CHECK: br %r14
  %res = fdiv half %f1, %f2
  ret half %res
}

; Check register division.
define float @f1(float %f1, float %f2) {
; CHECK-LABEL: f1:
; CHECK: debr %f0, %f2
; CHECK: br %r14
  %res = fdiv float %f1, %f2
  ret float %res
}

; Check the low end of the DEB range.
define float @f2(float %f1, ptr %ptr) {
; CHECK-LABEL: f2:
; CHECK: deb %f0, 0(%r2)
; CHECK: br %r14
  %f2 = load float, ptr %ptr
  %res = fdiv float %f1, %f2
  ret float %res
}

; Check the high end of the aligned DEB range.
define float @f3(float %f1, ptr %base) {
; CHECK-LABEL: f3:
; CHECK: deb %f0, 4092(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, ptr %base, i64 1023
  %f2 = load float, ptr %ptr
  %res = fdiv float %f1, %f2
  ret float %res
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define float @f4(float %f1, ptr %base) {
; CHECK-LABEL: f4:
; CHECK: aghi %r2, 4096
; CHECK: deb %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, ptr %base, i64 1024
  %f2 = load float, ptr %ptr
  %res = fdiv float %f1, %f2
  ret float %res
}

; Check negative displacements, which also need separate address logic.
define float @f5(float %f1, ptr %base) {
; CHECK-LABEL: f5:
; CHECK: aghi %r2, -4
; CHECK: deb %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, ptr %base, i64 -1
  %f2 = load float, ptr %ptr
  %res = fdiv float %f1, %f2
  ret float %res
}

; Check that DEB allows indices.
define float @f6(float %f1, ptr %base, i64 %index) {
; CHECK-LABEL: f6:
; CHECK: sllg %r1, %r3, 2
; CHECK: deb %f0, 400(%r1,%r2)
; CHECK: br %r14
  %ptr1 = getelementptr float, ptr %base, i64 %index
  %ptr2 = getelementptr float, ptr %ptr1, i64 100
  %f2 = load float, ptr %ptr2
  %res = fdiv float %f1, %f2
  ret float %res
}

; Check that divisions of spilled values can use DEB rather than DEBR.
define float @f7(ptr %ptr0) {
; CHECK-LABEL: f7:
; CHECK: brasl %r14, foo@PLT
; CHECK-SCALAR: deb %f0, 16{{[04]}}(%r15)
; CHECK: br %r14
  %ptr1 = getelementptr float, ptr %ptr0, i64 2
  %ptr2 = getelementptr float, ptr %ptr0, i64 4
  %ptr3 = getelementptr float, ptr %ptr0, i64 6
  %ptr4 = getelementptr float, ptr %ptr0, i64 8
  %ptr5 = getelementptr float, ptr %ptr0, i64 10
  %ptr6 = getelementptr float, ptr %ptr0, i64 12
  %ptr7 = getelementptr float, ptr %ptr0, i64 14
  %ptr8 = getelementptr float, ptr %ptr0, i64 16
  %ptr9 = getelementptr float, ptr %ptr0, i64 18
  %ptr10 = getelementptr float, ptr %ptr0, i64 20

  %val0 = load float, ptr %ptr0
  %val1 = load float, ptr %ptr1
  %val2 = load float, ptr %ptr2
  %val3 = load float, ptr %ptr3
  %val4 = load float, ptr %ptr4
  %val5 = load float, ptr %ptr5
  %val6 = load float, ptr %ptr6
  %val7 = load float, ptr %ptr7
  %val8 = load float, ptr %ptr8
  %val9 = load float, ptr %ptr9
  %val10 = load float, ptr %ptr10

  %ret = call float @foo()

  %div0 = fdiv float %ret, %val0
  %div1 = fdiv float %div0, %val1
  %div2 = fdiv float %div1, %val2
  %div3 = fdiv float %div2, %val3
  %div4 = fdiv float %div3, %val4
  %div5 = fdiv float %div4, %val5
  %div6 = fdiv float %div5, %val6
  %div7 = fdiv float %div6, %val7
  %div8 = fdiv float %div7, %val8
  %div9 = fdiv float %div8, %val9
  %div10 = fdiv float %div9, %val10

  ret float %div10
}
