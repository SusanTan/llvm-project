// RUN: fir-opt %s --fir-to-memref --allow-unregistered-dialect | FileCheck %s
// CHECK-LABEL: func.func @load_scalar
// CHECK:       [[DUMMY:%0]] = fir.undefined !fir.dscope
// CHECK-NEXT:  [[DECLARE:%[0-9]+]] = fir.declare %arg0 dummy_scope [[DUMMY]]
// CHECK-NEXT:  [[CONVERT:%[0-9]+]] = fir.convert [[DECLARE]] : (!fir.ref<!fir.logical<4>>) -> memref<i32>
// CHECK-NEXT:  [[LOAD:%[0-9]+]] = memref.load [[CONVERT]][] : memref<i32>
// CHECK-NEXT:  fir.convert [[LOAD]] : (i32) -> !fir.logical<4>
func.func @load_scalar(%arg0: !fir.ref<!fir.logical<4>>) {
  %0 = fir.undefined !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 {uniq_name = "a"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> !fir.ref<!fir.logical<4>>
  %2 = fir.load %1 : !fir.ref<!fir.logical<4>>
  return
}

// CHECK-LABEL: func.func @store_scalar
// CHECK:       [[CONSTTRUE:%.+]] = arith.constant true
// CHECK:       [[DUMMY:%0]] = fir.undefined !fir.dscope
// CHECK:       [[DECLARE:%[0-9]+]] = fir.declare %arg0 dummy_scope [[DUMMY]]
// CHECK-NEXT:  [[CONVERT:%[0-9]+]] = fir.convert [[CONSTTRUE]] : (i1) -> !fir.logical<4>
// CHECK-NEXT:  [[CONVERT1:%[0-9]+]] = fir.convert [[DECLARE]] : (!fir.ref<!fir.logical<4>>) -> memref<i32>
// CHECK-NEXT:  [[INT:%[0-9]+]] = fir.convert [[CONVERT]] : (!fir.logical<4>) -> i32
// CHECK-NEXT:  memref.store [[INT]], [[CONVERT1]][] : memref<i32>
func.func @store_scalar(%arg0: !fir.ref<!fir.logical<4>>) {
  %true = arith.constant true
  %0 = fir.undefined !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 {uniq_name = "b"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> !fir.ref<!fir.logical<4>>
  %2 = fir.convert %true : (i1) -> !fir.logical<4>
  fir.store %2 to %1 : !fir.ref<!fir.logical<4>>
  return
}

// CHECK-LABEL: func.func @store_loaded_logical
// CHECK:       [[DUMMY:%[0-9]+]] = fir.undefined !fir.dscope
// CHECK:       [[SRC_DECLARE:%[0-9]+]] = fir.declare %arg0 dummy_scope [[DUMMY]]
// CHECK:       [[DST_DECLARE:%[0-9]+]] = fir.declare %arg1 dummy_scope [[DUMMY]]
// CHECK:       [[SRC_MEM:%[0-9]+]] = fir.convert [[SRC_DECLARE]] : (!fir.ref<!fir.logical<4>>) -> memref<i32>
// CHECK:       [[LOAD:%[0-9]+]] = memref.load [[SRC_MEM]][] : memref<i32>
// CHECK:       [[TOLOGICAL:%[0-9]+]] = fir.convert [[LOAD]] : (i32) -> !fir.logical<4>
// CHECK:       [[DST_MEM:%[0-9]+]] = fir.convert [[DST_DECLARE]] : (!fir.ref<!fir.logical<4>>) -> memref<i32>
// CHECK-NOT:   fir.convert [[TOLOGICAL]] : (!fir.logical<4>) -> i32
// CHECK:       memref.store [[LOAD]], [[DST_MEM]][] : memref<i32>
func.func @store_loaded_logical(%arg0: !fir.ref<!fir.logical<4>>, %arg1: !fir.ref<!fir.logical<4>>) {
  %0 = fir.undefined !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 {uniq_name = "src"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> !fir.ref<!fir.logical<4>>
  %2 = fir.declare %arg1 dummy_scope %0 {uniq_name = "dst"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> !fir.ref<!fir.logical<4>>
  %3 = fir.load %1 : !fir.ref<!fir.logical<4>>
  fir.store %3 to %2 : !fir.ref<!fir.logical<4>>
  return
}
