// RUN: fir-opt %s --fir-to-memref --allow-unregistered-dialect | FileCheck %s

// Tests for fir.slice with a path component (projected component slice).
// A projected slice changes the element type of the boxed view, e.g.
// z%re projects complex<f32> -> f32.
//
// FIRToMemRef has two lowering strategies for projected complex (%re / %im):
//
//   Box/rebox (embox, array_coor on !fir.box): the Fortran descriptor already
//   describes the real-component view (element type, extents, byte strides).
//   Lowering uses fir.box_dims + memref.reinterpret_cast on a memref whose
//   element type matches the box (e.g. memref<4xf32>), then memref.load/store
//   with a single index per dimension.  No ...x2xT dimension is introduced.
//
//   Ref-backed array_coor (no box on the coor operand): storage remains
//   memref<...xcomplex<T>>; lowering reinterprets as memref<...x2xT> and appends
//   the component index (0=re, 1=im) as the final memref index.
//
// Derived from:
//   complex, target :: z(4) = 0.
//   real, pointer  :: r(:)
//   r => z%re
//   r = r + z(4:1:-1)%re

// ----------------------------------------------------------------------------
// Forward projected slice load: z(1:4:1)%re
//
// Fortran: z(1:4:1)%re with step=1, lb=1.
// FIR: fir.embox with path %c0 (%re) -> !fir.box<!fir.array<4xf32>>,
//      fir.array_coor %embox %i.
//
// The fir.convert and box metadata appear inside the loop body (insertion point
// tracks the array_coor).  elemIdx = (i - 1) * step + (lb - 1) = i - 1.
// Indices are reversed (column-major -> row-major) but for 1D that is a no-op.
// ----------------------------------------------------------------------------
// CHECK-LABEL: func.func @projected_slice_fwd
// CHECK:       fir.do_loop [[I:%.*]] =
// CHECK:         [[CONVERT:%[0-9]+]] = fir.convert %arg0 : (!fir.ref<!fir.array<4xcomplex<f32>>>) -> memref<4xf32>
// CHECK:         arith.subi [[I]], {{.*}} : index
// CHECK:         arith.muli
// CHECK:         [[IDX:%[0-9]+]] = arith.addi {{.*}} : index
// CHECK:         [[ELSIZE:%[0-9]+]] = fir.box_elesize {{.*}} : (!fir.box<!fir.array<4xf32>>) -> index
// CHECK:         [[DIMS:%[0-9]+]]:3 = fir.box_dims {{.*}} : (!fir.box<!fir.array<4xf32>>, index) -> (index, index, index)
// CHECK:         [[STRIDE:%[0-9]+]] = arith.divsi [[DIMS]]#2, [[ELSIZE]] : index
// CHECK:         [[REINT:%.+]] = memref.reinterpret_cast [[CONVERT]] to offset: {{.*}}, sizes: [[[DIMS]]#1], strides: [[[STRIDE]]] : memref<4xf32> to memref<?xf32, strided<[?], offset: ?>>
// CHECK:         memref.load [[REINT]][[[IDX]]] : memref<?xf32, strided<[?], offset: ?>>
// CHECK-NOT:     memref<{{.*}}x2xf32>
func.func @projected_slice_fwd(%arg0: !fir.ref<!fir.array<4xcomplex<f32>>>) {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %shape = fir.shape %c4 : (index) -> !fir.shape<1>
  %slice = fir.slice %c1, %c4, %c1 path %c0 : (index, index, index, index) -> !fir.slice<1>
  %embox = fir.embox %arg0(%shape) [%slice] : (!fir.ref<!fir.array<4xcomplex<f32>>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<4xf32>>
  fir.do_loop %i = %c1 to %c4 step %c1 unordered {
    %coor = fir.array_coor %embox %i : (!fir.box<!fir.array<4xf32>>, index) -> !fir.ref<f32>
    %val = fir.load %coor : !fir.ref<f32>
  }
  return
}

// ----------------------------------------------------------------------------
// Backward projected slice load: z(4:1:-1)%re
//
// Fortran: step = -1, lb = 4  ->  elemIdx = (i - 1) * (-1) + (4 - 1) = 3 - (i-1).
// Same box_dims path as forward; only the index arithmetic differs.
// ----------------------------------------------------------------------------
// CHECK-LABEL: func.func @projected_slice_bwd
// CHECK:       fir.do_loop [[I:%.*]] =
// CHECK:         fir.convert %arg0 : (!fir.ref<!fir.array<4xcomplex<f32>>>) -> memref<4xf32>
// CHECK:         arith.subi [[I]], {{.*}} : index
// CHECK:         arith.muli
// CHECK:         [[IDX:%[0-9]+]] = arith.addi {{.*}} : index
// CHECK:         fir.box_elesize
// CHECK:         fir.box_dims
// CHECK:         arith.divsi
// CHECK:         memref.reinterpret_cast {{.*}} : memref<4xf32> to memref<?xf32, strided<[?], offset: ?>>
// CHECK:         memref.load {{.*}}[[[IDX]]] : memref<?xf32, strided<[?], offset: ?>>
// CHECK-NOT:     memref<{{.*}}x2xf32>
func.func @projected_slice_bwd(%arg0: !fir.ref<!fir.array<4xcomplex<f32>>>) {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cm1 = arith.constant -1 : index
  %c0 = arith.constant 0 : index
  %shape = fir.shape %c4 : (index) -> !fir.shape<1>
  %slice = fir.slice %c4, %c1, %cm1 path %c0 : (index, index, index, index) -> !fir.slice<1>
  %embox = fir.embox %arg0(%shape) [%slice] : (!fir.ref<!fir.array<4xcomplex<f32>>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<4xf32>>
  fir.do_loop %i = %c1 to %c4 step %c1 unordered {
    %coor = fir.array_coor %embox %i : (!fir.box<!fir.array<4xf32>>, index) -> !fir.ref<f32>
    %val = fir.load %coor : !fir.ref<f32>
  }
  return
}

// ----------------------------------------------------------------------------
// Imaginary component store: z(1:4:1)%im = val
//
// FIR: fir.slice path %c1_im selects the imaginary part; embox yields
// !fir.box<!fir.array<4xf32>> with the box base offset already on %im.
// Direct scalar store — no read-modify-write, no complex.create.
// ----------------------------------------------------------------------------
// CHECK-LABEL: func.func @projected_slice_store_im
// CHECK:       fir.do_loop [[I:%.*]] =
// CHECK:         fir.convert %arg0 : (!fir.ref<!fir.array<4xcomplex<f32>>>) -> memref<4xf32>
// CHECK:         arith.subi [[I]], {{.*}} : index
// CHECK:         arith.muli
// CHECK:         [[IDX:%[0-9]+]] = arith.addi {{.*}} : index
// CHECK:         fir.box_elesize
// CHECK:         fir.box_dims
// CHECK:         arith.divsi
// CHECK:         memref.reinterpret_cast {{.*}} : memref<4xf32> to memref<?xf32, strided<[?], offset: ?>>
// CHECK:         memref.store %arg1, {{.*}}[[[IDX]]] : memref<?xf32, strided<[?], offset: ?>>
// CHECK-NOT:     memref<{{.*}}x2xf32>
func.func @projected_slice_store_im(%arg0: !fir.ref<!fir.array<4xcomplex<f32>>>,
                                    %arg1: f32) {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c1_im = arith.constant 1 : index  
  %shape = fir.shape %c4 : (index) -> !fir.shape<1>
  %slice = fir.slice %c1, %c4, %c1 path %c1_im : (index, index, index, index) -> !fir.slice<1>
  %embox = fir.embox %arg0(%shape) [%slice] : (!fir.ref<!fir.array<4xcomplex<f32>>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<4xf32>>
  fir.do_loop %i = %c1 to %c4 step %c1 unordered {
    %coor = fir.array_coor %embox %i : (!fir.box<!fir.array<4xf32>>, index) -> !fir.ref<f32>
    fir.store %arg1 to %coor : !fir.ref<f32>
  }
  return
}

// ----------------------------------------------------------------------------
// Ref-backed projected slice load: array_coor on !fir.ref (no embox)
//
// Flang normally lowers z%re via embox + box_dims (tests above).  This path
// covers fir.array_coor directly on the storage ref with a projected slice;
// FIRToMemRef reinterprets memref<...xcomplex<T>> as memref<...x2xT> and uses
// the component index (0=re, 1=im) as the trailing memref dimension.
// ----------------------------------------------------------------------------
// CHECK-LABEL: func.func @projected_slice_ref_re
// CHECK:       fir.do_loop [[I:%.*]] =
// CHECK:         [[BASE:%[0-9]+]] = fir.convert %arg0 : (!fir.ref<!fir.array<4xcomplex<f32>>>) -> memref<4xcomplex<f32>>
// CHECK:         arith.subi [[I]], {{.*}} : index
// CHECK:         arith.muli
// CHECK:         [[IDX:%[0-9]+]] = arith.addi {{.*}} : index
// CHECK:         [[VIEW:%[0-9]+]] = fir.convert [[BASE]] : (memref<4xcomplex<f32>>) -> memref<4x2xf32>
// CHECK:         memref.load [[VIEW]][[[IDX]], {{.*}}] : memref<4x2xf32>
// CHECK-NOT:     fir.box_dims
// CHECK-NOT:     memref.reinterpret_cast
func.func @projected_slice_ref_re(%arg0: !fir.ref<!fir.array<4xcomplex<f32>>>) {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %shape = fir.shape %c4 : (index) -> !fir.shape<1>
  %slice = fir.slice %c1, %c4, %c1 path %c0 : (index, index, index, index) -> !fir.slice<1>
  fir.do_loop %i = %c1 to %c4 step %c1 unordered {
    %coor = fir.array_coor %arg0(%shape) [%slice] %i : (!fir.ref<!fir.array<4xcomplex<f32>>>, !fir.shape<1>, !fir.slice<1>, index) -> !fir.ref<f32>
    %val = fir.load %coor : !fir.ref<f32>
  }
  return
}

// CHECK-LABEL: func.func @projected_slice_ref_im
// CHECK:       fir.do_loop [[I:%.*]] =
// CHECK:         [[BASE:%[0-9]+]] = fir.convert %arg0 : (!fir.ref<!fir.array<4xcomplex<f32>>>) -> memref<4xcomplex<f32>>
// CHECK:         arith.subi [[I]], {{.*}} : index
// CHECK:         arith.muli
// CHECK:         [[IDX:%[0-9]+]] = arith.addi {{.*}} : index
// CHECK:         [[VIEW:%[0-9]+]] = fir.convert [[BASE]] : (memref<4xcomplex<f32>>) -> memref<4x2xf32>
// CHECK:         memref.store %arg1, [[VIEW]][[[IDX]], {{.*}}] : memref<4x2xf32>
// CHECK-NOT:     fir.box_dims
// CHECK-NOT:     memref.reinterpret_cast
func.func @projected_slice_ref_im(%arg0: !fir.ref<!fir.array<4xcomplex<f32>>>,
                                  %arg1: f32) {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c1_im = arith.constant 1 : index
  %shape = fir.shape %c4 : (index) -> !fir.shape<1>
  %slice = fir.slice %c1, %c4, %c1 path %c1_im : (index, index, index, index) -> !fir.slice<1>
  fir.do_loop %i = %c1 to %c4 step %c1 unordered {
    %coor = fir.array_coor %arg0(%shape) [%slice] %i : (!fir.ref<!fir.array<4xcomplex<f32>>>, !fir.shape<1>, !fir.slice<1>, index) -> !fir.ref<f32>
    fir.store %arg1 to %coor : !fir.ref<f32>
  }
  return
}

// ----------------------------------------------------------------------------
// 2-D boxed projected slice load: z(1:2:1, 1:3:1)%re
//
// Storage: !fir.array<2x3xcomplex<f32>> (Fortran column-major).
//
// convertMemrefType reverses Fortran column-major extents to MLIR row-major:
//   !fir.ref<!fir.array<2x3xcomplex<f32>>> -> memref<3x2xcomplex<f32>>
// getFIRConvert for projected embox uses the box element type:
//   memref<3x2xcomplex<f32>> storage ref -> memref<3x2xf32> view
//
// Per-dimension element index (0-based, column-major):
//   elemIdx_i = (i-1)*1 + (1-1) = i-1   (Fortran dim 1, size 2)
//   elemIdx_j = (j-1)*1 + (1-1) = j-1   (Fortran dim 2, size 3)
//
// After reversing for MLIR row-major access:
//   memref.load [elemIdx_j, elemIdx_i]
//
// Strides for both dimensions come from fir.box_dims on %embox (not synthesized
// from fir.shape extents).
// ----------------------------------------------------------------------------
// CHECK-LABEL: func.func @projected_slice_2d
// CHECK:       fir.do_loop [[I:%.*]] =
// CHECK:         fir.do_loop [[J:%.*]] =
// CHECK:           fir.convert %arg0 : (!fir.ref<!fir.array<2x3xcomplex<f32>>>) -> memref<3x2xf32>
// CHECK:           arith.subi [[I]], {{.*}} : index
// CHECK:           arith.muli
// CHECK:           [[IDX_I:%[0-9]+]] = arith.addi {{.*}} : index
// CHECK:           arith.subi [[J]], {{.*}} : index
// CHECK:           arith.muli
// CHECK:           [[IDX_J:%[0-9]+]] = arith.addi {{.*}} : index
// CHECK:           fir.box_elesize
// CHECK:           fir.box_dims
// CHECK:           arith.divsi
// CHECK:           fir.box_dims
// CHECK:           arith.divsi
// CHECK:           memref.reinterpret_cast {{.*}} : memref<3x2xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
// CHECK:           memref.load {{.*}}[[[IDX_J]], [[IDX_I]]] : memref<?x?xf32, strided<[?, ?], offset: ?>>
// CHECK-NOT:     memref<{{.*}}x2xf32>
func.func @projected_slice_2d(%arg0: !fir.ref<!fir.array<2x3xcomplex<f32>>>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c0 = arith.constant 0 : index
  %shape = fir.shape %c2, %c3 : (index, index) -> !fir.shape<2>
  %slice = fir.slice %c1, %c2, %c1, %c1, %c3, %c1 path %c0 : (index, index, index, index, index, index, index) -> !fir.slice<2>
  %embox = fir.embox %arg0(%shape) [%slice] : (!fir.ref<!fir.array<2x3xcomplex<f32>>>, !fir.shape<2>, !fir.slice<2>) -> !fir.box<!fir.array<2x3xf32>>
  fir.do_loop %i = %c1 to %c2 step %c1 unordered {
    fir.do_loop %j = %c1 to %c3 step %c1 unordered {
      %coor = fir.array_coor %embox %i, %j : (!fir.box<!fir.array<2x3xf32>>, index, index) -> !fir.ref<f32>
      %val = fir.load %coor : !fir.ref<f32>
    }
  }
  return
}

// ----------------------------------------------------------------------------
// Derived-type component projection: a%x where a : TYPE{x:f64, y:complex<f64>}
//
// This is NOT a complex projection — the storage element is the derived type T,
// not complex<T>.  FIRToMemRef cannot safely lower this; downstream
// FIR-to-LLVM lowering handles it correctly via the descriptor.
//
// The fir.array_coor must survive (not be erased).
// The store must remain as fir.store, not memref.store.
// ----------------------------------------------------------------------------
// CHECK-LABEL: func.func @derived_component_not_projected
// CHECK:       fir.array_coor
// CHECK:       fir.store
// CHECK-NOT:   memref.store
func.func @derived_component_not_projected(
    %arg0: !fir.ref<!fir.array<4x!fir.type<T{x:f64,y:complex<f64>}>>>) {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant 9.9e+01 : f64
  %field = fir.field_index x, !fir.type<T{x:f64,y:complex<f64>}>
  %shape = fir.shape %c4 : (index) -> !fir.shape<1>
  %slice = fir.slice %c1, %c4, %c1 path %field : (index, index, index, !fir.field) -> !fir.slice<1>
  %embox = fir.embox %arg0(%shape) [%slice] : (!fir.ref<!fir.array<4x!fir.type<T{x:f64,y:complex<f64>}>>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<4xf64>>
  fir.do_loop %i = %c1 to %c4 step %c1 unordered {
    %coor = fir.array_coor %embox %i : (!fir.box<!fir.array<4xf64>>, index) -> !fir.ref<f64>
    fir.store %cst to %coor : !fir.ref<f64>
  }
  return
}
