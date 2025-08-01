//===-- Half-precision asinh(x) function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception.
//
//===----------------------------------------------------------------------===//

#include "src/math/asinhf16.h"
#include "src/__support/math/asinhf16.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float16, asinhf16, (float16 x)) { return math::asinhf16(x); }

} // namespace LIBC_NAMESPACE_DECL
