//===- llvm/CodeGen/GlobalISel/InstructionSelector.h ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file declares the API for the instruction selector.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_GLOBALISEL_INSTRUCTIONSELECTOR_H
#define LLVM_CODEGEN_GLOBALISEL_INSTRUCTIONSELECTOR_H

#include "llvm/CodeGen/GlobalISel/GIMatchTableExecutor.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
class GISelObserverWrapper;

class LLVM_ABI InstructionSelector : public GIMatchTableExecutor {
public:
  virtual ~InstructionSelector();

  /// Select the (possibly generic) instruction \p I to only use target-specific
  /// opcodes. It is OK to insert multiple instructions, but they cannot be
  /// generic pre-isel instructions.
  ///
  /// \returns whether selection succeeded.
  /// \pre  I.getParent() && I.getParent()->getParent()
  /// \post
  ///   if returns true:
  ///     for I in all mutated/inserted instructions:
  ///       !isPreISelGenericOpcode(I.getOpcode())
  virtual bool select(MachineInstr &I) = 0;

  // FIXME: Eliminate dependency on TargetPassConfig for NewPM transition
  const TargetPassConfig *TPC = nullptr;

  MachineOptimizationRemarkEmitter *MORE = nullptr;

  /// Note: InstructionSelect does not track changed instructions.
  /// changingInstr() and changedInstr() will never be called on these
  /// observers.
  GISelObserverWrapper *AllObservers = nullptr;
};
} // namespace llvm

#endif
