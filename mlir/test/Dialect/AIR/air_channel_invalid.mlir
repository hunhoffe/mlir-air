//===- air_channel_invalid.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt --split-input-file --verify-diagnostics %s

// -----

// Test: rank mismatch between size and broadcast_shape.
// expected-error @+1 {{'air.channel' op bundle rank should match broadcast_shape rank}}
air.channel @rank_mismatch [2, 2] {broadcast_shape = [4, 4, 4]}

// -----

// Test: valid broadcasting (positive test - should pass with no errors).
air.channel @valid_broadcast [1, 4] {broadcast_shape = [4, 4]}

// -----

// Test: valid 3D broadcasting (positive test).
air.channel @valid_3d [2, 1, 4] {broadcast_shape = [2, 4, 4]}

// -----

// Test: valid all-ones size broadcasting (positive test).
air.channel @valid_all_ones [1, 1] {broadcast_shape = [4, 4]}
