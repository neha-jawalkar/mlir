// RUN: mlir-opt -split-input-file -simplify-affine-structures %s | FileCheck %s

// CHECK-DAG: #map{{[0-9]+}} = (d0) -> (d0 mod 4)
#map = (d0) -> (d0 - (d0 floordiv 4) * 4) 
 
func @f(%A: memref<?x?xf32>)
{ 
	%N = dim %A, 0 : memref<?x?xf32> 
	affine.for %i = 0 to %N step 1 
	{ 
		%0 = affine.apply #map(%i)
	}
	return 
}
// -----


// CHECK-DAG: #map{{[0-9]+}} = (d0) -> ((d0 + 1) mod 4)
#map = (d0) -> (d0  - ( (d0 + 1) floordiv 4) * 4 + 1)
 
func @f(%A: memref<?x?xf32>)
{ 
	%N = dim %A, 0 : memref<?x?xf32> 
	affine.for %i = 0 to %N step 1 
	{ 
		%0 = affine.apply #map(%i)
	}
	return 
}
// -----


// CHECK-DAG: #map{{[0-9]+}} = (d0, d1) -> ((d0 + d1 * 2) mod 4)
#map = (d0, d1) -> (d1 + d0  - ( (d0 + 2 * d1) floordiv 4) * 4 + d1)
 
func @f(%A: memref<?x?xf32>)
{ 
	%N = dim %A, 0 : memref<?x?xf32> 
	affine.for %i = 0 to %N step 1 
	{ 
		%0 = affine.apply #map(%i, %i)
	}
	return 
}
// -----


// CHECK-DAG: #map{{[0-9]+}} = (d0)[s0] -> ((d0 + s0 * 2) mod 2)
#map = (d0)[s0] -> (d0 + 2 * s0  - ( (d0 + 2 * s0) floordiv 2) * 2)
 
func @f(%A: memref<?x?xf32>)
{ 
	%N = dim %A, 0 : memref<?x?xf32> 
	affine.for %i = 0 to %N step 1 
	{ 
		%0 = affine.apply #map(%i)[%N]
	}
	return 
}
// -----


// CHECK-DAG: #map{{[0-9]+}} = (d0) -> (d0 mod 8)
#map = (d0) -> (- (d0 floordiv 8) * 8 + d0)
 
func @f(%A: memref<?x?xf32>)
{ 
	%N = dim %A, 0 : memref<?x?xf32> 
	affine.for %i = 0 to %N step 1 
	{ 
		%0 = affine.apply #map(%i)
	}
	return 
}
// -----


// CHECK-DAG: #map{{[0-9]+}} = (d0) -> ((d0 + 1) mod 4 - 2)
#map = (d0) -> (d0  - ( (d0 + 1) floordiv 4) * 4 - 1)
 
func @f(%A: memref<?x?xf32>)
{ 
	%N = dim %A, 0 : memref<?x?xf32> 
	affine.for %i = 0 to %N step 1 
	{ 
		%0 = affine.apply #map(%i)
	}
	return 
}
// -----


// CHECK-DAG: #map{{[0-9]+}} = (d0) -> (((d0 + 1) mod 16) floordiv 8)
#map = (d0) -> ((d0 - ((d0 + 1) floordiv 16) * 16 + 1) floordiv 8)
 
func @f(%A: memref<?x?xf32>)
{ 
	%N = dim %A, 0 : memref<?x?xf32> 
	affine.for %i = 0 to %N step 1 
	{ 
		%0 = affine.apply #map(%i)
	}
	return 
}
// -----


// CHECK-DAG: #map{{[0-9]+}} = (d0) -> ((d0 floordiv 16) * -16)

// This should not simplify to anything.
#map = (d0) -> ( -(d0 floordiv 16) * 16)
 
func @f(%A: memref<?x?xf32>)
{ 
	%N = dim %A, 0 : memref<?x?xf32> 
	affine.for %i = 0 to %N step 1 
	{ 
		%0 = affine.apply #map(%i)
	}
	return 
}
// -----

