/////////////////////////////////////////////////////////////////////////////////////////
// This code contains NVIDIA Confidential Information and is disclosed
// under the Mutual Non-Disclosure Agreement.
//
// Notice
// ALL NVIDIA DESIGN SPECIFICATIONS AND CODE ("MATERIALS") ARE PROVIDED "AS IS" NVIDIA MAKES
// NO REPRESENTATIONS, WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO
// THE MATERIALS, AND EXPRESSLY DISCLAIMS ANY IMPLIED WARRANTIES OF NONINFRINGEMENT,
// MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
//
// NVIDIA Corporation assumes no responsibility for the consequences of use of such
// information or for any infringement of patents or other rights of third parties that may
// result from its use. No license is granted by implication or otherwise under any patent
// or patent rights of NVIDIA Corporation. No third party distribution is allowed unless
// expressly authorized by NVIDIA.  Details are subject to change without notice.
// This code supersedes and replaces all information previously supplied.
// NVIDIA Corporation products are not authorized for use as critical
// components in life support devices or systems without express written approval of
// NVIDIA Corporation.
//
// Copyright (c) 2015-2018 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and proprietary
// rights in and to this software and related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement from NVIDIA Corporation is
// strictly prohibited.
//
/////////////////////////////////////////////////////////////////////////////////////////

#include <cuda.h>
#include <stdexcept>
#include <dw/core/Types.h>

__constant__  float kStdDev[3] = {0.229f, 0.224f, 0.225f};

__global__ void applySTDResnet50(float32_t* image, const uint32_t width, const uint32_t height)
{
    const uint32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t tidy = blockDim.y * blockIdx.y + threadIdx.y;
    if (tidx >= width || tidy >= height) return;
    // calc CHW location for applying std dev value
    int dstIdxR = 0 * height * width + tidy * width + tidx;
    int dstIdxG = 1 * height * width + tidy * width + tidx;
    int dstIdxB = 2 * height * width + tidy * width + tidx;
	image[dstIdxR] /= kStdDev[0];
	image[dstIdxG] /= kStdDev[1];
	image[dstIdxB] /= kStdDev[2];
}

uint32_t iDivUp(const uint32_t a, const uint32_t b)
{
    return ((a % b) != 0U) ? ((a / b) + 1U) : (a / b);
}

void applyResNet50StdevToDeviceImage(float32_t *deviceCHWImageData, const uint32_t width, const uint32_t height, cudaStream_t stream)
{
	dim3 numThreads = dim3(32, 4, 1);
	applySTDResnet50 <<<dim3(iDivUp(width, numThreads.x),
				   iDivUp(height, numThreads.y)),
			numThreads, 0, stream >>>(deviceCHWImageData, width, height);
	cudaStreamSynchronize(stream);
	auto result = cudaGetLastError();
	if(result != cudaSuccess)
	{
		throw std::runtime_error(std::string("CUDA Error ")
								+ cudaGetErrorString(result)
								+ std::string(" executing CUDA function:\n " "applyStdDevToDeviceImage")
								+ std::string("\n at " __FILE__ ":") + std::to_string(__LINE__));
	}
}

