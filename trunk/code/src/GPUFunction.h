/* General functions for GPU in CUDA model
 * by Lin Cheng 2013@ConcordiaUniversity
 *    forestcheng@gmail.com
 */
#ifndef GPUFUNCTION_H
#define GPUFUNCTION_H

#include <cuda_runtime.h>
#include <driver_types.h>

//#include "defs.h"

#define CUDA_ERR do { cudaError_t error;      \
  if ((error = cudaGetLastError()) != cudaSuccess) {    \
      int device; \
      cudaGetDevice(&device); \
      printf("CUDA error on GPU %d: %s : %s, line %d\n", device, cudaGetErrorString(error), __FILE__, __LINE__); }} while(0);

typedef struct gpuinfo_s
{
	//device number
	int n_device;
	//device idx
	int* devices;
	//device property
	struct cudaDeviceProp* props;
} GPUInfo;

typedef struct Database_s
{
	uint width;
//	uint heightQuad;
	size_t allocDQ;
	uint *array;
} DatabaseQuad;

typedef struct
{
	/*fixed size */
	uint *hashQuad;
	float *hostSC;

	/*dynamic size */
	DatabaseQuad dbQuad;

} HostMemPtr;

typedef struct
{
	/*fixed size */
	uint *cudaHashQuad0;
	uint *cudaHashQuad1;
	float *cudaSC0;
	float *cudaSC1;
	uint8_t *cudaDP0;
	uint8_t *cudaDP1;

	uint4 *cudaOMrbv;

	/*dynamic size */
	uint *cudaSeqsQuad0;
	uint *cudaSeqsQuad1;

	cudaTextureObject_t tex0;
	cudaTextureObject_t tex1;

	size_t allocSQ1;
	size_t allocSQ2;

} CUDAMemPtr;

extern GPUInfo *getGPUInfo();
//extern uint *createInterDBQuad(ESL_SQ_BLOCK *block, uint width, uint heightQuad);
extern void allocCUDAFixed(HostMemPtr &hostPtr, CUDAMemPtr &cudaPtr, const P7_OPROFILE *om, uint numSeqs);
extern void loadCUDA(HostMemPtr &hostPtr, CUDAMemPtr &cudaPtr, ESL_SQ_BLOCK *block, WORKER_INFO *info, cudaStream_t *stream);
extern void freeMem(HostMemPtr &hostPtr, CUDAMemPtr &cudaPtr);
#endif
