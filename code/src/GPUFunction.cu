/* General functions for GPU in CUDA model
 * by Lin Cheng 2013@ConcordiaUniversity
 *    forestcheng@gmail.com
 */
#include "p7_config.h"

#include <cuda_runtime.h>
#include <math_functions.h>

#include <stdio.h>
#include <limits.h>

extern "C" {
#include "esl_alphabet.h"
#include "esl_msa.h"
#include "esl_sq.h"

#include "hmmer.h"
}

#ifdef HMMER_THREADS
#include <unistd.h>
extern "C" {
#include "esl_threads.h"
#include "esl_workqueue.h"
}
#endif /*HMMER_THREADS*/

#include "defs.h"
#include "GPUFunction.h"

typedef struct cudaDeviceProp CUDADeviceProp;
//typedef struct __device_builtin_texture_type__ texture CUDAtexture;

#define KERNEL_BLOCKSIZE 128
// 32KB / 16(sizeof(uint4)
#define SHARED_SIZE 2048

#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
inline void __checkCudaErrors(cudaError_t err, const char *file, const int line)
{
    if (cudaSuccess != err)
    {
        fprintf(stderr, "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, line %i.\n",
                err, cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

#define PTX_SIMD_QUAD(vop, rv, op1, op2) \
		asm(#vop".u32.u32.u32.sat %0, %1, %2, %3;" : "=r"(rv.x) : "r"(op1.x), "r"(op2), "r"(0)); \
		asm(#vop".u32.u32.u32.sat %0, %1, %2, %3;" : "=r"(rv.y) : "r"(op1.y), "r"(op2), "r"(0)); \
		asm(#vop".u32.u32.u32.sat %0, %1, %2, %3;" : "=r"(rv.z) : "r"(op1.z), "r"(op2), "r"(0)); \
		asm(#vop".u32.u32.u32.sat %0, %1, %2, %3;" : "=r"(rv.w) : "r"(op1.w), "r"(op2), "r"(0));

#define PTX_SIMD_QUADALL(vop, rv, op1, op2) \
		asm(#vop".u32.u32.u32.sat %0, %1, %2, %3;" : "=r"(rv.x) : "r"(op1.x), "r"(op2.x), "r"(0)); \
		asm(#vop".u32.u32.u32.sat %0, %1, %2, %3;" : "=r"(rv.y) : "r"(op1.y), "r"(op2.y), "r"(0)); \
		asm(#vop".u32.u32.u32.sat %0, %1, %2, %3;" : "=r"(rv.z) : "r"(op1.z), "r"(op2.z), "r"(0)); \
		asm(#vop".u32.u32.u32.sat %0, %1, %2, %3;" : "=r"(rv.w) : "r"(op1.w), "r"(op2.w), "r"(0));

#define PTX_SIMD_MAX_QUAD(rv, op1, op2) \
		asm("vmax4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(rv.x) : "r"(op1.x), "r"(op2), "r"(0)); \
		asm("vmax4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(rv.y) : "r"(op1.y), "r"(op2), "r"(0)); \
		asm("vmax4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(rv.z) : "r"(op1.z), "r"(op2), "r"(0)); \
		asm("vmax4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(rv.w) : "r"(op1.w), "r"(op2), "r"(0));

#define PTX_SIMD_MAX_QUADALL(rv, op1, op2) \
		asm("vmax4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(rv.x) : "r"(op1.x), "r"(op2.x), "r"(0)); \
		asm("vmax4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(rv.y) : "r"(op1.y), "r"(op2.y), "r"(0)); \
		asm("vmax4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(rv.z) : "r"(op1.z), "r"(op2.z), "r"(0)); \
		asm("vmax4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(rv.w) : "r"(op1.w), "r"(op2.w), "r"(0));

#define TEST_OVERFLOW(op1, op2, op3) \
		asm("vadd4.u32.u32.u32.sat %0, %1, %2, %3;" : "=r"(op1) : "r"(op2), "r"(op3), "r"(0)); \
		for (int j = 0; j < 4; ++j) { \
			if ((op1 & 0xFF) == 0xFF) { \
				bMax = true; \
			} \
			op1 >>= 8; \
		}

static __device__ __constant__ uint cudaOM[8];
static __device__ __constant__ float cudaScale_b;
cudaTextureDesc texDesc;
cudaResourceDesc resDesc0, resDesc1;

texture<uint4, cudaTextureType1D, cudaReadModeElementType> texOMrbv;


__global__ void msvSIMDKernel0(cudaTextureObject_t tex, uint *cudaHashQuad, uint numAligned, uint8_t *cudaDP, float *cudaSC, uint Num, int Q) //, uint4 *cudaOMrbv)
{
	uint idx;
	idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= Num) return;

	uint L = cudaHashQuad[idx]; // Length of sq

	int i, y;			   /* counter over sequence positions 1..L                      */
	int q;			   /* counter over vectors 0..nq-1                              */


	float sc = logf(3.0f / (float) (L+3));
	sc  = -1.0f * roundf(cudaScale_b * sc); // (om->scale_b * sc);       /* ugh. sc is now an integer cost represented in a float...    */

	uint tjb_b = (uint) sc;	/* and now we cast and saturate it to an unsigned char cost... */

	uint xJv16 = 0;		/* special states' scores    vector for states score        */

	uint4 mpv32;	/* previous row values                                       */
	uint4 xEv32;	/* E state: keeps max for Mk->E as we go                     */
	uint4 sv32;		/* temp storage of 1 curr row value in progress              */
	uint xBv32;


	uint basev16 = cudaOM[1];		/* offset for scores                                         */
//	uint8_t biasv16 = cudaOM[2];		/* emission bias in a vector                                 */
//	uint8_t tjbmv16 = cudaOM[3] + tjb_b;		/* vector for cost of moving from either J or N through B to an M state */
	tjb_b += cudaOM[3];
	uint tecv16  = cudaOM[4];		/* vector for E->C  cost                                     */

	uint biasv32 = (cudaOM[2] << 24) | (cudaOM[2] << 16) | (cudaOM[2] << 8) | (cudaOM[2]);

	xBv32 = basev16 - tjb_b;//SATURATE_SUB_UCHAR(basev16, tjb_b);

	xBv32 |= (xBv32 << 16); xBv32 |= (xBv32 << 8);

	uint4 *dp64 = (uint4 *) (cudaDP + (idx << 4));

	union { uint v; uint8_t us[4]; } u;
	for (y = 0, i = 0; i < L; ++y, i += DBSQ_LENGTH_ALIGNED) {
		tex1Dfetch(&u.v, tex, y * numAligned + idx);

		for (int k = 0; k < 4 && i + k < L; ++k) {
			xEv32.x = xEv32.y = xEv32.z = xEv32.w = 0;

			ulong2* ptr64 = (ulong2 *) (dp64 + (Q-1) * numAligned);
			ulong2 u2 = *ptr64;
	        ulong2 *tmp64 = (ulong2 *) &mpv32;
	        tmp64->x = u2.x;
	        tmp64->y = (u2.y << 8) | (tmp64->x >> 56 & 0xFF);
	        tmp64->x = tmp64->x << 8;

			for (q = 0; q < Q; q++)
			{
				PTX_SIMD_MAX_QUAD(sv32, mpv32, xBv32)

				PTX_SIMD_QUAD(vadd4, sv32, sv32, biasv32)

				uint4 rsc64 = tex1Dfetch(texOMrbv, u.us[k] * Q + q);

				PTX_SIMD_QUADALL(vsub4, sv32, sv32, rsc64);

				PTX_SIMD_MAX_QUADALL(xEv32, xEv32, sv32);
				mpv32 = *(dp64 + q * numAligned);
				*(dp64 + q * numAligned) = sv32;

			}

			/* CUDA SIMD Video instructions */
	        asm("vmax4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(xEv32.x) : "r"(xEv32.x), "r"(xEv32.y), "r"(0));
	        asm("vmax4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(xEv32.z) : "r"(xEv32.z), "r"(xEv32.w), "r"(0));
	        asm("vmax4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(xEv32.x) : "r"(xEv32.x), "r"(xEv32.z), "r"(0));
	        asm("vmax4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(xEv32.x) : "r"(xEv32.x), "r"(xEv32.x >> 16), "r"(0));
	        asm("vmax4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(xEv32.x) : "r"(xEv32.x), "r"(xEv32.x >> 8), "r"(0));

			uint tmp8 = (xEv32.x & 0xFF) - tecv16; //SATURATE_SUB_UCHAR((xEv32.x & 0xFF), tecv16);

			xJv16 = max(xJv16, tmp8);
			xBv32 = max(basev16, xJv16) - tjb_b;
			xBv32 |= (xBv32 << 16);
			xBv32 |= (xBv32 << 8);
		}
	}

	sc = ((float) (xJv16 - tjb_b + cudaOM[3]) - (float) basev16);
	sc /= cudaScale_b;
	sc -= 3.0; /* that's ~ L \log \frac{L}{L+3}, for our NN,CC,JJ */

	cudaSC[idx] = sc;
}

GPUInfo *getGPUInfo()
{
	int i;
	GPUInfo *gpuInfo = NULL;
	gpuInfo = (GPUInfo*) malloc(sizeof(GPUInfo));

	if (!gpuInfo) return gpuInfo;

	//get the number of CUDA-enabled GPUs
	gpuInfo->n_device = 0;
	cudaGetDeviceCount(&gpuInfo->n_device);

	if (gpuInfo->n_device > 0) {
		gpuInfo->devices = (int*) malloc(sizeof(int) * gpuInfo->n_device);
		gpuInfo->props = (CUDADeviceProp*) malloc(sizeof(CUDADeviceProp) * gpuInfo->n_device);
		int realDevice = 0;
		for (i = 0; i < gpuInfo->n_device; i++) {
			gpuInfo->devices[realDevice] = i;
			cudaGetDeviceProperties(&gpuInfo->props[realDevice], i);

			/*check the compute capability*/
//			if (gpuInfo->props[realDevice].regsPerBlock < 16384
//					|| gpuInfo->props[realDevice].major < 3) {
//				continue;
//			}
			realDevice++;
		}
		gpuInfo->n_device = realDevice;
	}

	return gpuInfo;
}

void loadDBQuad(ESL_SQ_BLOCK *block, size_t offset, size_t count, size_t widthAligned, uint *arrayQuad)
{
	uint i, j;

	uint *ptr;
	for (i = offset; i < offset+count; ++i) {

		/*get the db sequence and its length*/
		ESL_SQ *dbsq = block->list + i;
		ESL_DSQ *dsq = dbsq->dsq;

		int sqLen = dbsq->n;

		/*compute the aligned length for the sq to copy into the array */
		int sqAlignedLen = (sqLen + DBSQ_LENGTH_ALIGNED - 1) / DBSQ_LENGTH_ALIGNED;
		sqAlignedLen *= DBSQ_LENGTH_ALIGNED;

		ptr = arrayQuad + (i - offset);
//		uint base = 0;
		union { uint v; uint8_t us[4]; } u;
		/*dsq[0] is not needed for msvfilter */
		for (j = 1; j < sqAlignedLen; j += DBSQ_LENGTH_ALIGNED) {
			u.us[3] = ((j+3) <= sqLen) ? dsq[j+3] : DUMMY_AMINO_ACID;
//			base <<= 8;
			u.us[2] = ((j+2) <= sqLen) ? dsq[j+2] : DUMMY_AMINO_ACID;
//			base <<= 8;
			u.us[1] = ((j+1) <= sqLen) ? dsq[j+1] : DUMMY_AMINO_ACID;
//			base <<= 8;
			u.us[0] = (j <= sqLen) ? dsq[j] : DUMMY_AMINO_ACID;

			/*move to next row*/
			*ptr = u.v;
			ptr += widthAligned;
		}
	}

}

/**
 * load om parameters into CUDA
 */
void loadOM(const P7_OPROFILE *om, CUDAMemPtr &cudaPtr, uint numSeqs)
{
	uint omArray[8];
	omArray[0] = om->M;
	omArray[1] = om->base_b;
	omArray[2] = om->bias_b;
	omArray[3] = om->tbm_b;
	omArray[4] = om->tec_b;

	checkCudaErrors(cudaMemcpyToSymbol(cudaOM, omArray, 8 * sizeof(uint)));
	checkCudaErrors(cudaMemcpyToSymbol(cudaScale_b, &om->scale_b, sizeof(float)));

	size_t Q = p7O_NQB(om->M);

	size_t num = (numSeqs + 1) / 2;
	size_t size = (Q << 4) * num * sizeof(uint8_t);

	/*every sequence has a DP for computing */
	checkCudaErrors(cudaMalloc(&cudaPtr.cudaDP0, size));
	size = (Q << 4) * (numSeqs - num) * sizeof(uint8_t);
	checkCudaErrors(cudaMalloc(&cudaPtr.cudaDP1, size));

	size = Q * om->abc->Kp * sizeof(uint4);
	checkCudaErrors(cudaMalloc(&cudaPtr.cudaOMrbv, size));

	texOMrbv.addressMode[0] = cudaAddressModeClamp;
	texOMrbv.addressMode[1] = cudaAddressModeClamp;
	texOMrbv.filterMode = cudaFilterModePoint;
	texOMrbv.normalized = false;
	cudaChannelFormatDesc uint4_channelDesc = cudaCreateChannelDesc<uint4>();
	checkCudaErrors(cudaBindTexture(0, texOMrbv, cudaPtr.cudaOMrbv, uint4_channelDesc, size));

	checkCudaErrors(cudaMemcpy(cudaPtr.cudaOMrbv, om->rbv[0], size, cudaMemcpyHostToDevice));

}

void allocCUDAFixed(HostMemPtr &hostPtr, CUDAMemPtr &cudaPtr, const P7_OPROFILE *om, uint numSeqs)
{
	/*the length of each sq*/
	checkCudaErrors(cudaHostAlloc((void **)&hostPtr.hashQuad, numSeqs * sizeof(uint), cudaHostAllocDefault));

	checkCudaErrors(cudaHostAlloc((void **)&hostPtr.hostSC, numSeqs * sizeof(float), cudaHostAllocDefault));

	size_t num = (numSeqs + 1) / 2;
	checkCudaErrors(cudaMalloc(&cudaPtr.cudaHashQuad0, num *sizeof(uint)));
	checkCudaErrors(cudaMalloc(&cudaPtr.cudaHashQuad1, (numSeqs - num) *sizeof(uint)));

	checkCudaErrors(cudaMalloc(&cudaPtr.cudaSC0, num * sizeof(float)));
	checkCudaErrors(cudaMalloc(&cudaPtr.cudaSC1, (numSeqs - num) * sizeof(float)));

	loadOM(om, cudaPtr, numSeqs);

	cudaFuncSetCacheConfig(msvSIMDKernel0, cudaFuncCachePreferL1); /**< Prefer larger L1 cache and smaller shared memory */
}

void cpuPipeline(ESL_SQ_BLOCK *block, int count, WORKER_INFO *info)
{
	for (int i = 0; i < count; ++i)
	{
		ESL_SQ *dbsq = block->list + i;

		p7_pli_NewSeq(info->pli, dbsq);
		p7_bg_SetLength(info->bg, dbsq->n);
		p7_oprofile_ReconfigLength(info->om, dbsq->n);

		p7_Pipeline(info->pli, info->om, info->bg, dbsq, info->th);

		esl_sq_Reuse(dbsq);
		p7_pipeline_Reuse(info->pli);
	}
}

void loadCUDA(HostMemPtr &hostPtr, CUDAMemPtr &cudaPtr, ESL_SQ_BLOCK *block, WORKER_INFO *info, cudaStream_t *stream)
{
	int numSeqs = block->count - CPU_SIZE;
	if (numSeqs < 2) {
		cpuPipeline(block, block->count, info);
		return;
	}

	int num1 = (numSeqs + 1) / 2;
	int num1a = KERNEL_BLOCKSIZE * ( (num1 + KERNEL_BLOCKSIZE - 1) / KERNEL_BLOCKSIZE);
	int num2 = numSeqs - num1;
	int num2a = KERNEL_BLOCKSIZE * ( (num2 + KERNEL_BLOCKSIZE - 1) / KERNEL_BLOCKSIZE);

	uint maxLen1 = 0;
	uint maxLen2 = 0;
	for (int i = 0; i < num1; ++i)
	{
		ESL_SQ *dbsq = block->list + (i + CPU_SIZE);
		hostPtr.hashQuad[i] = dbsq->n;
		if (maxLen1 < hostPtr.hashQuad[i]) maxLen1 = hostPtr.hashQuad[i];
	}
	for (int i = num1; i < numSeqs; ++i)
	{
		ESL_SQ *dbsq = block->list + (i + CPU_SIZE);
		hostPtr.hashQuad[i] = dbsq->n;
		if (maxLen2 < hostPtr.hashQuad[i]) maxLen2 = hostPtr.hashQuad[i];
	}

	uint heightQuad1 = (maxLen1 + DBSQ_LENGTH_ALIGNED - 1) / DBSQ_LENGTH_ALIGNED;
	uint heightQuad2 = (maxLen2 + DBSQ_LENGTH_ALIGNED - 1) / DBSQ_LENGTH_ALIGNED;

	int newH1 = heightQuad1 * 1.5;
	int newH2 = heightQuad2 * 1.5;
	if ((num1a*heightQuad1 + num2a*heightQuad2) > hostPtr.dbQuad.allocDQ) {
		if (hostPtr.dbQuad.array) cudaFreeHost(hostPtr.dbQuad.array);

		/*allocate a little more*/
		hostPtr.dbQuad.allocDQ = num1a * newH1 + num2a * newH2;
		checkCudaErrors(cudaHostAlloc((void **)&hostPtr.dbQuad.array, hostPtr.dbQuad.allocDQ * sizeof(uint), cudaHostAllocDefault));
	}

	if (num1a*heightQuad1 > cudaPtr.allocSQ1) {
		if (cudaPtr.cudaSeqsQuad0)	{
			cudaFree(cudaPtr.cudaSeqsQuad0);
			cudaPtr.cudaSeqsQuad0 = NULL;
		}
		cudaPtr.allocSQ1 = num1a * newH1;
		checkCudaErrors(cudaMalloc(&cudaPtr.cudaSeqsQuad0, num1a * newH1 * sizeof(uint)));

		// create texture object
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.readMode = cudaReadModeElementType;

		memset(&resDesc0, 0, sizeof(resDesc0));
		resDesc0.resType = cudaResourceTypeLinear;
		resDesc0.res.linear.desc.f = cudaChannelFormatKindUnsigned;
		resDesc0.res.linear.desc.x = 32; // bits per channel
		resDesc0.res.linear.devPtr = cudaPtr.cudaSeqsQuad0;
		resDesc0.res.linear.sizeInBytes = num1a * heightQuad1 * sizeof(uint);

		if (cudaPtr.tex0) cudaDestroyTextureObject(cudaPtr.tex0);
		checkCudaErrors(cudaCreateTextureObject(&cudaPtr.tex0, &resDesc0, &texDesc, NULL));

	}

	if (num2a*heightQuad2 > cudaPtr.allocSQ2) {
		if (cudaPtr.cudaSeqsQuad1)	{
			cudaFree(cudaPtr.cudaSeqsQuad1);
			cudaPtr.cudaSeqsQuad1 = NULL;
		}
		cudaPtr.allocSQ2 = num2a * newH2;
		checkCudaErrors(cudaMalloc(&cudaPtr.cudaSeqsQuad1, num2a * newH2 * sizeof(uint)));

		memset(&resDesc1, 0, sizeof(resDesc1));
		resDesc1.resType = cudaResourceTypeLinear;
		resDesc1.res.linear.desc.f = cudaChannelFormatKindUnsigned;
		resDesc1.res.linear.desc.x = 32; // bits per channel
		resDesc1.res.linear.devPtr = cudaPtr.cudaSeqsQuad1;
		resDesc1.res.linear.sizeInBytes = num2a * heightQuad2 * sizeof(uint);

		if (cudaPtr.tex1) cudaDestroyTextureObject(cudaPtr.tex1);
		checkCudaErrors(cudaCreateTextureObject(&cudaPtr.tex1, &resDesc1, &texDesc, NULL));

	}

	hostPtr.dbQuad.width = numSeqs;

	/*stream[0] */
	loadDBQuad(block, CPU_SIZE, num1, num1a, hostPtr.dbQuad.array);

	/*copy the database sequences from host to GPU device*/
	checkCudaErrors(cudaMemcpyAsync(cudaPtr.cudaSeqsQuad0,
									hostPtr.dbQuad.array,
									num1a * heightQuad1 * sizeof(uint),
									cudaMemcpyHostToDevice,
									stream[0]));

	/* *
	 * hashQuad can combined with dbQuad
	 */
	/*allocate and memcopy cudaHashQuad*/
	checkCudaErrors(cudaMemcpyAsync(cudaPtr.cudaHashQuad0, hostPtr.hashQuad, num1 * sizeof(uint), cudaMemcpyHostToDevice, stream[0]));

	int Q = p7O_NQB(info->om->M);
	/*Set cudaDP 0 */
	size_t size = (Q << 4) * num1a * sizeof(uint8_t);
	/*every sequence has a DP for computing */
	checkCudaErrors(cudaMemset(cudaPtr.cudaDP0, 0, size));
	size = (Q << 4) * num2a * sizeof(uint8_t);
	checkCudaErrors(cudaMemset(cudaPtr.cudaDP1, 0, size));

	int numBlocks = (num1a + KERNEL_BLOCKSIZE - 1) / KERNEL_BLOCKSIZE;

	msvSIMDKernel0<<<numBlocks, KERNEL_BLOCKSIZE, 0, stream[0]>>>(cudaPtr.tex0, cudaPtr.cudaHashQuad0, num1a, cudaPtr.cudaDP0, cudaPtr.cudaSC0, num1, Q);

	/*stream[1] */
	uint *ptr = hostPtr.dbQuad.array + num1a * heightQuad1;
	loadDBQuad(block, (num1+CPU_SIZE), num2, num2a, ptr);

	checkCudaErrors(cudaMemcpyAsync(cudaPtr.cudaSeqsQuad1,
									ptr,
									num2a * heightQuad2 * sizeof(uint),
									cudaMemcpyHostToDevice,
									stream[1]));

	checkCudaErrors(cudaMemcpyAsync(cudaPtr.cudaHashQuad1, hostPtr.hashQuad + num1, num2 * sizeof(uint), cudaMemcpyHostToDevice, stream[1]));

	numBlocks = (num2a + KERNEL_BLOCKSIZE - 1) / KERNEL_BLOCKSIZE;
	msvSIMDKernel0<<<numBlocks, KERNEL_BLOCKSIZE, 0, stream[1]>>>(cudaPtr.tex1, cudaPtr.cudaHashQuad1, num2a, cudaPtr.cudaDP1, cudaPtr.cudaSC1, num2, Q);

	cudaMemcpyAsync(hostPtr.hostSC, cudaPtr.cudaSC0, num1 * sizeof(float), cudaMemcpyDeviceToHost, stream[0]);
	cudaMemcpyAsync(hostPtr.hostSC + num1, cudaPtr.cudaSC1, num2 * sizeof(float), cudaMemcpyDeviceToHost, stream[1]);

	cpuPipeline(block, CPU_SIZE, info);

	cudaStreamSynchronize(stream[0]);
	cudaStreamSynchronize(stream[1]);

	for (int i = CPU_SIZE; i < block->count; ++i) {

		ESL_SQ *dbsq = block->list + i;

		p7_pli_NewSeq(info->pli, dbsq);
		p7_bg_SetLength(info->bg, dbsq->n);
		p7_oprofile_ReconfigLength(info->om, dbsq->n);

		p7_PipelineShort(info->pli, info->om, info->bg, dbsq, info->th, hostPtr.hostSC[i-CPU_SIZE]);

		// check
#if 0
		p7_Pipeline(info->pli, info->om, info->bg, dbsq, info->th);
		if (abs(hostPtr.hostSC[i] - dbsq->sc) > 0.00001) {
			printf("Error sc %d, sc %f hostSC %f\n", i, dbsq->sc, hostPtr.hostSC[i]);
		}
#endif
		esl_sq_Reuse(dbsq);
		p7_pipeline_Reuse(info->pli);

	}

}

void freeMem(HostMemPtr &hostPtr, CUDAMemPtr &cudaPtr)
{
	if (hostPtr.hashQuad) 	cudaFreeHost(hostPtr.hashQuad);
	if (hostPtr.hostSC) 	cudaFreeHost(hostPtr.hostSC);

	/*free dbQuad and hashQuad as soon as possible */
	if (hostPtr.dbQuad.array) {
		cudaFreeHost(hostPtr.dbQuad.array);
	}

	if (cudaPtr.cudaHashQuad0)	cudaFree(cudaPtr.cudaHashQuad0);
	if (cudaPtr.cudaHashQuad1)	cudaFree(cudaPtr.cudaHashQuad1);
	if (cudaPtr.cudaSC0)		cudaFree(cudaPtr.cudaSC0);
	if (cudaPtr.cudaSC1)		cudaFree(cudaPtr.cudaSC1);
	if (cudaPtr.cudaDP0)		cudaFree(cudaPtr.cudaDP0);
	if (cudaPtr.cudaDP1)		cudaFree(cudaPtr.cudaDP1);

	if (cudaPtr.cudaOMrbv)		cudaFree(cudaPtr.cudaOMrbv);
	if (cudaPtr.cudaSeqsQuad0)	cudaFree(cudaPtr.cudaSeqsQuad0);
	if (cudaPtr.cudaSeqsQuad1)	cudaFree(cudaPtr.cudaSeqsQuad1);

	if (cudaPtr.tex0) cudaDestroyTextureObject(cudaPtr.tex0);
	if (cudaPtr.tex1) cudaDestroyTextureObject(cudaPtr.tex1);

	cudaUnbindTexture(texOMrbv);

}

