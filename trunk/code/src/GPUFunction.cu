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
//#include "easel.h"
#include "esl_alphabet.h"
//#include "esl_getopts.h"
#include "esl_msa.h"
//#include "esl_msafile.h"
#include "esl_sq.h"
//#include "esl_sqio.h"
//#include "esl_stopwatch.h"

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
//#include <cuda_runtime_api.h>
//#include <cuda_texture_types.h>
//#include <texture_fetch_functions.h>

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


//texture<uint, 2, cudaReadModeElementType> texSeqsQuad0;
//texture<uint, 2, cudaReadModeElementType> texSeqsQuad1;
texture<uint4, cudaTextureType1D, cudaReadModeElementType> texOMrbv;


#if 1
__global__ void msvSIMDKernel0(cudaTextureObject_t tex, uint *cudaHashQuad, uint numAligned, uint8_t *cudaDP, float *cudaSC, uint Num, int Q) //, uint4 *cudaOMrbv)
{
	uint idx;
	idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= Num) return;

	uint L = cudaHashQuad[idx]; // Length of sq
//	if (L == 0) return;

	int i, y;			   /* counter over sequence positions 1..L                      */
	int q;			   /* counter over vectors 0..nq-1                              */

//	uint seqLenQ = (L + DBSQ_LENGTH_ALIGNED - 1) / DBSQ_LENGTH_ALIGNED;

	float sc = logf(3.0f / (float) (L+3));
	sc  = -1.0f * roundf(cudaScale_b * sc); // (om->scale_b * sc);       /* ugh. sc is now an integer cost represented in a float...    */

	uint tjb_b = (uint) sc;	/* and now we cast and saturate it to an unsigned char cost... */

//	uint8_t xBv16;		/* B state: splatted vector of B[i-1] for B->Mk calculations */
	uint xJv16 = 0;		/* special states' scores    vector for states score        */

	uint4 mpv32;	/* previous row values                                       */
	uint4 xEv32;	/* E state: keeps max for Mk->E as we go                     */
	uint4 sv32;		/* temp storage of 1 curr row value in progress              */
	uint xBv32;

//	int Q = p7O_NQB(M);   /* segment length: # of vectors                              */

//	uint4 *rsc64ptr;		/* will point at om->rbv[x] for residue x[i]                 */

	uint basev16 = cudaOM[1];		/* offset for scores                                         */
//	uint8_t biasv16 = cudaOM[2];		/* emission bias in a vector                                 */
//	uint8_t tjbmv16 = cudaOM[3] + tjb_b;		/* vector for cost of moving from either J or N through B to an M state */
	tjb_b += cudaOM[3];
	uint tecv16  = cudaOM[4];		/* vector for E->C  cost                                     */

	uint biasv32 = (cudaOM[2] << 24) | (cudaOM[2] << 16) | (cudaOM[2] << 8) | (cudaOM[2]);

//	xBv16 = SATURATE_SUB_UCHAR(basev16, tjbmv16);
	xBv32 = basev16 - tjb_b;//SATURATE_SUB_UCHAR(basev16, tjb_b);

	xBv32 |= (xBv32 << 16); xBv32 |= (xBv32 << 8);
//	memset(&xBv32, xBv32, sizeof(uint));

	uint4 *dp64 = (uint4 *) (cudaDP + (idx << 4));

	union { uint v; uint8_t us[4]; } u;
//	bool bMax = false;
	for (y = 0, i = 0; i < L; ++y, i += DBSQ_LENGTH_ALIGNED) {
//		u.v = tex2D(texSeqsQuad0, idx, y);
		tex1Dfetch(&u.v, tex, y * numAligned + idx);

		for (int k = 0; k < 4 && i + k < L; ++k) {
//			rsc64ptr = cudaOMrbv + u.us[k] * Q; //(uint4 *) (cudaOMrbv + u.us[k] * (Q << 4));

//			memset(&xEv32, 0, sizeof(uint4));
			xEv32.x = xEv32.y = xEv32.z = xEv32.w = 0;

//			uint4 *ptr = (uint4 *) (dp16 + ((Q-1) << 4));

			ulong2* ptr64 = (ulong2 *) (dp64 + (Q-1) * numAligned);
			ulong2 u2 = *ptr64;
	        ulong2 *tmp64 = (ulong2 *) &mpv32;
	        tmp64->x = u2.x;
	        tmp64->y = (u2.y << 8) | (tmp64->x >> 56 & 0xFF);
	        tmp64->x = tmp64->x << 8;

			for (q = 0; q < Q; q++)
			{
				/* Calculate new MMXo(i,q); don't store it yet, hold it in sv. */
//				uint4 *ptr = (uint4 *) (dp16 + (q << 4));

				PTX_SIMD_MAX_QUAD(sv32, mpv32, xBv32)

				PTX_SIMD_QUAD(vadd4, sv32, sv32, biasv32)

				uint4 rsc64 = tex1Dfetch(texOMrbv, u.us[k] * Q + q);

				PTX_SIMD_QUADALL(vsub4, sv32, sv32, rsc64);

				PTX_SIMD_MAX_QUADALL(xEv32, xEv32, sv32);
				mpv32 = *(dp64 + q * numAligned);
				*(dp64 + q * numAligned) = sv32;
//				memcpy(&mpv32, ptr, 16);
//				memcpy(ptr,  &sv32, 16);

//				rsc64ptr++;
			}
			/* test for the overflow condition */
			/* immediately detect overflow */
//			uint8_t tmp = 0;

//			TEST_OVERFLOW(tmp32, xEv32.x, biasv32)
//			TEST_OVERFLOW(tmp32, xEv32.y, biasv32)
//			TEST_OVERFLOW(tmp32, xEv32.z, biasv32)
//			TEST_OVERFLOW(tmp32, xEv32.w, biasv32)

	        asm("vmax4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(xEv32.x) : "r"(xEv32.x), "r"(xEv32.y), "r"(0));
	        asm("vmax4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(xEv32.z) : "r"(xEv32.z), "r"(xEv32.w), "r"(0));
	        asm("vmax4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(xEv32.x) : "r"(xEv32.x), "r"(xEv32.z), "r"(0));
	        asm("vmax4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(xEv32.x) : "r"(xEv32.x), "r"(xEv32.x >> 16), "r"(0));
	        asm("vmax4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(xEv32.x) : "r"(xEv32.x), "r"(xEv32.x >> 8), "r"(0));

			uint tmp8 = (xEv32.x & 0xFF) - tecv16; //SATURATE_SUB_UCHAR((xEv32.x & 0xFF), tecv16);

//	        memset(&xEv32, tmp8, 16);

			xJv16 = max(xJv16, tmp8);
			xBv32 = max(basev16, xJv16) - tjb_b;
//			xBv16 = SATURATE_SUB_UCHAR(xBv16, tjbmv16);
//			xBv16 = xBv16 - tjb_b; //SATURATE_SUB_UCHAR(xBv16, tjb_b);

//			xBv32 = xBv16;
//			memset(&xBv32, xBv32, sizeof(uint));
			xBv32 |= (xBv32 << 16);
			xBv32 |= (xBv32 << 8);
//			memset(&xBv32, xBv16, sizeof(uint));
		}
	}

	sc = ((float) (xJv16 - tjb_b + cudaOM[3]) - (float) basev16);
	sc /= cudaScale_b;
	sc -= 3.0; /* that's ~ L \log \frac{L}{L+3}, for our NN,CC,JJ */

	cudaSC[idx] = sc;
//	if (bMax) cudaSC[idx] = eslINFINITY;
//	else cudaSC[idx] = sc;
}
#endif

GPUInfo *getGPUInfo()
{
	int i;
	GPUInfo *gpuInfo = NULL;
	gpuInfo = (GPUInfo*) malloc(sizeof(GPUInfo));

	if (!gpuInfo) return gpuInfo;

	//get the number of CUDA-enabled GPUs
	gpuInfo->n_device = 0;
	cudaGetDeviceCount(&gpuInfo->n_device);
//	CUDA_ERR

	if (gpuInfo->n_device > 0) {
		gpuInfo->devices = (int*) malloc(sizeof(int) * gpuInfo->n_device);
		gpuInfo->props = (CUDADeviceProp*) malloc(sizeof(CUDADeviceProp) * gpuInfo->n_device);
		int realDevice = 0;
		for (i = 0; i < gpuInfo->n_device; i++) {
			gpuInfo->devices[realDevice] = i;
			cudaGetDeviceProperties(&gpuInfo->props[realDevice], i);
//			CUDA_ERR

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
//	memset(hostPtr.hashQuad, 8, numSeqs * sizeof(uint));

	checkCudaErrors(cudaHostAlloc((void **)&hostPtr.hostSC, numSeqs * sizeof(float), cudaHostAllocDefault));

	size_t num = (numSeqs + 1) / 2;
	checkCudaErrors(cudaMalloc(&cudaPtr.cudaHashQuad0, num *sizeof(uint)));
	checkCudaErrors(cudaMalloc(&cudaPtr.cudaHashQuad1, (numSeqs - num) *sizeof(uint)));

//	cudaMemset(cudaPtr.cudaHashQuad0, 8, num *sizeof(int));
//	cudaMemset(cudaPtr.cudaHashQuad1, 8, (numSeqs - num) *sizeof(int));

	checkCudaErrors(cudaMalloc(&cudaPtr.cudaSC0, num * sizeof(float)));
	checkCudaErrors(cudaMalloc(&cudaPtr.cudaSC1, (numSeqs - num) * sizeof(float)));

	loadOM(om, cudaPtr, numSeqs);

//	texSeqsQuad0.addressMode[0] = cudaAddressModeClamp;
//	texSeqsQuad0.addressMode[1] = cudaAddressModeClamp;
//	texSeqsQuad0.filterMode = cudaFilterModePoint;
//	texSeqsQuad0.normalized = false;
//
//	texSeqsQuad1.addressMode[0] = cudaAddressModeClamp;
//	texSeqsQuad1.addressMode[1] = cudaAddressModeClamp;
//	texSeqsQuad1.filterMode = cudaFilterModePoint;
//	texSeqsQuad1.normalized = false;

	cudaFuncSetCacheConfig(msvSIMDKernel0, cudaFuncCachePreferL1); /**< Prefer larger L1 cache and smaller shared memory */
//	cudaFuncSetCacheConfig(msvSIMDKernel1, cudaFuncCachePreferL1); /**< Prefer larger L1 cache and smaller shared memory */
//	cudaFuncSetCacheConfig(msvKernel, cudaFuncCachePreferL1); /**< Prefer larger L1 cache and smaller shared memory */

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
//	GPUInfo * gpuInfo = NULL;// getGPUInfo();
//
//#if eslDEBUG
//	FILE *ofp = NULL;
//	char str[64];
//	snprintf(str, sizeof(str), "%u", pthread_self());
//	if ((ofp = fopen(strcat(str, "_data.txt"), "w")) == NULL) p7_Fail("Failed to open output file %s for writing\n", strcat(str, "_data.txt"));
//#endif
//
//	gpuInfo = getGPUInfo();
//	if (gpuInfo) {
//		cudaSetDevice(gpuInfo->devices[0]);
//		CUDA_ERR
//#if eslDEBUG
//		fprintf(ofp, "GPU %s : globalMem - %d MB\n", gpuInfo->props[0].name, gpuInfo->props->totalGlobalMem);
//#endif
//		if (gpuInfo->devices) free(gpuInfo->devices);
//		if (gpuInfo->props) free(gpuInfo->props);
//		free(gpuInfo);
//	}

	//	  SQ_Entry *sortedSQs = getSortedSqEntry(block);
	/*compute the height of the CUDA array*/
	//	  int heightHex = sortedSQs[block->count - 1].alignedLen;


	int numSeqs = block->count - CPU_SIZE;
	if (numSeqs < 2) {
		cpuPipeline(block, block->count, info);
		return;
	}

	int num1 = (numSeqs + 1) / 2;
	int num1a = KERNEL_BLOCKSIZE * ( (num1 + KERNEL_BLOCKSIZE - 1) / KERNEL_BLOCKSIZE);
	int num2 = numSeqs - num1;
	int num2a = KERNEL_BLOCKSIZE * ( (num2 + KERNEL_BLOCKSIZE - 1) / KERNEL_BLOCKSIZE);

//	if (numSeqs != 8192) printf("not 8192 : num %d, num1: %d, num1a: %d, num2: %d, num2a: %d\n", numSeqs, num1, num1a, num2, num2a);

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
//		if (hostPtr.dbQuad.array) free(hostPtr.dbQuad.array);
		if (hostPtr.dbQuad.array) cudaFreeHost(hostPtr.dbQuad.array);

		//	checkCudaErrors(cudaMallocHost((void **)&hostPtr.dbQuad.array, numSeqs * newHeightQuad * sizeof(uint)));
		/*allocate a little more*/
		hostPtr.dbQuad.allocDQ = num1a * newH1 + num2a * newH2;
//		hostPtr.dbQuad.array = (uint *)malloc(hostPtr.dbQuad.allocDQ * sizeof(uint));
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

//	struct cudaChannelFormatDesc uint_channelDesc = cudaCreateChannelDesc<uint>();

	hostPtr.dbQuad.width = numSeqs;
//	hostPtr.dbQuad.heightQuad = newHeightQuad;

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

//	msvKernel<<<grid, block>>>(cudaPtr.cudaHashQuad, numSeqs, cudaPtr.cudaDP, cudaPtr.cudaOMrbv, cudaPtr.cudaSC);
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

//	launchCUDAKernel(cudaPtr, numSeqs, om, hostPtr, block);

	cpuPipeline(block, CPU_SIZE, info);

	cudaStreamSynchronize(stream[0]);
	cudaStreamSynchronize(stream[1]);

	for (int i = CPU_SIZE; i < block->count; ++i) {

		ESL_SQ *dbsq = block->list + i;

		p7_pli_NewSeq(info->pli, dbsq);
		p7_bg_SetLength(info->bg, dbsq->n);
		p7_oprofile_ReconfigLength(info->om, dbsq->n);


		//  from p7_Pipeline
		//			  			  if (sq->n == 0) return eslOK;    /* silently skip length 0 seqs; they'd cause us all sorts of weird problems */
		//
		//			  			  p7_omx_GrowTo(pli->oxf, om->M, 0, sq->n);    /* expand the one-row omx if needed */
		//
		//			  			  /* Base null model score (we could calculate this in NewSeq(), for a scan pipeline) */
		//			  			  p7_bg_NullOne  (bg, sq->dsq, sq->n, &nullsc);

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

//	cudaUnbindTexture(texSeqsQuad0);
//	cudaUnbindTexture(texSeqsQuad1);

}

void freeMem(HostMemPtr &hostPtr, CUDAMemPtr &cudaPtr)
{
	if (hostPtr.hashQuad) 	cudaFreeHost(hostPtr.hashQuad);
	if (hostPtr.hostSC) 	cudaFreeHost(hostPtr.hostSC);
//	if (hostPtr.hostResult)	free(hostPtr.hostResult);

	/*free dbQuad and hashQuad as soon as possible */
	if (hostPtr.dbQuad.array) {
		cudaFreeHost(hostPtr.dbQuad.array);
//		free(hostPtr.dbQuad.array);
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

//	if (cudaPtr.cudaResult)		cudaFree(cudaPtr.cudaResult);
}

#if 0
__global__ void msvSIMDKernel1(uint *cudaHashQuad, uint numAligned, uint8_t *cudaDP, uint4 *cudaOMrbv, float *cudaSC, uint Num)
{
	uint idx;
	idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= Num) return;

	uint L = cudaHashQuad[idx]; // Length of sq
//	if (L == 0) return;

	uint seqLenQ = (L + DBSQ_LENGTH_ALIGNED - 1) / DBSQ_LENGTH_ALIGNED;

	float sc = logf(3.0f / (float) (L+3));
	sc  = -1.0f * roundf(cudaScale_b * sc); // (om->scale_b * sc);       /* ugh. sc is now an integer cost represented in a float...    */

	uint8_t tjb_b = (sc > 255.) ? 255 : (uint8_t) sc;	/* and now we cast and saturate it to an unsigned char cost... */

	uint8_t xBv16;		/* B state: splatted vector of B[i-1] for B->Mk calculations */
	uint8_t xJv16 = 0;		/* special states' scores    vector for states score        */

	uint4 mpv32;	/* previous row values                                       */
	uint4 xEv32;	/* E state: keeps max for Mk->E as we go                     */
	uint4 sv32;		/* temp storage of 1 curr row value in progress              */
	uint xBv32;

	int i, y;			   /* counter over sequence positions 1..L                      */
	int q;			   /* counter over vectors 0..nq-1                              */
	int Q = p7O_NQB(cudaOM[0]);   /* segment length: # of vectors                              */

	uint4 *rsc64ptr;		/* will point at om->rbv[x] for residue x[i]                 */

	uint8_t basev16 = cudaOM[1];		/* offset for scores                                         */
//	uint8_t biasv16 = cudaOM[2];		/* emission bias in a vector                                 */
//	uint8_t tjbmv16 = cudaOM[3] + tjb_b;		/* vector for cost of moving from either J or N through B to an M state */
	tjb_b += cudaOM[3];
	uint8_t tecv16  = cudaOM[4];		/* vector for E->C  cost                                     */

	uint biasv32 = (cudaOM[2] << 24) | (cudaOM[2] << 16) | (cudaOM[2] << 8) | (cudaOM[2]);

//	xBv16 = SATURATE_SUB_UCHAR(basev16, tjbmv16);
	xBv16 = SATURATE_SUB_UCHAR(basev16, tjb_b);

	memset(&xBv32, xBv16, sizeof(uint));

	uint4 *dp64 = (uint4 *) (cudaDP + (idx << 4));

	union { uint v; uint8_t us[4]; } u;
//	bool bMax = false;
	for (y = 0, i = 0; y < seqLenQ && i < L; ++y, i += DBSQ_LENGTH_ALIGNED) {
		u.v = tex2D(texSeqsQuad1, idx, y);

		for (int k = 0; k < 4 && i + k < L; ++k) {
			rsc64ptr = cudaOMrbv + u.us[k] * Q; //(uint4 *) (cudaOMrbv + u.us[k] * (Q << 4));

//			memset(&xEv32, 0, sizeof(uint4));
			xEv32.x = xEv32.y = xEv32.z = xEv32.w = 0;

//			uint4 *ptr = (uint4 *) (dp16 + ((Q-1) << 4));

			ulong2* ptr64 = (ulong2 *) (dp64 + (Q-1) * numAligned);
			ulong2 u2 = *ptr64;
	        ulong2 *tmp64 = (ulong2 *) &mpv32;
	        tmp64->x = u2.x;
	        tmp64->y = (u2.y << 8) | (tmp64->x >> 56 & 0xFF);
	        tmp64->x = tmp64->x << 8;

			uint8_t tmp8;
			for (q = 0; q < Q; q++)
			{
				/* Calculate new MMXo(i,q); don't store it yet, hold it in sv. */
//				uint4 *ptr = (uint4 *) (dp16 + (q << 4));

				PTX_SIMD_MAX_QUAD(sv32, mpv32, xBv32)

				PTX_SIMD_QUAD(vadd4, sv32, sv32, biasv32)

				uint4 rsc64 = *rsc64ptr;
				PTX_SIMD_QUADALL(vsub4, sv32, sv32, rsc64);
//				asm("vsub4.u32.u32.u32.sat %0, %1, %2, %3;" : "=r"(sv32.x) : "r"(sv32.x), "r"(ptr32[0]), "r"(0));
//				asm("vsub4.u32.u32.u32.sat %0, %1, %2, %3;" : "=r"(sv32.y) : "r"(sv32.y), "r"(ptr32[1]), "r"(0));
//				asm("vsub4.u32.u32.u32.sat %0, %1, %2, %3;" : "=r"(sv32.z) : "r"(sv32.z), "r"(ptr32[2]), "r"(0));
//				asm("vsub4.u32.u32.u32.sat %0, %1, %2, %3;" : "=r"(sv32.w) : "r"(sv32.w), "r"(ptr32[3]), "r"(0));

				PTX_SIMD_MAX_QUADALL(xEv32, xEv32, sv32);
				mpv32 = *(dp64 + q * numAligned);
				*(dp64 + q * numAligned) = sv32;
//				memcpy(&mpv32, ptr, 16);
//				memcpy(ptr,  &sv32, 16);

				rsc64ptr++;
			}
			/* test for the overflow condition */
			/* immediately detect overflow */
//			uint8_t tmp = 0;

//			TEST_OVERFLOW(tmp32, xEv32.x, biasv32)
//			TEST_OVERFLOW(tmp32, xEv32.y, biasv32)
//			TEST_OVERFLOW(tmp32, xEv32.z, biasv32)
//			TEST_OVERFLOW(tmp32, xEv32.w, biasv32)

	        asm("vmax4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(xEv32.x) : "r"(xEv32.x), "r"(xEv32.y), "r"(0));
	        asm("vmax4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(xEv32.z) : "r"(xEv32.z), "r"(xEv32.w), "r"(0));
	        asm("vmax4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(xEv32.x) : "r"(xEv32.x), "r"(xEv32.z), "r"(0));
	        asm("vmax4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(xEv32.x) : "r"(xEv32.x), "r"(xEv32.x >> 16), "r"(0));
	        asm("vmax4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(xEv32.x) : "r"(xEv32.x), "r"(xEv32.x >> 8), "r"(0));

	        tmp8 = SATURATE_SUB_UCHAR((xEv32.x & 0xFF), tecv16);

	        memset(&xEv32, tmp8, 16);

			xJv16 = max(xJv16, tmp8);
			xBv16 = max(basev16, xJv16);
//			xBv16 = SATURATE_SUB_UCHAR(xBv16, tjbmv16);
			xBv16 = SATURATE_SUB_UCHAR(xBv16, tjb_b);

			memset(&xBv32, xBv16, sizeof(uint));
		}
	}

	sc = ((float) (xJv16 - tjb_b + cudaOM[3]) - (float) basev16);
	sc /= cudaScale_b;
	sc -= 3.0; /* that's ~ L \log \frac{L}{L+3}, for our NN,CC,JJ */

	cudaSC[idx] = sc;
//	if (bMax) cudaSC[idx] = eslINFINITY;
//	else cudaSC[idx] = sc;
}

__global__ void msvKernel(uint *cudaHashQuad, uint numSeqs, uint8_t *cudaDP, uint8_t *cudaOMrbv, float *cudaSC)
{
	uint idx;
	idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= numSeqs) return;

	uint L = cudaHashQuad[idx]; // Length of sq
//	if (L == 0) return;

	uint seqLenQ = (L + DBSQ_LENGTH_ALIGNED - 1) / DBSQ_LENGTH_ALIGNED;

	float sc = logf(3.0f / (float) (L+3));
	sc  = -1.0f * roundf(cudaScale_b * sc); // (om->scale_b * sc);       /* ugh. sc is now an integer cost represented in a float...    */

	uint8_t tjb_b = (sc > 255.) ? 255 : (uint8_t) sc;	/* and now we cast and saturate it to an unsigned char cost... */

//	uint pack = 0;

	  uint8_t mpv16[16];	/* previous row values                                       */
	  uint8_t xEv16[16];	/* E state: keeps max for Mk->E as we go                     */
	  uint8_t sv16[16];		/* temp storage of 1 curr row value in progress              */
	  uint8_t xBv16;		/* B state: splatted vector of B[i-1] for B->Mk calculations */

	  uint8_t  xJv16 = 0;		/* special states' scores    vector for states score        */
	  int i, y;			   /* counter over sequence positions 1..L                      */
	  int q;			   /* counter over vectors 0..nq-1                              */
	  int Q = p7O_NQB(cudaOM[0]);   /* segment length: # of vectors                              */

	  uint8_t *rsc16;		/* will point at om->rbv[x] for residue x[i]                 */

	  uint8_t basev16 = cudaOM[1];		/* offset for scores                                         */
	  uint8_t biasv16 = cudaOM[2];		/* emission bias in a vector                                 */
	  uint8_t tjbmv16 = cudaOM[3] + tjb_b;		/* vector for cost of moving from either J or N through B to an M state */
	  uint8_t tecv16  = cudaOM[4];		/* vector for E->C  cost                                     */

	  xBv16 = SATURATE_SUB_UCHAR(basev16, tjbmv16);

	  uint8_t *dp16 = cudaDP + idx * (Q << 4) * sizeof(uint8_t);

	union { uint v; uint8_t us[4]; } u;
	bool bMax = false;
	for (y = 0, i = 0; y < seqLenQ && i < L; ++y, i += DBSQ_LENGTH_ALIGNED) {
		u.v = tex2D(texSeqsQuad, idx, y);
//		cudaResult[y*DBSQ_LENGTH_ALIGNED * numSeqs + idx] = u.us[0];
//		cudaResult[(y*DBSQ_LENGTH_ALIGNED+1) * numSeqs + idx] = u.us[1];
//		cudaResult[(y*DBSQ_LENGTH_ALIGNED+2) * numSeqs + idx] = u.us[2];
//		cudaResult[(y*DBSQ_LENGTH_ALIGNED+3) * numSeqs + idx] = u.us[3];

		for (int k = 0; k < 4 && i + k < L; ++k) {
////			cudaResult[(y*DBSQ_LENGTH_ALIGNED+k) * numSeqs + idx] = u.us[k];
			rsc16 = cudaOMrbv + u.us[k] * (Q << 4);
			memset(xEv16, 0, 16);

		      uint8_t *ptr = dp16 + ((Q-1) << 4);
		      for (int j = 0; j < 15; ++j) {
		    	  mpv16[j + 1] = *(ptr + j);
		      }
		      mpv16[0] = 0;

		      for (q = 0; q < Q; q++)
		      {
		        /* Calculate new MMXo(i,q); don't store it yet, hold it in sv. */
		    	ptr = dp16 + (q << 4);
		        for (int j = 0; j < 16; ++j) {
		        	sv16[j] = max(mpv16[j], xBv16);
		        	sv16[j] = SATURATE_ADD_UCHAR(sv16[j], biasv16);
		        	sv16[j] = SATURATE_SUB_UCHAR(sv16[j], *(rsc16 + j));
		        	xEv16[j] = max(xEv16[j], sv16[j]);
		        	mpv16[j] = *(ptr + j);	/* Load {MDI}(i-1,q) into mpv */
		        	*(ptr + j) = sv16[j];	/* Do delayed store of M(i,q) now that memory is usable */
		        }
		        rsc16 += 16;
		      }
		      /* test for the overflow condition */
		      /* immediately detect overflow */
		      uint8_t tmp = 0;
		      for (int j = 0; j < 16; ++j) {
		    	  tmp = SATURATE_ADD_UCHAR(xEv16[j], biasv16);
		    	  if (tmp == UCHAR_MAX) {
		    		  bMax = true;
//		    		  cudaSC[idx] = eslINFINITY;
//		    		  return;
		    	  }
		      }
		      tmp = 0;
		      for (int j = 0; j < 16; ++j) {
		    	  if (xEv16[j] > tmp) tmp = xEv16[j];
		      }
		      tmp = SATURATE_SUB_UCHAR(tmp, tecv16);
		      for (int j = 0; j < 16; ++j) {
		    	  xEv16[j] = tmp;
		      }
			  xJv16 = max(xJv16, tmp);
			  xBv16 = max(basev16, xJv16);
			  xBv16 = SATURATE_SUB_UCHAR(xBv16, tjbmv16);

		}
	}

	  sc = ((float) (xJv16 - tjb_b) - (float) basev16);
	  sc /= cudaScale_b;
	  sc -= 3.0; /* that's ~ L \log \frac{L}{L+3}, for our NN,CC,JJ */

//	switch (seqLen & (DBSQ_LENGTH_ALIGNED -1)) {
//	case 1:
//		pack = tex2D(cudaSeqsQuad, idx, seqLenQ);
//		cudaResult[seqLenQ*DBSQ_LENGTH_ALIGNED * numSeqs + idx] = pack & 0x0FF;
//		break;
//	case 2:
//		pack = tex2D(cudaSeqsQuad, idx, seqLenQ);
//		cudaResult[(seqLenQ*DBSQ_LENGTH_ALIGNED) * numSeqs + idx] = pack & 0x0FF;
//		pack >>= 8;
//		cudaResult[(seqLenQ*DBSQ_LENGTH_ALIGNED+1) * numSeqs + idx] = pack & 0x0FF;
//		break;
//	case 3:
//		pack = tex2D(cudaSeqsQuad, idx, seqLenQ);
//		cudaResult[(seqLenQ*DBSQ_LENGTH_ALIGNED) * numSeqs + idx] = pack & 0x0FF;
//		pack >>= 8;
//		cudaResult[(seqLenQ*DBSQ_LENGTH_ALIGNED+1) * numSeqs + idx] = pack & 0x0FF;
//		pack >>= 8;
//		cudaResult[(seqLenQ*DBSQ_LENGTH_ALIGNED+2) * numSeqs + idx] = pack & 0x0FF;
//		break;
//	}

	/* bg->p1 */
//	float p1 = (float) L / (float) (L+1);

	/* p7_bg_NullOne  (bg, sq->dsq, sq->n, &nullsc); */
//	float nullsc = (float) L * log(p1) + log(1. - p1);

//	sc = (sc - nullsc) / eslCONST_LOG2;

	  if (bMax) cudaSC[idx] = eslINFINITY;
	  else cudaSC[idx] = sc;

}

__global__ void testKernel(uint *cudaHashQuad, uint numSeqs, uint8_t *cudaDP, uint8_t *cudaDPtmp, uint8_t *cudaOMrbv, float *cudaSC)
{
	uint idx;
	idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= numSeqs) return;

#if 0
	int h = 0;
	int hd = 0xf2f3f4fd;
	int sub = 0x06060606;

	if (idx == 0) {
		for (int i = 0; i < 2; ++i) {
		uint8_t *mpv16 = cudaDP + i * 16;

		uint8_t *ptr = cudaOMrbv + i * 16;
		for (int j = 0; j < 15; ++j) {
			mpv16[j + 1] = *(ptr + j);
		}
		mpv16[0] = 0;

		uint64_t* rbv = (uint64_t *) (cudaOMrbv + i * 16);
		cudaInt[i*2] = rbv[0] << 8;
		cudaInt[i*2 + 1] = rbv[1] << 8 | (rbv[0] >> 56 & 0xFF);

        ulong2* rbv2 = (ulong2 *) (cudaOMrbv + i * 16);
        cudaInt[i*2] = (*rbv2).x << 8;
        cudaInt[i*2 + 1] = (*rbv2).y << 8 | ((*rbv2).x >> 56 & 0xFF);
        rbv2->x = 3;

        ulong2 tmp;
        tmp.x = rbv2->x;
        tmp.y = rbv2->y << 8 | (tmp.x >> 56 & 0xFF);
        tmp.x = tmp.x << 8;

        uint4 mpv64;
        memcpy(&mpv64, &tmp, sizeof(ulong2));

        printf("cuda in2@\t %016lX %016lX\n cuda mpv\t %08X%08X %08X%08X\n", tmp.x, tmp.y, mpv64.x, mpv64.y, mpv64.z, mpv64.w);

        printf("cuda 8bit\t");
        uint8_t *t8 = (uint8_t *) &tmp;
        for (int i = 0; i < 16; ++i)
                printf("%02X", t8[i]);
        printf("\n");

		}

	asm("vadd4.s32.s32.s32.sat %0, %1, %2, %3;" : "=r"(h) : "r"(hd), "r"(sub), "r"(0));

	}
	if (idx == 1) {
		asm("vsub4.s32.s32.s32.sat %0, %1, %2, %3;" : "=r"(h) : "r"(hd), "r"(sub), "r"(0));

	}
	if (idx == 2) {
		asm("vsub4.s32.s32.s32.sat %0, %1, %2, %3;" : "=r"(h) : "r"(sub), "r"(hd), "r"(0));

	}
#endif

	uint L = cudaHashQuad[idx]; // Length of sq
	if (L == 0) return;

	uint seqLenQ = (L + DBSQ_LENGTH_ALIGNED - 1) / DBSQ_LENGTH_ALIGNED;

	float sc = logf(3.0f / (float) (L+3));
	sc  = -1.0f * roundf(cudaScale_b * sc); // (om->scale_b * sc);       /* ugh. sc is now an integer cost represented in a float...    */

	uint8_t tjb_b = (sc > 255.) ? 255 : (uint8_t) sc;	/* and now we cast and saturate it to an unsigned char cost... */

	uint8_t mpv16[16];	/* previous row values                                       */
	uint8_t xEv16[16];	/* E state: keeps max for Mk->E as we go                     */
	uint8_t sv16[16];		/* temp storage of 1 curr row value in progress              */
	uint8_t xBv16;		/* B state: splatted vector of B[i-1] for B->Mk calculations */
	uint8_t  xJv16 = 0;		/* special states' scores    vector for states score        */

	uint4 mpv32;
	uint4 xEv32;
	uint4 sv32;
	uint xBv32;

	int i, y;			   /* counter over sequence positions 1..L                      */
	int q;			   /* counter over vectors 0..nq-1                              */
	int Q = p7O_NQB(cudaOM[0]);   /* segment length: # of vectors                              */

	uint8_t *rsc16;		/* will point at om->rbv[x] for residue x[i]                 */

	uint8_t basev16 = cudaOM[1];		/* offset for scores                                         */
	uint8_t biasv16 = cudaOM[2];		/* emission bias in a vector                                 */
	uint8_t tjbmv16 = cudaOM[3] + tjb_b;		/* vector for cost of moving from either J or N through B to an M state */
	uint8_t tecv16  = cudaOM[4];		/* vector for E->C  cost                                     */

//	uint basev32 = (basev16 << 24) | (basev16 << 16) | (basev16 << 8) | (basev16);
	uint biasv32 = (biasv16 << 24) | (biasv16 << 16) | (biasv16 << 8) | (biasv16);
//	uint tjbmv32 = (tjbmv16 << 24) | (tjbmv16 << 16) | (tjbmv16 << 8) | (tjbmv16);

	xBv16 = SATURATE_SUB_UCHAR(basev16, tjbmv16);

	memset(&xBv32, xBv16, sizeof(uint));
//	asm("vsub4.u32.u32.u32.sat %0, %1, %2, %3;" : "=r"(xBv32) : "r"(basev32), "r"(tjbmv32), "r"(0));

	uint8_t *dp16 = cudaDP + idx * (Q << 4) * sizeof(uint8_t);
	uint8_t *dp16tmp = cudaDPtmp + idx * (Q << 4) * sizeof(uint8_t);

	union { uint v; uint8_t us[4]; } u;
	for (y = 0, i = 0; y < seqLenQ && i < L; ++y, i += DBSQ_LENGTH_ALIGNED) {
		u.v = tex2D(texSeqsQuad, idx, y);

		for (int k = 0; k < 4 && i + k < L; ++k) {
			//			cudaResult[(y*DBSQ_LENGTH_ALIGNED+k) * numSeqs + idx] = u.us[k];
			rsc16 = cudaOMrbv + u.us[k] * (Q << 4);
			memset(xEv16, 0, 16);

			memset(&xEv32, 0, sizeof(uint4));

			uint8_t *ptr = dp16 + ((Q-1) << 4);
			for (int j = 0; j < 15; ++j) {
				mpv16[j + 1] = *(ptr + j);
			}
			mpv16[0] = 0;

			ulong2* ptr64 = (ulong2 *) (dp16tmp + ((Q-1) << 4));
	        ulong2 *tmp64 = (ulong2 *) &mpv32;
	        tmp64->x = ptr64->x;
	        tmp64->y = (ptr64->y << 8) | (tmp64->x >> 56 & 0xFF);
	        tmp64->x = tmp64->x << 8;

	        // check
	        uint8_t *ptrtmp = (uint8_t *) &mpv32;
	        for (int j = 0; j < 16; ++j) {
	        	if (mpv16[j] != ptrtmp[j]) {
	        		printf("Error mpv32!");
	        		return;
	        	}
	        }

			for (q = 0; q < Q; q++)
			{
				/* Calculate new MMXo(i,q); don't store it yet, hold it in sv. */
				ptr = dp16 + (q << 4);

				ptrtmp = dp16tmp + (q << 4);

				for (int j = 0; j < 16; ++j) {
					sv16[j] = max(mpv16[j], xBv16);
					sv16[j] = SATURATE_ADD_UCHAR(sv16[j], biasv16);
					sv16[j] = SATURATE_SUB_UCHAR(sv16[j], *(rsc16 + j));
					xEv16[j] = max(xEv16[j], sv16[j]);
					mpv16[j] = *(ptr + j);	/* Load {MDI}(i-1,q) into mpv */
					*(ptr + j) = sv16[j];	/* Do delayed store of M(i,q) now that memory is usable */
				}

//				asm("vmax4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(sv32.x) : "r"(mpv32.x), "r"(xBv32), "r"(0));
//				asm("vmax4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(sv32.y) : "r"(mpv32.y), "r"(xBv32), "r"(0));
//				asm("vmax4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(sv32.z) : "r"(mpv32.z), "r"(xBv32), "r"(0));
//				asm("vmax4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(sv32.w) : "r"(mpv32.w), "r"(xBv32), "r"(0));

				PTX_SIMD_MAX_QUAD(sv32, mpv32, xBv32)

//				asm("vadd4.u32.u32.u32.sat %0, %1, %2, %3;" : "=r"(sv32.x) : "r"(sv32.x), "r"(biasv32), "r"(0));
				PTX_SIMD_QUAD(vadd4, sv32, sv32, biasv32)

				uint *ptr32 = (uint *)rsc16;
				asm("vsub4.u32.u32.u32.sat %0, %1, %2, %3;" : "=r"(sv32.x) : "r"(sv32.x), "r"(ptr32[0]), "r"(0));
				asm("vsub4.u32.u32.u32.sat %0, %1, %2, %3;" : "=r"(sv32.y) : "r"(sv32.y), "r"(ptr32[1]), "r"(0));
				asm("vsub4.u32.u32.u32.sat %0, %1, %2, %3;" : "=r"(sv32.z) : "r"(sv32.z), "r"(ptr32[2]), "r"(0));
				asm("vsub4.u32.u32.u32.sat %0, %1, %2, %3;" : "=r"(sv32.w) : "r"(sv32.w), "r"(ptr32[3]), "r"(0));

				PTX_SIMD_MAX_QUADALL(xEv32, xEv32, sv32);
				memcpy(&mpv32, ptrtmp, 16);
				memcpy(ptrtmp,  &sv32, 16);

				rsc16 += 16;

		        // check
		        uint8_t *pt1 = (uint8_t *) &mpv32;
		        uint8_t *pt2 = (uint8_t *) &sv32;
		        for (int j = 0; j < 16; ++j) {
		        	if (mpv16[j] != pt1[j]) {
		        		printf("Error mpv !!!!");
		        		return;
		        	}
		        	if (sv16[j] != pt2[j]) {
		        		printf("Error sv!\n");
		        		printf("xBv16 %X xBv32 %X\n", xBv16, xBv32);
		        		for (int k = 0; k < 16; ++k)
		        			printf("sv16 %2X sv32 %2X\n", sv16[k], pt2[k]);
		        		return;
		        	}
		        }
			}
			/* test for the overflow condition */
			/* immediately detect overflow */
			uint8_t tmp = 0;
			bool bInf = false;
			for (int j = 0; j < 16; ++j) {
				tmp = SATURATE_ADD_UCHAR(xEv16[j], biasv16);
				if (tmp == UCHAR_MAX) {
					cudaSC[idx] = eslINFINITY;
					bInf = true;
//					return;
				}
			}

			uint tmp32;
//			asm("vadd4.u32.u32.u32.sat %0, %1, %2, %3;" : "=r"(tmp32) : "r"(xEv32.x), "r"(biasv32), "r"(0));
//			asm("vabsdiff4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(tmp32) : "r"(tmp32), "r"(0xFFFFFFFF), "r"(0));
//			for (int j = 0; j < 4; ++j) {
//				if ((tmp32 & 0xFF) == 0xFF) {
//					cudaSC[idx] = eslINFINITY;
//					return;
//				}
//				tmp32 >>= 8;
//			}

			TEST_OVERFLOW(tmp32, xEv32.x, biasv32)
			TEST_OVERFLOW(tmp32, xEv32.y, biasv32)
			TEST_OVERFLOW(tmp32, xEv32.z, biasv32)
			TEST_OVERFLOW(tmp32, xEv32.w, biasv32)

			if (bInf) printf("Error eslINFINITY\n");

			tmp = 0;
			for (int j = 0; j < 16; ++j) {
				if (xEv16[j] > tmp) tmp = xEv16[j];
			}

			tmp = SATURATE_SUB_UCHAR(tmp, tecv16);
			for (int j = 0; j < 16; ++j) {
				xEv16[j] = tmp;
			}
//			xJv16 = max(xJv16, tmp);
//			xBv16 = max(basev16, xJv16);
//			xBv16 = SATURATE_SUB_UCHAR(xBv16, tjbmv16);

	        asm("vmax4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(xEv32.x) : "r"(xEv32.x), "r"(xEv32.y), "r"(0));
	        asm("vmax4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(xEv32.z) : "r"(xEv32.z), "r"(xEv32.w), "r"(0));
	        asm("vmax4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(xEv32.x) : "r"(xEv32.x), "r"(xEv32.z), "r"(0));
	        asm("vmax4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(xEv32.x) : "r"(xEv32.x), "r"(xEv32.x >> 16), "r"(0));
	        asm("vmax4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(xEv32.x) : "r"(xEv32.x), "r"(xEv32.x >> 8), "r"(0));

	        tmp = SATURATE_SUB_UCHAR((xEv32.x & 0xFF), tecv16);

	        memset(&xEv32, tmp, 16);

	        // check
	        uint8_t *ptmp = (uint8_t *) &xEv32;
	        for (int j = 0; j < 16; ++j) {
	        	if (xEv16[j] != ptmp[j]) {
	        		printf("Error xEv!");
	        		return;
	        	}
	        }

			xJv16 = max(xJv16, tmp);
			xBv16 = max(basev16, xJv16);
			xBv16 = SATURATE_SUB_UCHAR(xBv16, tjbmv16);

			memset(&xBv32, xBv16, sizeof(uint));
		}
	}

	sc = ((float) (xJv16 - tjb_b) - (float) basev16);
	sc /= cudaScale_b;
	sc -= 3.0; /* that's ~ L \log \frac{L}{L+3}, for our NN,CC,JJ */

	cudaSC[idx] = sc;
}

#endif





