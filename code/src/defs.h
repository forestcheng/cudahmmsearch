/* General functions for CPU interacting with GPU
 * by Lin Cheng 2013@ConcordiaUniversity
 *    forestcheng@gmail.com
 */
#ifndef DEFS_H
#define DEFS_H

#define eslDEBUG 1

#define DBSQ_LENGTH_ALIGNED 4
#define MAX_AMINO_ACIDS     	23
#define DUMMY_AMINO_ACID   		(MAX_AMINO_ACIDS + 1)

#define GPU_SIZE 5120
#define CPU_SIZE 3500
#define BLOCK_SIZE (GPU_SIZE + CPU_SIZE)


#define SATURATE_ADD_UCHAR(x, y) (((y)>UCHAR_MAX-(x)) ? UCHAR_MAX : ((x)+(y)))
#define SATURATE_SUB_UCHAR(x, y) (((x)>(y)) ? ((x)-(y)) : 0)
//#define max(x, y) (((x) < (y)) ? (y) : (x) )

typedef struct {
#ifdef HMMER_THREADS
  ESL_WORK_QUEUE   *queue;
#endif /*HMMER_THREADS*/
  P7_BG            *bg;	         /* null model                              */
  P7_PIPELINE      *pli;         /* work pipeline                           */
  P7_TOPHITS       *th;          /* top hit results                         */
  P7_OPROFILE      *om;          /* optimized query profile                 */
} WORKER_INFO;

#endif
