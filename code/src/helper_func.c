/* General functions for CPU interacting with GPU
 * by Lin Cheng 2013@ConcordiaUniversity
 *    forestcheng@gmail.com
 */
#include "defs.h"
#include "helper_func.h"

int compare_ascent(const void * va, const void * vb) {
	const SQ_Entry* a = (const SQ_Entry*) va;
	const SQ_Entry* b = (const SQ_Entry*) vb;

	if (a->alignedLen > b->alignedLen)
		return 1;
	if (a->alignedLen < b->alignedLen)
		return -1;

	return 0;
}

SQ_Entry *getSortedSqEntry(ESL_SQ_BLOCK *block)
{
	int status;

	SQ_Entry *sortedSQs = NULL;
	ESL_ALLOC(sortedSQs, sizeof(SQ_Entry) * block->count);

	for (int i = 0; i < block->count; ++i)
	{
		  ESL_SQ *dbsq = block->list + i;

		  int alignedLen = (dbsq->n + DBSQ_LENGTH_ALIGNED - 1) / DBSQ_LENGTH_ALIGNED;
		  sortedSQs[i].idx = i;
		  sortedSQs[i].alignedLen = alignedLen;
	}
	qsort(sortedSQs, block->count, sizeof(SQ_Entry), compare_ascent);

ERROR:
	return sortedSQs;
}











