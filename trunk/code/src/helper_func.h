/* General functions for CPU interacting with GPU
 * by Lin Cheng 2013@ConcordiaUniversity
 *    forestcheng@gmail.com
 */
#ifndef HELPER_FUNC_H
#define HELPER_FUNC_H

#include "p7_config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "easel.h"
#include "esl_alphabet.h"
#include "esl_getopts.h"
#include "esl_msa.h"
#include "esl_msafile.h"
#include "esl_sq.h"
#include "esl_sqio.h"
#include "esl_stopwatch.h"

#include "hmmer.h"

typedef struct sq_entry_s
{
	int idx;
	int alignedLen;
} SQ_Entry;

extern SQ_Entry *getSortedSqEntry(ESL_SQ_BLOCK *block);

#endif
