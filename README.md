# Implementing and Accelerating HMMER3 Protein Sequence Search on CUDA-Enabled GPU
Thesis Abstract of Lin CHENG For the Master Degree of Computer Science of Concordia University @Montreal

July 2014

The recent emergence of multi-core CPU and many-core GPU architectures has made parallel computing more accessible.  Hundreds of industrial and research applications have been mapped onto GPUs to further utilize the extra computing resource.  In bioinformatics, HMMER is a set of widely used applications for sequence analysis based on Hidden Markov Model.  One of the tools in HMMER, hmmsearch, and the Smith-Waterman algorithm are two important tools for protein sequence analysis that use dynamic programming.  Both tools are particularly well-suited for many-core GPU architecture due to the parallel nature of sequence database searches.

After studying the existing research on CUDA acceleration in bioinformatics, this thesis investigated the acceleration of the key Multiple Segment Viterbi algorithm in HMMER version 3.  A fully-featured  CUDA-enabled  protein  database  search  tool cudaHmmsearch was  designed,  implemented  and  optimized.   We  demonstrated  a  variety  of  optimization  strategies  that  are  useful  for general purpose GPU-based applications.  Based on our optimization experience in parallel computing, six steps were summarized for optimizing performance using CUDA programming.

We  made  comprehensive  tests  and  analysis  for  multiple  enhancements  in  our  GPU  kernels  inorder  to  demonstrate  the  effectiveness  of  selected  approaches.   The  performance  analysis  showed that GPUs are able to deal with intensive computations, but are very sensitive to random accessesto the global memory.  The results show that our implementation achieved 2.5x speedup over the single-threaded HMMER3 CPU SSE2 implementation on average.

For details of the thesis, please view https://github.com/forestcheng/cudahmmsearch/blob/master/thesis/main.pdf
