/* ------------------------------------------------------------
 *  gf2_matmul_mex.c
 *  Bit-packed Boolean matrix product  C = A * B  (mod 2).
 *
 *  A : logical or double (m x k)       (row-major in MATLAB)
 *  B : logical or double (k x n)
 *  C : logical (m x n)
 *
 *  Usage from MATLAB
 *      C = gf2_matmul_mex(A, B);
 *
 *  Compile (64-bit words, SSE2/AVX2):
 *      mex -O CFLAGS="-O3 -mavx2" gf2_matmul_mex.c
 *
 *  Optimizations:
 *  - Bit-packing of B.
 *  - Cache blocking for C (process blocks of rows).
 *  - Loop reordering: For a block of C rows, iterate k, then i.
 *    This allows reusing the packed row of B for multiple rows of A.
 * ----------------------------------------------------------- */
#include "mex.h"
#include <stdint.h>
#include <string.h>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#elif defined(__aarch64__) || defined(__arm64__)
#include <arm_neon.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#define WORD_BITS 64
#define WORDS(n)  (((n) + WORD_BITS - 1) / WORD_BITS)
#define WORD_IDX(j) ((j) >> 6)
#define BIT_POS(j)  ((j) & 63)

/* Block size for rows of C to fit in L1/L2 cache */
#define BLOCK_ROWS 256

/* pack a logical k x n into (k x WORDS(n)) uint64_t block  */
static void pack_rows_logical(const mxLogical *in, uint64_t *out,
                              mwSize rows, mwSize cols)
{
    for (mwSize i = 0; i < rows; ++i)
    {
        const mxLogical *row = in + i;            /* column-major */
        uint64_t *prow      = out + i * WORDS(cols);

        for (mwSize j = 0; j < cols; ++j)
            if (row[j * rows])                    /* A(i,j) in MATLAB */
                prow[WORD_IDX(j)] |= (uint64_t)1 << BIT_POS(j);
    }
}

/* pack a double k x n into (k x WORDS(n)) uint64_t block  */
static void pack_rows_double(const double *in, uint64_t *out,
                             mwSize rows, mwSize cols)
{
    for (mwSize i = 0; i < rows; ++i)
    {
        const double *row = in + i;               /* column-major */
        uint64_t *prow    = out + i * WORDS(cols);

        for (mwSize j = 0; j < cols; ++j)
            if (((int)row[j * rows]) & 1)         /* Check parity */
                prow[WORD_IDX(j)] |= (uint64_t)1 << BIT_POS(j);
    }
}

/* main entry ------------------------------------------------- */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    if (nrhs != 2)  mexErrMsgTxt("Need two input matrices A,B.");
    if (nlhs  > 1)  mexErrMsgTxt("Too many outputs.");

    if (mxIsSparse(prhs[0]) || mxIsSparse(prhs[1]))
        mexErrMsgTxt("Sparse matrices not supported. Convert to full using full().");

    /* dimensions */
    mwSize m = mxGetM(prhs[0]);  mwSize k1 = mxGetN(prhs[0]);
    mwSize k2 = mxGetM(prhs[1]); mwSize n  = mxGetN(prhs[1]);
    if (k1 != k2) mexErrMsgTxt("Inner dimensions must agree.");

    if ((!mxIsLogical(prhs[0]) && !mxIsDouble(prhs[0])) || 
        (!mxIsLogical(prhs[1]) && !mxIsDouble(prhs[1])))
        mexErrMsgTxt("Inputs must be logical or double matrices.");

    mwSize wordsB = WORDS(n);
    mwSize wordsC = wordsB;                     /* same width */

    /* ---- pack rows of B once (k x wordsB) ---- */
    uint64_t *Bpack = mxCalloc((size_t)k2 * wordsB, sizeof(uint64_t));
    
    if (mxIsDouble(prhs[1])) {
        pack_rows_double((const double*)mxGetData(prhs[1]), Bpack, k2, n);
    } else {
        pack_rows_logical((const mxLogical*)mxGetData(prhs[1]), Bpack, k2, n);
    }

    /* ---- allocate output logical(m x n) ---- */
    plhs[0] = mxCreateLogicalMatrix(m, n);
    mxLogical *C = (mxLogical*)mxGetData(plhs[0]);

    /* Determine type of A */
    int A_is_double = mxIsDouble(prhs[0]);
    const void *A_ptr = mxGetData(prhs[0]);

    /* ---- Main Loop: Process C in blocks of rows ---- */
    #pragma omp parallel for schedule(static)
    for (mwSize i_base = 0; i_base < m; i_base += BLOCK_ROWS)
    {
    {
        /* Thread-local allocation */
        uint64_t *Cblock = (uint64_t*)mxCalloc((size_t)BLOCK_ROWS * wordsC, sizeof(uint64_t));

        mwSize i_end = (i_base + BLOCK_ROWS < m) ? (i_base + BLOCK_ROWS) : m;
        mwSize block_height = i_end - i_base;

        /* Clear the block buffer */
        memset(Cblock, 0, (size_t)BLOCK_ROWS * wordsC * sizeof(uint64_t));

        /* 
         * Iterate over k (columns of A / rows of B).
         */
        for (mwSize k = 0; k < k1; ++k)
        {
            /* Pointer to row k of B (packed) */
            uint64_t *Bk = Bpack + k * wordsB;
            
            /* Pointer to column k of A, starting at i_base */
            /* We handle double/logical distinction here */
            
            if (A_is_double) {
                const double *Ak = (const double*)A_ptr + k * m + i_base;
                for (mwSize i = 0; i < block_height; ++i)
                {
                    if (((int)Ak[i]) & 1) /* A(i_base + i, k) is odd */
                    {
                        /* XOR B(k, :) into Cblock(i, :) */
                        uint64_t *Ci = Cblock + i * wordsC;
                        mwSize w = 0;
                        /* AVX512 / NEON / Scalar logic ... */
                        #ifdef __AVX512F__
                        for (; w + 8 <= wordsB; w += 8) {
                            __m512i vC = _mm512_loadu_si512((void const*)&Ci[w]);
                            __m512i vB = _mm512_loadu_si512((void const*)&Bk[w]);
                            _mm512_storeu_si512((void*)&Ci[w], _mm512_xor_si512(vC, vB));
                        }
                        #endif
                        #if defined(__aarch64__) || defined(__arm64__)
                        for (; w + 2 <= wordsB; w += 2) {
                            uint64x2_t vC = vld1q_u64(&Ci[w]);
                            uint64x2_t vB = vld1q_u64(&Bk[w]);
                            vst1q_u64(&Ci[w], veorq_u64(vC, vB));
                        }
                        #endif
                        for (; w < wordsB; ++w) Ci[w] ^= Bk[w];
                    }
                }
            } else {
                const mxLogical *Ak = (const mxLogical*)A_ptr + k * m + i_base;
                for (mwSize i = 0; i < block_height; ++i)
                {
                    if (Ak[i]) /* A(i_base + i, k) is true */
                    {
                        /* XOR B(k, :) into Cblock(i, :) */
                        uint64_t *Ci = Cblock + i * wordsC;
                        mwSize w = 0;
                        /* AVX512 / NEON / Scalar logic ... */
                        #ifdef __AVX512F__
                        for (; w + 8 <= wordsB; w += 8) {
                            __m512i vC = _mm512_loadu_si512((void const*)&Ci[w]);
                            __m512i vB = _mm512_loadu_si512((void const*)&Bk[w]);
                            _mm512_storeu_si512((void*)&Ci[w], _mm512_xor_si512(vC, vB));
                        }
                        #endif
                        #if defined(__aarch64__) || defined(__arm64__)
                        for (; w + 2 <= wordsB; w += 2) {
                            uint64x2_t vC = vld1q_u64(&Ci[w]);
                            uint64x2_t vB = vld1q_u64(&Bk[w]);
                            vst1q_u64(&Ci[w], veorq_u64(vC, vB));
                        }
                        #endif
                        for (; w < wordsB; ++w) Ci[w] ^= Bk[w];
                    }
                }
            }
        }

        /* Unpack the block into MATLAB output C */
        for (mwSize i = 0; i < block_height; ++i)
        {
            uint64_t *Ci = Cblock + i * wordsC;
            mwSize row_idx = i_base + i;
            
            for (mwSize j = 0; j < n; ++j)
            {
                if ((Ci[WORD_IDX(j)] >> BIT_POS(j)) & 1ULL)
                {
                    C[row_idx + j * m] = 1; /* C is column-major */
                }
            }
        }
        mxFree(Cblock);
    }
    }

    mxFree(Bpack);
}