/* ------------------------------------------------------------
 *  mela_matmul_m4ri.c
 *  Bit-packed Boolean matrix product  C = A * B  (mod 2)
 *  using the Method of the Four Russians (M4RI).
 *
 *  A : logical or double (m x k)
 *  B : logical or double (k x n)
 *  C : logical (m x n)
 *
 *  Compile (64-bit words, SSE2/AVX2):
 *      mex -O CFLAGS="-O3 -mavx2" mela_matmul_m4ri.c
 * ----------------------------------------------------------- */
#include "mex.h"
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

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

/* M4RI Parameters */
#define M4RI_K 8
#define TABLE_SIZE (1 << M4RI_K) /* 256 */

/* Block size for rows of C to fit in L1/L2 cache */
#define BLOCK_ROWS 256

/* ------------------------------------------------------------------------
 * xor_rows: dst ^= src, over words_per_row 64-bit words
 * ------------------------------------------------------------------------ */
static inline void xor_rows(uint64_t *__restrict dst,
                            const uint64_t *__restrict src,
                            mwSize words_per_row)
{
    mwSize w = 0;

#ifdef __AVX512F__
    for (; w + 8 <= words_per_row; w += 8) {
        __m512i a = _mm512_loadu_si512((void const*)&dst[w]);
        __m512i b = _mm512_loadu_si512((void const*)&src[w]);
        _mm512_storeu_si512((void*)&dst[w], _mm512_xor_si512(a, b));
    }
#endif

#ifdef __AVX2__
    for (; w + 4 <= words_per_row; w += 4) {
        __m256i a = _mm256_loadu_si256((void const*)&dst[w]);
        __m256i b = _mm256_loadu_si256((void const*)&src[w]);
        _mm256_storeu_si256((void*)&dst[w], _mm256_xor_si256(a, b));
    }
#endif

#if defined(__aarch64__) || defined(__arm64__)
    for (; w + 2 <= words_per_row; w += 2) {
        uint64x2_t a = vld1q_u64(&dst[w]);
        uint64x2_t b = vld1q_u64(&src[w]);
        vst1q_u64(&dst[w], veorq_u64(a, b));
    }
#endif

    for (; w < words_per_row; ++w) {
        dst[w] ^= src[w];
    }
}

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

/* ------------------------------------------------------------------------
 * make_table: Build lookup table for M4RI
 * T: TABLE_SIZE x words_per_row
 * rows: pointer to the start of the k rows in Bpack
 * k: number of rows to combine (usually M4RI_K, but can be less for last block)
 * ------------------------------------------------------------------------ */
static void make_table(uint64_t *T, const uint64_t *rows, int k, mwSize words_per_row)
{
    /* T[0] is all zeros */
    memset(T, 0, (size_t)words_per_row * sizeof(uint64_t));

    /* We build the table iteratively.
       For i = 0 to k-1:
         The new entries are T[2^i ... 2^(i+1)-1].
         T[j + 2^i] = T[j] ^ rows[i]
    */
    int size = 1;
    for (int i = 0; i < k; ++i) {
        const uint64_t *row_i = rows + i * words_per_row;
        for (int j = 0; j < size; ++j) {
            uint64_t *dst = T + (size + j) * words_per_row;
            uint64_t *src = T + j * words_per_row;
            
            /* Copy T[j] to dst */
            memcpy(dst, src, (size_t)words_per_row * sizeof(uint64_t));
            /* XOR with row_i */
            xor_rows(dst, row_i, words_per_row);
        }
        size <<= 1;
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
        /* Thread-local allocation */
        /* Cblock accumulates the result for BLOCK_ROWS rows */
        uint64_t *Cblock = (uint64_t*)mxCalloc((size_t)BLOCK_ROWS * wordsC, sizeof(uint64_t));
        
        /* Table for M4RI: 2^K rows */
        uint64_t *T = (uint64_t*)mxCalloc((size_t)TABLE_SIZE * wordsC, sizeof(uint64_t));

        mwSize i_end = (i_base + BLOCK_ROWS < m) ? (i_base + BLOCK_ROWS) : m;
        mwSize block_height = i_end - i_base;

        /* Iterate over strips of columns of A (width M4RI_K) */
        for (mwSize k_base = 0; k_base < k1; k_base += M4RI_K)
        {
            int k_curr = (k_base + M4RI_K <= k1) ? M4RI_K : (int)(k1 - k_base);
            
            /* 1. Build the lookup table for this strip of B */
            /* The relevant rows of B are Bpack[k_base ... k_base + k_curr - 1] */
            make_table(T, Bpack + k_base * wordsB, k_curr, wordsB);

            /* 2. Process rows of A in the current block */
            for (mwSize i = 0; i < block_height; ++i)
            {
                /* Construct index from A(i_base + i, k_base ... k_base + k_curr - 1) */
                unsigned int idx = 0;
                mwSize row_idx = i_base + i;
                
                if (A_is_double) {
                    const double *A_col = (const double*)A_ptr + k_base * m + row_idx;
                    for (int bit = 0; bit < k_curr; ++bit) {
                        if (((int)A_col[bit * m]) & 1) {
                            idx |= (1U << bit);
                        }
                    }
                } else {
                    const mxLogical *A_col = (const mxLogical*)A_ptr + k_base * m + row_idx;
                    for (int bit = 0; bit < k_curr; ++bit) {
                        if (A_col[bit * m]) {
                            idx |= (1U << bit);
                        }
                    }
                }

                /* XOR the table entry into Cblock */
                if (idx > 0) {
                    xor_rows(Cblock + i * wordsC, T + idx * wordsC, wordsC);
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
        
        mxFree(T);
        mxFree(Cblock);
    }

    mxFree(Bpack);
}
