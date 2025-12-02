#include "mex.h"
#include "matrix.h"
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

/* Block size for M4RI (number of pivots per block) */
#define M4RI_K      10
#define M4RI_TWO_K  (1 << M4RI_K)

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

/* ------------------------------------------------------------------------
 * m4ri_make_table:
 * T has size 2^n_pivots rows, each words_per_row 64-bit words.
 * pivots[i] points to the i-th pivot row.
 * T[0] = 0; T[idx] is XOR of pivot rows with bits set in idx.
 * ------------------------------------------------------------------------ */
static void m4ri_make_table(uint64_t *T,
                            uint64_t **pivots,
                            int n_pivots,
                            mwSize words_per_row)
{
    /* T[0] = 0 */
    memset(T, 0, (size_t)words_per_row * sizeof(uint64_t));

    int size = 1;  /* current table size: 2^k */

    for (int k = 0; k < n_pivots; ++k) {
        uint64_t *pivot = pivots[k];

        for (int j = 0; j < size; ++j) {
            uint64_t *dst = &T[(size + j) * words_per_row];
            uint64_t *src = &T[j * words_per_row];

            /* dst = src ^ pivot (small table; extra memcpy is fine) */
            memcpy(dst, src, (size_t)words_per_row * sizeof(uint64_t));
            xor_rows(dst, pivot, words_per_row);
        }
        size <<= 1;
    }
}

/* ------------------------------------------------------------------------
 * rank_m4ri: compute rank via block M4RI
 * mat: n_row x words_per_row packed GF(2) matrix (row-major)
 * ------------------------------------------------------------------------ */
static void rank_m4ri(uint64_t *mat,
                      mwSize n_row,
                      mwSize n_col,
                      mwSize words_per_row,
                      double *rk_out)
{
    double rk = 0.0;
    mwSize r = 0;  /* current top row (rank so far) */
    mwSize c = 0;  /* current block start column */

    /* Pre-allocate table for up to M4RI_K pivots: 2^K rows */
    uint64_t *T = (uint64_t*)malloc((size_t)M4RI_TWO_K *
                                    (size_t)words_per_row *
                                    sizeof(uint64_t));
    if (!T) {
        mexErrMsgTxt("Memory allocation failed for M4RI table.");
    }

    /* Pivot row pointers and column indices */
    uint64_t *pivots[M4RI_K];
    mwSize    pivot_indices[M4RI_K];

    /* Precomputed word indices and bit masks for pivot columns */
    int      pivot_word[M4RI_K];
    uint64_t pivot_mask[M4RI_K];

    while (r < n_row && c < n_col) {
        /* number of columns in this block */
        int k_curr = (c + M4RI_K <= n_col) ? M4RI_K : (int)(n_col - c);
        if (k_curr <= 0) break;

        int p_count = 0;  /* pivots found in this block */

        /* candidate window for future pivot rows */
        mwSize limit = r + M4RI_K;
        if (limit > n_row) limit = n_row;

        /* ------------------------------------------------------------
         * Step 1: find up to k_curr pivots in this block, only
         * updating rows in [r, limit) during pivoting.
         * ------------------------------------------------------------ */
        for (int k = 0; k < k_curr; ++k) {
            if (r + p_count >= n_row) break;

            mwSize curr_col = c + (mwSize)k;
            mwSize word_idx = curr_col >> 6;           /* /64 */
            mwSize bit_idx  = curr_col & 63;           /* %64 */
            uint64_t mask   = (uint64_t)1U << bit_idx;

            /* search for pivot in rows [r + p_count, limit) */
            mwSize pivot_row = r + p_count;
            int found = 0;
            
            /* 1. Search in current clean window */
            for (mwSize i = r + p_count; i < limit; ++i) {
                if (mat[i * words_per_row + word_idx] & mask) {
                    pivot_row = i;
                    found = 1;
                    break;
                }
            }
            
            /* 2. If not found, extend window and search */
            while (!found && limit < n_row) {
                mwSize old_limit = limit;
                limit = (limit + M4RI_K <= n_row) ? limit + M4RI_K : n_row;
                
                /* Clean the new batch of rows against existing pivots in this block */
                for (int p = 0; p < p_count; ++p) {
                    uint64_t *p_ptr = pivots[p];
                    mwSize p_col = pivot_indices[p];
                    mwSize p_wi = p_col >> 6;
                    uint64_t p_mask = (uint64_t)1U << (p_col & 63);
                    
                    for (mwSize j = old_limit; j < limit; ++j) {
                        if (mat[j * words_per_row + p_wi] & p_mask) {
                            xor_rows(&mat[j * words_per_row], p_ptr, words_per_row);
                        }
                    }
                }
                
                /* Now search in the new batch */
                for (mwSize i = old_limit; i < limit; ++i) {
                    if (mat[i * words_per_row + word_idx] & mask) {
                        pivot_row = i;
                        found = 1;
                        break;
                    }
                }
            }

            if (!found) continue;

            /* swap pivot row into position r + p_count */
            if (pivot_row != r + p_count) {
                uint64_t *row_a = &mat[(r + p_count) * words_per_row];
                uint64_t *row_b = &mat[pivot_row      * words_per_row];
                for (mwSize w = 0; w < words_per_row; ++w) {
                    uint64_t tmp = row_a[w];
                    row_a[w] = row_b[w];
                    row_b[w] = tmp;
                }
            }

            uint64_t *pivot_ptr = &mat[(r + p_count) * words_per_row];

            /* eliminate this pivot from rows in the candidate window
               (r + p_count + 1 ... limit-1) */
            for (mwSize j = r + p_count + 1; j < limit; ++j) {
                uint64_t *row_ptr = &mat[j * words_per_row];
                if (row_ptr[word_idx] & mask) {
                    xor_rows(row_ptr, pivot_ptr, words_per_row);
                }
            }

            pivots[p_count]       = pivot_ptr;
            pivot_indices[p_count] = curr_col;
            ++p_count;
        }

        if (p_count > 0) {
            /* ------------------------------------------------------------
             * Step 1.5: Diagonalize the pivots (Gauss-Jordan)
             * Eliminate pivots[i] from pivots[j] for j < i.
             * This ensures that pivots[j] has 0 at column pivot_indices[i].
             * This allows us to determine the coefficients for the table lookup
             * independently for each column.
             * ------------------------------------------------------------ */
            for (int i = 0; i < p_count; ++i) {
                uint64_t *pivot_i = pivots[i];
                mwSize pc = pivot_indices[i];
                mwSize wi = pc >> 6;
                uint64_t mask = (uint64_t)1U << (pc & 63);
                
                for (int j = 0; j < i; ++j) {
                    uint64_t *pivot_j = pivots[j];
                    if (pivot_j[wi] & mask) {
                        xor_rows(pivot_j, pivot_i, words_per_row);
                    }
                }
            }

            /* precompute word/mask for pivot columns (used in hot loop) */
            for (int i = 0; i < p_count; ++i) {
                mwSize pc = pivot_indices[i];
                pivot_word[i] = (int)(pc >> 6);                  /* /64 */
                pivot_mask[i] = (uint64_t)1U << (pc & 63);       /* %64 */
            }

            /* build table of all 2^p_count combinations */
            m4ri_make_table(T, pivots, p_count, words_per_row);

            /* --------------------------------------------------------
             * Step 2: use lookup table to eliminate rows below window
             * rows [limit ... n_row)
             * -------------------------------------------------------- */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (mwSize j = limit; j < n_row; ++j) {
                uint64_t *row_ptr = &mat[j * words_per_row];

                unsigned int idx = 0U;
                for (int i = 0; i < p_count; ++i) {
                    if (row_ptr[pivot_word[i]] & pivot_mask[i]) {
                        idx |= (1U << i);
                    }
                }

                if (idx != 0U) {
                    uint64_t *src = &T[(size_t)idx * words_per_row];
                    xor_rows(row_ptr, src, words_per_row);
                }
            }

            rk += (double)p_count;
            r  += (mwSize)p_count;
        }

        /* advance to next block of columns */
        c += (mwSize)k_curr;
    }

    free(T);
    *rk_out = rk;
}

/* ------------------------------------------------------------------------
 * mexFunction: rank over GF(2) using M4RI
 * Usage: r = rank_m4ri_mex(H);
 * H can be double or logical; non-zero is treated as 1.
 * ------------------------------------------------------------------------ */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    if (nrhs != 1) {
        mexErrMsgTxt("One input required.");
    }

    mwSize n_row = mxGetM(prhs[0]);
    mwSize n_col = mxGetN(prhs[0]);

    /* words per row, padded to multiple of 8 (64 bytes) */
    mwSize words_per_row_unaligned = (n_col + 63) / 64;
    mwSize words_per_row = (words_per_row_unaligned + 7) & ~((mwSize)7);

    size_t total_words = (size_t)n_row * (size_t)words_per_row;

    uint64_t *mat = (uint64_t*)calloc(total_words, sizeof(uint64_t));
    if (!mat) {
        mexErrMsgTxt("Memory allocation failed.");
    }

    /* pack MATLAB matrix into GF(2) bit-packed rows */
    if (mxIsDouble(prhs[0])) {
        double *in = mxGetPr(prhs[0]);
        for (mwSize j = 0; j < n_col; ++j) {
            for (mwSize i = 0; i < n_row; ++i) {
                int v = (int)in[i + j * n_row];
                if (v & 1) {
                    mwSize word = j >> 6;
                    mwSize bit  = j & 63;
                    mat[i * words_per_row + word] |= (uint64_t)1U << bit;
                }
            }
        }
    } else if (mxIsLogical(prhs[0])) {
        mxLogical *in = mxGetLogicals(prhs[0]);
        for (mwSize j = 0; j < n_col; ++j) {
            for (mwSize i = 0; i < n_row; ++i) {
                if (in[i + j * n_row]) {
                    mwSize word = j >> 6;
                    mwSize bit  = j & 63;
                    mat[i * words_per_row + word] |= (uint64_t)1U << bit;
                }
            }
        }
    } else {
        free(mat);
        mexErrMsgTxt("Input must be double or logical.");
    }

    plhs[0] = mxCreateDoubleScalar(0.0);
    double *rk = mxGetPr(plhs[0]);

    rank_m4ri(mat, n_row, n_col, words_per_row, rk);

    free(mat);
}