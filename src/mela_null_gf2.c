/*
 * mela_null_gf2.c
 *
 * Compute a basis for the null space of the binary matrix H (over GF(2))
 * using Standard Gaussian Elimination (Bit-Packed).
 *
 * Usage:
 *   N = mela_null_gf2(H);
 *
 * Compile:
 *   mex -O CFLAGS="-O3 -mavx2" mela_null_gf2.c
 */

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

/* Get the number of 64-bit words needed to store n bits */
static inline mwSize word_count(mwSize n) {
    return (n + 63U) >> 6; /* ceil(n/64) */
}

/* Set the bit at position col in the row */
static inline void set_bit(uint64_t *row, mwSize col) {
    row[col >> 6] |= (uint64_t)1 << (col & 63);
}

/* Check if a bit at 'col' is set in 'row' */
static inline int get_bit(const uint64_t *row, mwSize col) {
    return (row[col >> 6] >> (col & 63)) & 1U;
}

/* XOR two rows: dest ^= src (word-wise) with unrolling */
static inline void xor_row(uint64_t *dest, const uint64_t *src, mwSize wc) {
    mwSize i = 0;
#ifdef __AVX512F__
    for (; i + 8 <= wc; i += 8) {
        __m512i vDest = _mm512_loadu_si512((void const*)&dest[i]);
        __m512i vSrc  = _mm512_loadu_si512((void const*)&src[i]);
        _mm512_storeu_si512((void*)&dest[i], _mm512_xor_si512(vDest, vSrc));
    }
#endif
#if defined(__aarch64__) || defined(__arm64__)
    for (; i + 2 <= wc; i += 2) {
        uint64x2_t vDest = vld1q_u64(&dest[i]);
        uint64x2_t vSrc  = vld1q_u64(&src[i]);
        vst1q_u64(&dest[i], veorq_u64(vDest, vSrc));
    }
#endif
    for (; i + 3 < wc; i += 4) {
        dest[i]     ^= src[i];
        dest[i + 1] ^= src[i + 1];
        dest[i + 2] ^= src[i + 2];
        dest[i + 3] ^= src[i + 3];
    }
    for (; i < wc; i++) {
        dest[i] ^= src[i];
    }
}

/* Swap two rows of wc words each */
static inline void swap_rows(uint64_t *A, uint64_t *B, mwSize wc) {
    for (mwSize i = 0; i < wc; i++) {
        uint64_t temp = A[i];
        A[i] = B[i];
        B[i] = temp;
    }
}

/* 
 * GF(2) RREF on a bitmask-based matrix using Standard Gaussian Elimination
 * Input: 
 *   M: matrix data, m rows, n columns
 * Output:
 *   M will be transformed into RREF
 *   pivots: array of pivot column indices
 * Returns number of pivots
 */
static int gf2_rref_standard(uint64_t *M, mwSize m, mwSize n, int *pivots) {
    mwSize wc = word_count(n);
    mwSize pivot_row = 0;
    int pivot_count = 0;

    for (mwSize c = 0; c < n && pivot_row < m; c++) {
        /* Find pivot in column c, starting from pivot_row */
        mwSize pivot_r = m;
        uint64_t *row_ptr = M + pivot_row * wc;
        
        for (mwSize r = pivot_row; r < m; r++) {
            if (get_bit(row_ptr, c)) {
                pivot_r = r;
                break;
            }
            row_ptr += wc;
        }

        if (pivot_r != m) {
            /* Found a pivot */
            if (pivot_r != pivot_row) {
                swap_rows(M + pivot_r*wc, M + pivot_row*wc, wc);
            }
            
            pivots[pivot_count++] = (int)c;
            
            /* Eliminate this column from ALL other rows (RREF) */
            /* We can parallelize this loop */
            #pragma omp parallel for schedule(static)
            for (mwSize r = 0; r < m; r++) {
                if (r != pivot_row) {
                    uint64_t *curr_row = M + r * wc;
                    if (get_bit(curr_row, c)) {
                        xor_row(curr_row, M + pivot_row * wc, wc);
                    }
                }
            }
            
            pivot_row++;
        }
    }

    return pivot_count;
}

/*
 * mela_null_bitpacked:
 *
 * Input:
 *   H: m-by-n matrix over GF(2) (logical or double mod 2)
 *
 * Output:
 *   N: n-by-(num_free_vars) logical matrix representing a basis for the nullspace over GF(2)
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 1) {
        mexErrMsgTxt("One input required: H.");
    }
    if (nlhs > 1) {
        mexErrMsgTxt("One output required.");
    }

    const mxArray *H_in = prhs[0];
    if (mxIsSparse(H_in)) {
        mexErrMsgTxt("Sparse matrices not supported. Convert to full using full().");
    }
    if (!mxIsDouble(H_in) && !mxIsLogical(H_in)) {
        mexErrMsgTxt("H must be double or logical.");
    }

    mwSize m = mxGetM(H_in);
    mwSize n = mxGetN(H_in);
    mwSize wc = word_count(n);

    /* Allocate matrix M as bitmasks */
    uint64_t *M = (uint64_t *)mxCalloc(m*wc, sizeof(uint64_t));
    if (M == NULL) {
        mexErrMsgTxt("Out of memory.");
    }

    /* Convert H to bitmask form */
    if (mxIsDouble(H_in)) {
        double *H_d = (double *)mxGetData(H_in);
        for (mwSize j = 0; j < n; j++) {
            for (mwSize i = 0; i < m; i++) {
                if (((int)H_d[i + j*m]) & 1) {
                    set_bit(M + i*wc, j);
                }
            }
        }
    } else {
        bool *H_b = (bool *)mxGetData(H_in);
        for (mwSize j = 0; j < n; j++) {
            for (mwSize i = 0; i < m; i++) {
                if (H_b[i + j*m]) {
                    set_bit(M + i*wc, j);
                }
            }
        }
    }

    /* Array to hold pivot positions */
    int *pivots = (int *)mxCalloc((m < n ? m : n), sizeof(int));
    int pivot_count = gf2_rref_standard(M, m, n, pivots);

    /* Determine which columns are pivots */
    bool *isPivot = (bool *)mxCalloc(n, sizeof(bool));
    for (int i = 0; i < pivot_count; i++) {
        isPivot[pivots[i]] = true;
    }

    /* Count free variables */
    int free_count = 0;
    for (mwSize col = 0; col < n; col++) {
        if (!isPivot[col]) {
            free_count++;
        }
    }

    /* Create output N: n-by-free_count (logical) */
    plhs[0] = mxCreateLogicalMatrix(n, free_count);
    bool *N = (bool *)mxGetData(plhs[0]);

    /* Extract the free variables */
    int *free_vars = (int *)mxCalloc(free_count, sizeof(int));
    {
        int idx = 0;
        for (mwSize col = 0; col < n; col++) {
            if (!isPivot[col]) {
                free_vars[idx++] = (int)col;
            }
        }
    }

    /* Back-substitution */
    uint64_t *v = (uint64_t *)mxCalloc(wc, sizeof(uint64_t));
    for (int fv_i = 0; fv_i < free_count; fv_i++) {
        /* Reset v to zero */
        for (mwSize w = 0; w < wc; w++) {
            v[w] = 0ULL;
        }
        /* Set the free variable bit */
        set_bit(v, free_vars[fv_i]);

        /* Back-substitute */
        for (int pi = pivot_count - 1; pi >= 0; pi--) {
            int c = pivots[pi]; /* pivot column */
            
            uint64_t sumvec[wc];
            memcpy(sumvec, M + pi*wc, wc*sizeof(uint64_t));

            /* Clear the pivot bit itself */
            sumvec[c >> 6] &= ~((uint64_t)1 << (c & 63));

            /* XOR sum */
            uint64_t acc = 0ULL;
            mwSize w = 0;
            for (; w + 3 < wc; w += 4) {
                acc ^= (v[w] & sumvec[w]);
                acc ^= (v[w+1] & sumvec[w+1]);
                acc ^= (v[w+2] & sumvec[w+2]);
                acc ^= (v[w+3] & sumvec[w+3]);
            }
            for (; w < wc; w++) {
                acc ^= (v[w] & sumvec[w]);
            }

            /* Compute parity */
            int parity = 0;
            while (acc) {
                acc &= (acc - 1); 
                parity = !parity;
            }

            /* Set v[c] = parity */
            if (parity) {
                v[c >> 6] ^= ((uint64_t)1 << (c & 63));
            } else {
                v[c >> 6] &= ~((uint64_t)1 << (c & 63));
            }
        }

        /* Store v into N(:, fv_i) */
        for (mwSize col = 0; col < n; col++) {
            N[col + fv_i*n] = get_bit(v, col);
        }
    }

    mxFree(M);
    mxFree(pivots);
    mxFree(isPivot);
    mxFree(free_vars);
    mxFree(v);
}
