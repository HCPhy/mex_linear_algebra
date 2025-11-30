/*
 * null_gf2_mex.c
 *
 * Compute a basis for the null space of the binary matrix H (over GF(2))
 * using the Method of the Four Russians (M4RI) for fast Gaussian elimination.
 *
 * Usage:
 *   N = null_gf2_mex(H);
 *
 * Compile:
 *   mex -O CFLAGS="-O3 -mavx2" null_gf2_mex.c
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

/* M4RI Parameters */
#define M4RI_K 8
#define M4RI_TWO_K (1 << M4RI_K)

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
 * GF(2) RREF on a bitmask-based matrix using M4RI
 * Input: 
 *   M: matrix data, m rows, n columns
 * Output:
 *   M will be transformed into RREF
 *   pivots: array of pivot column indices
 * Returns number of pivots
 */
static int gf2_rref_m4ri(uint64_t *M, mwSize m, mwSize n, int *pivots) {
    mwSize wc = word_count(n);
    mwSize pivot_row = 0;
    int pivot_count = 0;

    /* Process columns in blocks of size k (or less) */
    /* Note: M4RI is typically for eliminating multiple columns at once.
       Here we do a simplified version:
       1. Find k pivots.
       2. Build a table of all linear combinations of these k pivot rows.
       3. Eliminate these k columns from all other rows using the table.
    */

    for (mwSize col_base = 0; col_base < n && pivot_row < m; ) {
        /* 
         * Phase 1: Gaussian Elimination for a small block of columns 
         * to find up to M4RI_K pivots.
         */
        int k_pivots = 0;
        int current_pivots[M4RI_K]; /* Column indices of pivots found in this block */
        mwSize current_pivot_rows[M4RI_K]; /* Row indices of these pivots */

        mwSize c = col_base;
        /* Try to find up to M4RI_K pivots */
        while (c < n && k_pivots < M4RI_K && pivot_row + k_pivots < m) {
            /* Search for pivot in column c starting from row (pivot_row + k_pivots) */
            mwSize start_row = pivot_row + k_pivots;
            mwSize pivot_r = m;
            
            uint64_t *row_ptr = M + start_row * wc;
            for (mwSize r = start_row; r < m; r++) {
                if (get_bit(row_ptr, c)) {
                    pivot_r = r;
                    break;
                }
                row_ptr += wc;
            }

            if (pivot_r != m) {
                /* Found a pivot */
                if (pivot_r != start_row) {
                    swap_rows(M + pivot_r*wc, M + start_row*wc, wc);
                }
                
                /* Record pivot */
                current_pivots[k_pivots] = (int)c;
                current_pivot_rows[k_pivots] = start_row;
                pivots[pivot_count++] = (int)c;
                
                /* Eliminate this column from subsequent rows *within the block* 
                   to ensure we find independent pivots */
                uint64_t *curr_row_ptr = M + start_row * wc;
                uint64_t *next_row_ptr = curr_row_ptr + wc;
                for (mwSize r = start_row + 1; r < m; r++) {
                     /* Only need to check if we haven't found k pivots yet?
                        Actually, standard GE requires eliminating below. 
                        We do it lazily or eagerly. Eagerly is simpler for finding next pivots.
                     */
                     if (get_bit(next_row_ptr, c)) {
                         xor_row(next_row_ptr, curr_row_ptr, wc);
                     }
                     next_row_ptr += wc;
                }

                k_pivots++;
            }
            c++;
        }

        if (k_pivots == 0) {
            /* No pivots found in remaining columns? Should not happen if loop condition is right
               unless we ran out of columns without finding pivots.
               If c reached n, we are done.
            */
            if (c >= n) break;
            /* Otherwise, just advance col_base */
            col_base = c;
            continue;
        }

        /* 
         * Phase 2: M4RI Table Construction
         * Build a table T of size 2^k_pivots.
         * T[idx] = linear combination of pivot rows corresponding to bits set in idx.
         */
        int num_entries = 1 << k_pivots;
        uint64_t *T = (uint64_t *)mxCalloc(num_entries * wc, sizeof(uint64_t));
        
        /* T[0] is all zeros (already calloc'd) */
        
        /* Gray code enumeration or simple DP to fill T */
        /* T[i] = T[i without msb] ^ row[msb] */
        for (int i = 1; i < num_entries; i++) {
            /* Find the highest set bit (or any set bit) to decompose */
            int bit = 0;
            while (!((i >> bit) & 1)) bit++;
            
            /* T[i] = T[i ^ (1<<bit)] ^ pivot_row[bit] */
            int prev = i ^ (1 << bit);
            
            /* Copy T[prev] to T[i] */
            memcpy(T + i*wc, T + prev*wc, wc * sizeof(uint64_t));
            
            /* XOR with the corresponding pivot row */
            /* The 'bit'-th pivot row is at current_pivot_rows[bit] */
            xor_row(T + i*wc, M + current_pivot_rows[bit]*wc, wc);
        }

        /* 
         * Phase 3: Block Elimination
         * Eliminate these k columns from ALL rows (except the pivot rows themselves).
         * For each row r, we form an index 'idx' based on the bits at the pivot columns.
         * Then row[r] ^= T[idx].
         */
        
        /* We need to eliminate from rows 0 to m-1, excluding current_pivot_rows */
        /* ... comments ... */

        #pragma omp parallel for schedule(static)
        for (mwSize r = 0; r < m; r++) {
            /* Check if r is one of the current pivot rows */
            int is_pivot_row = 0;
            if (r >= pivot_row && r < pivot_row + k_pivots) is_pivot_row = 1;
            
            if (!is_pivot_row) {
                /* Construct index into table T */
                int idx = 0;
                uint64_t *row_ptr = M + r * wc;
                
                for (int j = 0; j < k_pivots; j++) {
                    if (get_bit(row_ptr, current_pivots[j])) {
                        idx |= (1 << j);
                    }
                }

                if (idx > 0) {
                    xor_row(row_ptr, T + idx*wc, wc);
                }
            }
        }

        mxFree(T);
        
        pivot_row += k_pivots;
        col_base = c; /* Continue from where we left off */
    }

    return pivot_count;
}

/*
 * null_gf2_mex:
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
    int pivot_count = gf2_rref_m4ri(M, m, n, pivots);

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

    /* Back-substitution:  
       For each free variable f:
         - Initialize v with v[f] = 1 and others = 0
         - Solve for pivot columns using the RREF
         
       The RREF form after gf2_rref_m4ri is such that:
       Each pivot row corresponds to one pivot. Pivot i in pivots[i].
       The pivot row is i (0-based) since we processed them in order.
    */

    /* We'll represent v as a bitmask as well for speed */
    uint64_t *v = (uint64_t *)mxCalloc(wc, sizeof(uint64_t));
    for (int fv_i = 0; fv_i < free_count; fv_i++) {
        /* Reset v to zero */
        for (mwSize w = 0; w < wc; w++) {
            v[w] = 0ULL;
        }
        /* Set the free variable bit */
        set_bit(v, free_vars[fv_i]);

        /* Back-substitute: 
           We know pivot rows: i-th pivot in row i.
           Solve from bottom pivot up:
        */
        for (int pi = pivot_count - 1; pi >= 0; pi--) {
            int c = pivots[pi]; /* pivot column */
            /* Compute the sum of known variables in that pivot row */
            /* M[row=pi, :] represents equation:
               x[c] = sum(M[pi, all other set bits]*x[that col])
             */
            uint64_t sumvec[wc];
            memcpy(sumvec, M + pi*wc, wc*sizeof(uint64_t));

            /* Clear the pivot bit itself to get the other variables */
            sumvec[c >> 6] &= ~((uint64_t)1 << (c & 63));

            /* The value of x[c] is the XOR of all x[cols] where M[pi,cols] = 1 */
            /* XOR sum of v & sumvec (AND first to select relevant bits) */
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

            /* Compute parity of acc */
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
        /* v is a bitmask of length n, N is n-by-free_count column major */
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