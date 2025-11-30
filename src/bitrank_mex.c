#include "mex.h"
#include "matrix.h"
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

/**
 * Function: bitrank_packed
 * ------------------------
 * Computes the rank of a binary matrix using Gaussian elimination over GF(2) with bit-packing.
 * This version does not use AVX2 intrinsics and relies on standard C operations.
 *
 * Parameters:
 *   mat_packed    - Pointer to the bit-packed binary matrix.
 *   n_row         - Number of rows in the matrix.
 *   n_col         - Number of columns in the matrix.
 *   words_per_row - Number of 64-bit words per row.
 *   rk            - Pointer to store the computed rank.
 */
void bitrank_packed(uint64_t *mat_packed, mwSize n_row, mwSize n_col, mwSize words_per_row, double *rk) {
    *rk = 0.0;
    mwSize col = 0;
    mwSize bits_per_word = 64;

    // Current pivot row index
    mwSize i = 0;

    while (i < n_row && col < n_col) {
        mwSize word_idx = col / bits_per_word;
        mwSize bit_idx = col % bits_per_word;
        uint64_t mask = ((uint64_t)1) << bit_idx;

        // Find a pivot in the current column, starting from row i
        mwSize pivot_row = i;
        int found = 0;
        
        // Optimization: Use pointer arithmetic for the search loop
        uint64_t *row_ptr = &mat_packed[i * words_per_row + word_idx];
        
        for (mwSize j = i; j < n_row; j++) {
            if (*row_ptr & mask) {
                pivot_row = j;
                found = 1;
                break;
            }
            row_ptr += words_per_row;
        }

        if (!found) {
            col++;
            continue;
        }

        // Swap rows if necessary
        if (pivot_row != i) {
            uint64_t *row_i = &mat_packed[i * words_per_row];
            uint64_t *row_p = &mat_packed[pivot_row * words_per_row];
            for (mwSize w = 0; w < words_per_row; w++) {
                uint64_t temp = row_i[w];
                row_i[w] = row_p[w];
                row_p[w] = temp;
            }
        }

        (*rk)++; // Increment rank

        // Eliminate rows BELOW the pivot (Gaussian Elimination)
        // We don't need to eliminate above for rank calculation.
        
        uint64_t *pivot_row_ptr = &mat_packed[i * words_per_row];
        
        // Optimization: Pre-calculate pointers for the elimination loop
        uint64_t *target_row_ptr_base = pivot_row_ptr + words_per_row;

        #pragma omp parallel for schedule(static)
        for (mwSize j = i + 1; j < n_row; j++) {
            uint64_t *target_row_ptr = target_row_ptr_base + (j - (i + 1)) * words_per_row;
            // Check if the bit at (j, col) is set
            if (target_row_ptr[word_idx] & mask) {
                // XOR the entire row j with the pivot row i
                // Unroll loop for better performance
                mwSize w = 0;
#ifdef __AVX512F__
                for (; w + 8 <= words_per_row; w += 8) {
                    __m512i vTarget = _mm512_loadu_si512((void const*)&target_row_ptr[w]);
                    __m512i vPivot  = _mm512_loadu_si512((void const*)&pivot_row_ptr[w]);
                    _mm512_storeu_si512((void*)&target_row_ptr[w], _mm512_xor_si512(vTarget, vPivot));
                }
#endif
#if defined(__aarch64__) || defined(__arm64__)
                for (; w + 2 <= words_per_row; w += 2) {
                    uint64x2_t vTarget = vld1q_u64(&target_row_ptr[w]);
                    uint64x2_t vPivot  = vld1q_u64(&pivot_row_ptr[w]);
                    vst1q_u64(&target_row_ptr[w], veorq_u64(vTarget, vPivot));
                }
#endif
                for (; w + 3 < words_per_row; w += 4) {
                    target_row_ptr[w]     ^= pivot_row_ptr[w];
                    target_row_ptr[w + 1] ^= pivot_row_ptr[w + 1];
                    target_row_ptr[w + 2] ^= pivot_row_ptr[w + 2];
                    target_row_ptr[w + 3] ^= pivot_row_ptr[w + 3];
                }
                // Handle remaining words
                for (; w < words_per_row; w++) {
                    target_row_ptr[w] ^= pivot_row_ptr[w];
                }
            }
            /* target_row_ptr += words_per_row; // Handled by calculation in parallel loop */
        }

        i++;   // Move to next row
        col++; // Move to next column
    }
}

/**
 * MEX Gateway Function
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // Input Validation
    if (nrhs != 1) {
        mexErrMsgIdAndTxt("bitrank_mex:nargin", "One input required.");
    }
    if (!mxIsLogical(prhs[0])) {
        mexErrMsgIdAndTxt("bitrank_mex:inputNotLogical", "Input must be a logical matrix.");
    }

    // Retrieve Matrix Dimensions
    mwSize n_row = mxGetM(prhs[0]);
    mwSize n_col = mxGetN(prhs[0]);

    // Calculate the number of 64-bit words per row
    mwSize bits_per_word = 64;
    mwSize words_per_row = (n_col + bits_per_word - 1) / bits_per_word;

    // Access Input Matrix Data
    const unsigned char *input_mat = (const unsigned char*)mxGetLogicals(prhs[0]);

    // Allocate Bit-Packed Matrix
    uint64_t *mat_packed = (uint64_t*)mxCalloc(n_row * words_per_row, sizeof(uint64_t));
    if (mat_packed == NULL) {
        mexErrMsgIdAndTxt("bitrank_mex:memory", "Unable to allocate memory for packed matrix.");
    }

    // Pack the input matrix into mat_packed
    for (mwSize row = 0; row < n_row; row++) {
        for (mwSize col_idx = 0; col_idx < n_col; col_idx++) {
            if (input_mat[row + col_idx * n_row]) { // MATLAB is column-major
                mwSize word_idx = col_idx / bits_per_word;
                mwSize bit_idx = col_idx % bits_per_word;
                mat_packed[row * words_per_row + word_idx] |= ((uint64_t)1) << bit_idx;
            }
        }
    }

    // Create Output for Rank
    plhs[0] = mxCreateDoubleScalar(0);
    double *rk = mxGetPr(plhs[0]);

    // Compute the Rank
    bitrank_packed(mat_packed, n_row, n_col, words_per_row, rk);

    // Free Allocated Memory
    mxFree(mat_packed);
}