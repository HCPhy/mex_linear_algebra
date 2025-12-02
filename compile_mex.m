function compile_mex()
% COMPILE_MEX Compile all MEX files in the mex_linear_algebra directory.
%
% Usage:
%   compile_mex
%
% This script compiles:
%   - mela_matmul_gf2.c
%   - mela_null_gf2.c
%   - mela_rank_gf2.c
%
% It applies optimization flags (-O3, -mavx2) for performance.

% Get the directory of this script
[scriptDir, ~, ~] = fileparts(mfilename('fullpath'));
cd(scriptDir);

fprintf('Compiling MEX files in %s...\n', scriptDir);

% Common flags
% -O: Optimize
% CFLAGS: Add -O3. Add -mavx2 ONLY for Intel/AMD (x86_64).
arch = computer('arch');
if strcmp(arch, 'maca64')
    % Apple Silicon (ARM64) - AVX2 is not supported.
    % Clang will auto-vectorize for NEON with -O3.

    % Check for Homebrew libomp (for headers only)
    omp_path = '/opt/homebrew/opt/libomp';
    if exist(omp_path, 'dir')
        fprintf('Detected Apple Silicon (maca64) with libomp headers. Using -O3 and OpenMP.\n');
        fprintf('Linking against MATLAB''s internal libomp to prevent runtime conflicts.\n');

        % Path to MATLAB's internal libomp
        matlab_bin = fullfile(matlabroot, 'bin', 'maca64');

        % Note: Apple Clang needs -Xpreprocessor -fopenmp
        % We use Homebrew for includes, but MATLAB for linking.
        flags = {'-O', ...
            sprintf('CFLAGS="$CFLAGS -O3 -Xpreprocessor -fopenmp -I%s/include"', omp_path), ...
            sprintf('LDFLAGS="$LDFLAGS -L%s -lomp"', matlab_bin)};
    else
        fprintf('Detected Apple Silicon (maca64). libomp not found. Using -O3 (single-threaded).\n');
        fprintf('  To enable OpenMP: brew install libomp\n');
        flags = {'-O', 'CFLAGS="$CFLAGS -O3"'};
    end
elseif strcmp(arch, 'glnxa64')
    % Linux (x86_64) - Check for AVX512
    [status, cmdout] = system('grep avx512f /proc/cpuinfo');
    if status == 0 && ~isempty(cmdout)
        fprintf('Detected AVX512 support. Using -O3 -mavx512f -mavx512bw -mavx512dq -fopenmp.\n');
        flags = {'-O', 'CFLAGS="$CFLAGS -O3 -mavx512f -mavx512bw -mavx512dq -fopenmp"', 'LDFLAGS="$LDFLAGS -fopenmp"'};
    else
        fprintf('Detected x86_64 (glnxa64) without AVX512. Using -O3 -mavx2 -fopenmp.\n');
        flags = {'-O', 'CFLAGS="$CFLAGS -O3 -mavx2 -fopenmp"', 'LDFLAGS="$LDFLAGS -fopenmp"'};
    end
else
    % Windows or other x86_64 - Enable AVX2.
    fprintf('Detected x86_64 (%s). Using -O3 -mavx2.\n', arch);
    flags = {'-O', 'CFLAGS="$CFLAGS -O3 -mavx2"'};
end

% 1. mela_matmul_gf2
fprintf('Compiling mela_matmul_gf2.c...\n');
try
    mex(flags{:}, '-outdir', 'bin', fullfile('src', 'mela_matmul_gf2.c'));
    fprintf('  Success.\n');
catch ME
    fprintf('  FAILED: %s\n', ME.message);
end

% 2. mela_null_gf2
fprintf('Compiling mela_null_gf2.c...\n');
try
    mex(flags{:}, '-outdir', 'bin', fullfile('src', 'mela_null_gf2.c'));
    fprintf('  Success.\n');
catch ME
    fprintf('  FAILED: %s\n', ME.message);
end

% 3. mela_rank_gf2
fprintf('Compiling mela_rank_gf2.c...\n');
try
    mex(flags{:}, '-outdir', 'bin', fullfile('src', 'mela_rank_gf2.c'));
    fprintf('  Success.\n');
catch ME
    fprintf('  FAILED: %s\n', ME.message);
end

% 4. mela_rank_m4ri
fprintf('Compiling mela_rank_m4ri.c...\n');
try
    mex(flags{:}, '-outdir', 'bin', fullfile('src', 'mela_rank_m4ri.c'));
    fprintf('  Success.\n');
catch ME
    fprintf('  FAILED: %s\n', ME.message);
end

fprintf('Done compiling.\n');

%% Testing
addpath(fullfile(scriptDir, 'bin')); % Add bin to path for testing

%% Testing
fprintf('\n----------------------------------------\n');
fprintf('Running Tests...\n');
fprintf('----------------------------------------\n');

% 1. Test mela_matmul_gf2
fprintf('Testing mela_matmul_gf2...\n');
try
    A = logical(randi([0, 1], 100, 50));
    B = logical(randi([0, 1], 50, 30));
    C_mex = mela_matmul_gf2(A, B);
    C_ref = mod(double(A) * double(B), 2);
    if isequal(C_mex, C_ref)
        fprintf('  [PASSED] Matrix multiplication matches MATLAB reference.\n');
    else
        fprintf('  [FAILED] Matrix multiplication mismatch.\n');
        error('mela_matmul_gf2 failed verification.');
    end
catch ME
    fprintf('  [ERROR] %s\n', ME.message);
end

% 2. Test mela_null_gf2
fprintf('Testing mela_null_gf2...\n');
try
    A = logical(randi([0, 1], 50, 100)); % Fat matrix usually has null space
    Z = mela_null_gf2(A);

    % Check dimensions
    [~, n] = size(A);
    [nz_rows, ~] = size(Z);
    if nz_rows ~= n
        fprintf('  [FAILED] Null space matrix has wrong number of rows.\n');
    else
        % Verify A * Z = 0
        prod = mela_matmul_gf2(A, Z);
        if all(prod(:) == 0)
            fprintf('  [PASSED] A * Z is zero matrix.\n');
        else
            fprintf('  [FAILED] A * Z is NOT zero matrix.\n');
            error('mela_null_gf2 failed verification.');
        end
    end
catch ME
    fprintf('  [ERROR] %s\n', ME.message);
end

% 3. Test mela_rank_gf2
fprintf('Testing mela_rank_gf2...\n');
try
    % Check if gfrank is available (MATLAB R2023b+)
    has_gfrank = exist('gfrank', 'builtin') || exist('gfrank', 'file');

    if has_gfrank
        fprintf('  Using gfrank(A, 2) as reference implementation.\n');

        % Test against gfrank for various matrices
        % Test 1: Random matrix
        A = logical(randi([0, 1], 50, 50));
        r_mex = mela_rank_gf2(A);
        r_ref = gfrank(A, 2);
        if r_mex == r_ref
            fprintf('  [PASSED] Random matrix rank matches gfrank. Rank = %d\n', r_mex);
        else
            fprintf('  [FAILED] Random matrix rank mismatch: mela_rank_gf2=%d, gfrank=%d\n', r_mex, r_ref);
            error('mela_rank_gf2 failed verification against gfrank.');
        end

        % Test 2: Zero matrix
        Z = false(10);
        r_Z = mela_rank_gf2(Z);
        r_Z_ref = gfrank(Z, 2);
        if r_Z == r_Z_ref && r_Z == 0
            fprintf('  [PASSED] Zero matrix rank matches gfrank. Rank = %d\n', r_Z);
        else
            fprintf('  [FAILED] Zero matrix rank: mela_rank_gf2=%d, gfrank=%d\n', r_Z, r_Z_ref);
        end

        % Test 3: Identity matrix
        I = eye(20, 'logical');
        r_I = mela_rank_gf2(I);
        r_I_ref = gfrank(I, 2);
        if r_I == r_I_ref && r_I == 20
            fprintf('  [PASSED] Identity matrix rank matches gfrank. Rank = %d\n', r_I);
        else
            fprintf('  [FAILED] Identity matrix rank: mela_rank_gf2=%d, gfrank=%d\n', r_I, r_I_ref);
        end

        % Test 4: Transpose consistency
        A = logical(randi([0, 1], 50, 50));
        r_A = mela_rank_gf2(A);
        r_AT = mela_rank_gf2(A');
        if r_A == r_AT
            fprintf('  [PASSED] Rank consistency check (Rank(A) == Rank(A'')). Rank = %d\n', r_A);
        else
            fprintf('  [FAILED] Rank consistency mismatch: Rank(A)=%d, Rank(A'')=%d\n', r_A, r_AT);
        end
    else
        fprintf('  gfrank not available. Using consistency checks.\n');

        % Random matrix consistency check (Rank(A) == Rank(A'))
        A = logical(randi([0, 1], 50, 50));
        r_A = mela_rank_gf2(A);
        r_AT = mela_rank_gf2(A');
        if r_A == r_AT
            fprintf('  [PASSED] Rank consistency check (Rank(A) == Rank(A'')). Rank = %d\n', r_A);
        else
            fprintf('  [FAILED] Rank consistency mismatch: Rank(A)=%d, Rank(A'')=%d\n', r_A, r_AT);
            error('mela_rank_gf2 failed verification.');
        end

        % Zero matrix - rank should be 0
        Z = false(10);
        r_Z = mela_rank_gf2(Z);
        if r_Z == 0
            fprintf('  [PASSED] Rank of zero matrix is correct.\n');
        else
            fprintf('  [FAILED] Rank of zero matrix is %d (expected 0).\n', r_Z);
        end

        % Random matrix sanity check
        A = logical(randi([0, 1], 50, 50));
        r = mela_rank_gf2(A);
        if r <= 50 && r >= 0
            fprintf('  [PASSED] Rank of random matrix is within bounds.\n');
        else
            fprintf('  [FAILED] Rank of random matrix is out of bounds: %d\n', r);
        end
    end
catch ME
    fprintf('  [ERROR] %s\n', ME.message);
end

% 4. Test mela_rank_m4ri
fprintf('Testing mela_rank_m4ri...\n');
try
    % Check if gfrank is available (MATLAB R2023b+)
    has_gfrank = exist('gfrank', 'builtin') || exist('gfrank', 'file');

    if has_gfrank
        fprintf('  Using gfrank(A, 2) as reference implementation.\n');

        % Test 1: Random matrix
        A = logical(randi([0, 1], 64, 64)); % Use multiple of 8 for M4RI check
        r_mex = mela_rank_m4ri(A);
        r_ref = gfrank(A, 2);
        if r_mex == r_ref
            fprintf('  [PASSED] Random matrix rank matches gfrank. Rank = %d\n', r_mex);
        else
            fprintf('  [FAILED] Random matrix rank mismatch: mela_rank_m4ri=%d, gfrank=%d\n', r_mex, r_ref);
            error('mela_rank_m4ri failed verification against gfrank.');
        end

        % Test 2: Consistency with mela_rank_gf2
        r_gf2 = mela_rank_gf2(A);
        if r_mex == r_gf2
            fprintf('  [PASSED] Consistent with mela_rank_gf2.\n');
        else
            fprintf('  [FAILED] Inconsistent with mela_rank_gf2: M4RI=%d, Basic=%d\n', r_mex, r_gf2);
        end

    else
        fprintf('  gfrank not available. Using consistency checks.\n');

        % Random matrix consistency check (Rank(A) == Rank(A'))
        A = logical(randi([0, 1], 64, 64));
        r_A = mela_rank_m4ri(A);
        r_AT = mela_rank_m4ri(A');
        if r_A == r_AT
            fprintf('  [PASSED] Rank consistency check (Rank(A) == Rank(A'')). Rank = %d\n', r_A);
        else
            fprintf('  [FAILED] Rank consistency mismatch: Rank(A)=%d, Rank(A'')=%d\n', r_A, r_AT);
            error('mela_rank_m4ri failed verification.');
        end

        % Consistency with mela_rank_gf2
        r_basic = mela_rank_gf2(A);
        if r_A == r_basic
            fprintf('  [PASSED] Consistent with mela_rank_gf2.\n');
        else
            fprintf('  [FAILED] Inconsistent with mela_rank_gf2: M4RI=%d, Basic=%d\n', r_A, r_basic);
        end
    end
catch ME
    fprintf('  [ERROR] %s\n', ME.message);
end

% 5. Test Double/Integer Inputs
fprintf('Testing Double/Integer Inputs...\n');
try
    % mela_matmul_gf2 with doubles
    A = randi([0, 1], 10, 10); % double by default
    B = randi([0, 1], 10, 10);
    C_mex = mela_matmul_gf2(A, B);
    C_ref = mod(A * B, 2);
    if isequal(double(C_mex), C_ref)
        fprintf('  [PASSED] mela_matmul_gf2 supports double inputs.\n');
    else
        fprintf('  [FAILED] mela_matmul_gf2 double input mismatch.\n');
    end

    % mela_rank_gf2 with doubles (using odd numbers)
    A = [2, 3; 4, 5]; % [0, 1; 0, 1] mod 2. Rank should be 1.
    r = mela_rank_gf2(A);
    if r == 1
        fprintf('  [PASSED] mela_rank_gf2 supports double inputs (parity check).\n');
    else
        fprintf('  [FAILED] mela_rank_gf2 double input failed. Rank=%d (expected 1)\n', r);
    end
catch ME
    fprintf('  [ERROR] %s\n', ME.message);
end

fprintf('\nAll tests completed.\n');

end
