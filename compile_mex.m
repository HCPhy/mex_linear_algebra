function compile_mex()
% COMPILE_MEX Compile all MEX files in the mex_linear_algebra directory.
%
% Usage:
%   compile_mex
%
% This script compiles:
%   - gf2_matmul_mex.c
%   - null_gf2_mex.c
%   - bitrank_mex.c
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

% 1. gf2_matmul_mex
fprintf('Compiling gf2_matmul_mex.c...\n');
try
    mex(flags{:}, '-outdir', 'bin', fullfile('src', 'gf2_matmul_mex.c'));
    fprintf('  Success.\n');
catch ME
    fprintf('  FAILED: %s\n', ME.message);
end

% 2. null_gf2_mex
fprintf('Compiling null_gf2_mex.c...\n');
try
    mex(flags{:}, '-outdir', 'bin', fullfile('src', 'null_gf2_mex.c'));
    fprintf('  Success.\n');
catch ME
    fprintf('  FAILED: %s\n', ME.message);
end

% 3. bitrank_mex
fprintf('Compiling bitrank_mex.c...\n');
try
    mex(flags{:}, '-outdir', 'bin', fullfile('src', 'bitrank_mex.c'));
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

% 1. Test gf2_matmul_mex
fprintf('Testing gf2_matmul_mex...\n');
try
    A = logical(randi([0, 1], 100, 50));
    B = logical(randi([0, 1], 50, 30));
    C_mex = gf2_matmul_mex(A, B);
    C_ref = mod(double(A) * double(B), 2);
    if isequal(C_mex, C_ref)
        fprintf('  [PASSED] Matrix multiplication matches MATLAB reference.\n');
    else
        fprintf('  [FAILED] Matrix multiplication mismatch.\n');
        error('gf2_matmul_mex failed verification.');
    end
catch ME
    fprintf('  [ERROR] %s\n', ME.message);
end

% 2. Test null_gf2_mex
fprintf('Testing null_gf2_mex...\n');
try
    A = logical(randi([0, 1], 50, 100)); % Fat matrix usually has null space
    Z = null_gf2_mex(A);

    % Check dimensions
    [~, n] = size(A);
    [nz_rows, ~] = size(Z);
    if nz_rows ~= n
        fprintf('  [FAILED] Null space matrix has wrong number of rows.\n');
    else
        % Verify A * Z = 0
        prod = gf2_matmul_mex(A, Z);
        if all(prod(:) == 0)
            fprintf('  [PASSED] A * Z is zero matrix.\n');
        else
            fprintf('  [FAILED] A * Z is NOT zero matrix.\n');
            error('null_gf2_mex failed verification.');
        end
    end
catch ME
    fprintf('  [ERROR] %s\n', ME.message);
end

% 3. Test bitrank_mex
fprintf('Testing bitrank_mex...\n');
try
    % Random matrix consistency check (Rank(A) == Rank(A'))
    A = logical(randi([0, 1], 50, 50));
    r_A = bitrank_mex(A);
    r_AT = bitrank_mex(A');
    if r_A == r_AT
        fprintf('  [PASSED] Rank consistency check (Rank(A) == Rank(A'')). Rank = %d\n', r_A);
    else
        fprintf('  [FAILED] Rank consistency mismatch: Rank(A)=%d, Rank(A'')=%d\n', r_A, r_AT);
        error('bitrank_mex failed verification.');
    end

    % Zero matrix - rank should be 0
    Z = logical(zeros(10));
    r_Z = bitrank_mex(Z);
    if r_Z == 0
        fprintf('  [PASSED] Rank of zero matrix is correct.\n');
    else
        fprintf('  [FAILED] Rank of zero matrix is %d (expected 0).\n', r_Z);
    end

    % Random matrix sanity check
    A = logical(randi([0, 1], 50, 50));
    r = bitrank_mex(A);
    if r <= 50 && r >= 0
        fprintf('  [PASSED] Rank of random matrix is within bounds.\n');
    else
        fprintf('  [FAILED] Rank of random matrix is out of bounds: %d\n', r);
    end
catch ME
    fprintf('  [ERROR] %s\n', ME.message);
end

% 4. Test Double/Integer Inputs
fprintf('Testing Double/Integer Inputs...\n');
try
    % gf2_matmul_mex with doubles
    A = randi([0, 1], 10, 10); % double by default
    B = randi([0, 1], 10, 10);
    C_mex = gf2_matmul_mex(A, B);
    C_ref = mod(A * B, 2);
    if isequal(double(C_mex), C_ref)
        fprintf('  [PASSED] gf2_matmul_mex supports double inputs.\n');
    else
        fprintf('  [FAILED] gf2_matmul_mex double input mismatch.\n');
    end

    % bitrank_mex with doubles (using odd numbers)
    A = [2, 3; 4, 5]; % [0, 1; 0, 1] mod 2. Rank should be 1.
    r = bitrank_mex(A);
    if r == 1
        fprintf('  [PASSED] bitrank_mex supports double inputs (parity check).\n');
    else
        fprintf('  [FAILED] bitrank_mex double input failed. Rank=%d (expected 1)\n', r);
    end
catch ME
    fprintf('  [ERROR] %s\n', ME.message);
end

fprintf('\nAll tests completed.\n');

end
