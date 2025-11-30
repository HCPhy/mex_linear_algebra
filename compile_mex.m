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
    mex(flags{:}, 'gf2_matmul_mex.c');
    fprintf('  Success.\n');
catch ME
    fprintf('  FAILED: %s\n', ME.message);
end

% 2. null_gf2_mex
fprintf('Compiling null_gf2_mex.c...\n');
try
    mex(flags{:}, 'null_gf2_mex.c');
    fprintf('  Success.\n');
catch ME
    fprintf('  FAILED: %s\n', ME.message);
end

% 3. bitrank_mex
fprintf('Compiling bitrank_mex.c...\n');
try
    mex(flags{:}, 'bitrank_mex.c');
    fprintf('  Success.\n');
catch ME
    fprintf('  FAILED: %s\n', ME.message);
end


fprintf('Done.\n');

end
