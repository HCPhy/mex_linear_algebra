function benchmark_mex()
% BENCHMARK_MEX Benchmark MEX functions against MATLAB built-ins.
%
% This script compares the performance of custom MEX implementations
% against MATLAB built-in functions for GF(2) linear algebra operations:
%   - mela_matmul_gf2 vs gf(A,1) * gf(B,1)
%   - mela_rank_gf2 vs rank(gf(A,1))
%   - mela_null_gf2 vs gf2null(gf(A,1))
%
% The script tests various matrix sizes and generates performance plots.

% Get the directory of this script
[scriptDir, ~, ~] = fileparts(mfilename('fullpath'));
cd(scriptDir);

% Add paths
addpath(fullfile(scriptDir, 'bin'));
if exist(fullfile(scriptDir, 'gf2null'), 'dir')
    addpath(fullfile(scriptDir, 'gf2null'));
end

% Configuration
sizes = [32, 64, 128, 256, 512, 1024, 2048];

% Check dependencies
if ~exist('mela_matmul_gf2', 'file') || ...
        ~exist('mela_null_gf2', 'file') || ...
        ~exist('mela_rank_gf2', 'file')
    error('MEX functions not found. Please run compile_mex first.');
end

has_gf = exist('gf', 'file') == 2;
has_gf2null = exist('gf2null', 'file') == 2;

if ~has_gf2null && has_gf
    warning('gf2null not found. Add gf2null directory to path for null space benchmarking.');
end

fprintf('========================================\n');
fprintf('MEX Function Benchmark Suite\n');
fprintf('========================================\n\n');

%% Benchmark 1: GF(2) Matrix Multiplication
fprintf('Benchmarking mela_matmul_gf2...\n');
times_mex_matmul = zeros(size(sizes));
times_m4ri_matmul = zeros(size(sizes));
times_builtin_matmul = zeros(size(sizes));

for i = 1:length(sizes)
    n = sizes(i);
    fprintf('  Size %dx%d: ', n, n);

    % Generate random test matrices
    A = logical(randi([0, 1], n, n));
    B = logical(randi([0, 1], n, n));

    % Compute results
    C_mex = mela_matmul_gf2(A, B);
    C_m4ri = mela_matmul_m4ri(A, B);
    C_gf = gf(A, 1) * gf(B, 1);
    C_matlab = C_gf.x;

    % Verify correctness
    if ~isequal(C_mex, C_matlab)
        warning('Size %dx%d: Results do not match!', n, n);
        fprintf('  [MISMATCH] Skipping this size.\n');
        continue;
    end
    if ~isequal(C_m4ri, C_matlab)
        warning('Size %dx%d: M4RI Results do not match!', n, n);
        fprintf('  [MISMATCH] Skipping this size.\n');
        continue;
    end

    % Benchmark MEX function
    t_mex = timeit(@() mela_matmul_gf2(A, B), 1);
    times_mex_matmul(i) = t_mex;

    % Benchmark M4RI function
    t_m4ri = timeit(@() mela_matmul_m4ri(A, B), 1);
    times_m4ri_matmul(i) = t_m4ri;

    % Benchmark MATLAB GF(2) matrix multiplication
    t_builtin = timeit(@() gf(A, 1) * gf(B, 1), 1);
    times_builtin_matmul(i) = t_builtin;

    speedup = t_builtin / t_mex;
    speedup_m4ri = t_builtin / t_m4ri;
    fprintf('MEX: %.4f ms, M4RI: %.4f ms, MATLAB: %.4f ms\n', t_mex*1000, t_m4ri*1000, t_builtin*1000);
    fprintf('  Speedup vs MATLAB: MEX %.2fx, M4RI %.2fx\n', speedup, speedup_m4ri);
end

%% Benchmark 2: Rank Computation
fprintf('\nBenchmarking mela_rank_gf2...\n');
times_mex_rank = zeros(size(sizes));
times_m4ri_rank = zeros(size(sizes));
times_builtin_rank = zeros(size(sizes));

if has_gf
    fprintf('  Using rank(gf(A, 1)) as reference.\n');
    for i = 1:length(sizes)
        n = sizes(i);
        fprintf('  Size %dx%d: ', n, n);

        % Generate random test matrix
        A = logical(randi([0, 1], n, n));

        % Compute results
        r_mex = mela_rank_gf2(A);
        r_m4ri = mela_rank_m4ri(A);
        r_matlab = rank(gf(A, 1));

        % Verify correctness
        if r_mex ~= r_matlab
            warning('Size %dx%d: Rank mismatch! MEX=%d, MATLAB=%d', n, n, r_mex, r_matlab);
            fprintf('  [MISMATCH] Skipping this size.\n');
            continue;
        end
        if r_m4ri ~= r_matlab
            warning('Size %dx%d: M4RI Rank mismatch! M4RI=%d, MATLAB=%d', n, n, r_m4ri, r_matlab);
            fprintf('  [MISMATCH] Skipping this size.\n');
            continue;
        end

        % Benchmark MEX function
        t_mex = timeit(@() mela_rank_gf2(A), 1);
        times_mex_rank(i) = t_mex;

        % Benchmark M4RI function
        t_m4ri = timeit(@() mela_rank_m4ri(A), 1);
        times_m4ri_rank(i) = t_m4ri;

        % Benchmark MATLAB GF(2) rank
        t_builtin = timeit(@() rank(gf(A, 1)), 1);
        times_builtin_rank(i) = t_builtin;

        speedup = t_builtin / t_mex;
        speedup_m4ri = t_builtin / t_m4ri;
        speedup_vs_mex = t_mex / t_m4ri;
        fprintf('MEX: %.4f ms, M4RI: %.4f ms, MATLAB gf: %.4f ms\n', t_mex*1000, t_m4ri*1000, t_builtin*1000);
        fprintf('  Speedup vs MATLAB: MEX %.2fx, M4RI %.2fx\n', speedup, speedup_m4ri);
        fprintf('  M4RI vs MEX: %.2fx faster\n', speedup_vs_mex);
    end
else
    fprintf('  gf not available (needs Communications Toolbox) - benchmarking MEX only.\n');
    for i = 1:length(sizes)
        n = sizes(i);
        fprintf('  Size %dx%d: ', n, n);

        % Generate random test matrix
        A = logical(randi([0, 1], n, n));

        % Verify consistency between MEX versions
        r_mex = mela_rank_gf2(A);
        r_m4ri = mela_rank_m4ri(A);
        if r_mex ~= r_m4ri
            warning('Size %dx%d: Rank mismatch! MEX=%d, M4RI=%d', n, n, r_mex, r_m4ri);
        end

        % Benchmark MEX function
        t_mex = timeit(@() mela_rank_gf2(A), 1);
        times_mex_rank(i) = t_mex;

        % Benchmark M4RI function
        t_m4ri = timeit(@() mela_rank_m4ri(A), 1);
        times_m4ri_rank(i) = t_m4ri;

        times_builtin_rank(i) = NaN;

        speedup_vs_mex = t_mex / t_m4ri;
        fprintf('MEX: %.4f ms, M4RI: %.4f ms\n', t_mex*1000, t_m4ri*1000);
        fprintf('  M4RI vs MEX: %.2fx faster\n', speedup_vs_mex);
    end
end

%% Benchmark 3: Null Space Computation
fprintf('\nBenchmarking mela_null_gf2...\n');
times_mex_null = zeros(size(sizes));
times_m4ri_null = zeros(size(sizes));
times_bp_null = zeros(size(sizes));
times_builtin_null = zeros(size(sizes));

if has_gf
    fprintf('  Using gf2null(gf(A, 1)) as reference.\n');
    for i = 1:length(sizes)
        m = sizes(i);
        n = round(1.5 * m); % Fat matrix to ensure null space exists
        fprintf('  Size %dx%d: ', m, n);

        % Generate random test matrix
        A = logical(randi([0, 1], m, n));

        % Compute results
        Z_m4ri = mela_null_m4ri(A);
        Z_bp = mela_null_gf2(A);
        % Use gf2null for proper GF(2) null space computation
        Z_gf = gf2null(gf(A, 1));



        if ~isempty(Z_m4ri)
            prod_m4ri = mela_matmul_gf2(A, Z_m4ri);
            if ~all(prod_m4ri(:) == 0)
                warning('Size %dx%d: M4RI null space verification failed!', m, n);
                fprintf('  [MISMATCH] A*Z_m4ri is not zero. Skipping.\n');
                continue;
            end
        end

        if ~isempty(Z_bp)
            prod_bp = mela_matmul_gf2(A, Z_bp);
            if ~all(prod_bp(:) == 0)
                warning('Size %dx%d: BP null space verification failed!', m, n);
                fprintf('  [MISMATCH] A*Z_bp is not zero. Skipping.\n');
                continue;
            end
        end

        if ~isempty(Z_gf)
            prod_gf = gf(A, 1) * Z_gf;
            prod_matlab = prod_gf.x;
            if ~all(prod_matlab(:) == 0)
                warning('Size %dx%d: MATLAB null space verification failed!', m, n);
                fprintf('  [MISMATCH] A*Z_matlab is not zero. Skipping.\n');
                continue;
            end
        end

        % Benchmark M4RI function
        t_m4ri = timeit(@() mela_null_m4ri(A), 1);
        times_m4ri_null(i) = t_m4ri;

        t_bp = timeit(@() mela_null_gf2(A), 1);
        times_bp_null(i) = t_bp;

        % Benchmark MATLAB GF(2) null space using gf2null
        t_builtin = timeit(@() gf2null(gf(A, 1)), 1);
        times_builtin_null(i) = t_builtin;

        speedup_m4ri = t_builtin / t_m4ri;
        speedup_bp = t_builtin / t_bp;

        % Benchmark MEX function (mela_null_gf2)
        t_mex = timeit(@() mela_null_gf2(A), 1);
        times_mex_null(i) = t_mex;

        fprintf('MEX: %.4f ms, M4RI: %.4f ms, GF2 (BP): %.4f ms, MATLAB gf: %.4f ms\n', t_mex*1000, t_m4ri*1000, t_bp*1000, t_builtin*1000);
        fprintf('  Speedup vs MATLAB: M4RI %.2fx, GF2 (BP) %.2fx\n', speedup_m4ri, speedup_bp);
    end
else
    fprintf('  gf not available (needs Communications Toolbox) - benchmarking MEX only.\n');
    for i = 1:length(sizes)
        m = sizes(i);
        n = round(1.5 * m); % Fat matrix to ensure null space exists
        fprintf('  Size %dx%d: ', m, n);

        % Generate random test matrix
        A = logical(randi([0, 1], m, n));

        t_m4ri = timeit(@() mela_null_m4ri(A), 1);
        times_m4ri_null(i) = t_m4ri;

        t_bp = timeit(@() mela_null_gf2(A), 1);
        times_bp_null(i) = t_bp;

        times_builtin_null(i) = NaN;

        fprintf('M4RI: %.4f ms, GF2 (BP): %.4f ms\n', t_m4ri*1000, t_bp*1000);
    end
end

%% Generate Performance Plots
fprintf('\nGenerating performance plots...\n');

% Create figure with subplots
fig = figure('Position', [100, 100, 1400, 500]);

% Plot 1: Matrix Multiplication
subplot(1, 3, 1);
plot(sizes, times_mex_matmul*1000, 'o-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'MEX (Bit-Packed)');
hold on;
plot(sizes, times_m4ri_matmul*1000, '^-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'MEX (M4RI)');
plot(sizes, times_builtin_matmul*1000, 's-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'MATLAB');
hold off;
xlabel('Matrix Size (n×n)');
ylabel('Time (ms)');
title('GF(2) Matrix Multiplication');
legend('Location', 'northwest');
grid on;
set(gca, 'XScale', 'log', 'YScale', 'log');

% Plot 2: Rank Computation
subplot(1, 3, 2);
plot(sizes, times_mex_rank*1000, 'o-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'MEX (Row Reduction)');
hold on;
plot(sizes, times_m4ri_rank*1000, '^-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'MEX (M4RI)');
if has_gf
    plot(sizes, times_builtin_rank*1000, 's-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'MATLAB gf');
end
hold off;
xlabel('Matrix Size (n×n)');
ylabel('Time (ms)');
title('GF(2) Rank Computation');
legend('Location', 'northwest');
grid on;
set(gca, 'XScale', 'log', 'YScale', 'log');

% Plot 3: Null Space Computation
subplot(1, 3, 3);
plot(sizes, times_m4ri_null*1000, '^-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'MEX (M4RI)');
hold on;
plot(sizes, times_bp_null*1000, 'x-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'MEX (Bit-Packed)');
if has_gf
    plot(sizes, times_builtin_null*1000, 's-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'MATLAB gf');
end
hold off;
xlabel('Matrix Size (m, n≈1.5m)');
ylabel('Time (ms)');
title('GF(2) Null Space Computation');
if has_gf
    legend('Location', 'northwest');
end
grid on;
set(gca, 'XScale', 'log', 'YScale', 'log');

% Save figure
saveas(fig, fullfile(scriptDir, 'benchmark_results.png'));
fprintf('  Saved plot to: %s\n', fullfile(scriptDir, 'benchmark_results.png'));

%% Summary Statistics
fprintf('\n========================================\n');
fprintf('Summary\n');
fprintf('========================================\n');

fprintf('\nMatrix Multiplication (avg speedup over MATLAB):\n');
avg_speedup_matmul = mean(times_builtin_matmul ./ times_mex_matmul);
avg_speedup_m4ri = mean(times_builtin_matmul ./ times_m4ri_matmul);
fprintf('  MEX: %.2fx faster\n', avg_speedup_matmul);
fprintf('  M4RI: %.2fx faster\n', avg_speedup_m4ri);

if has_gf
    fprintf('\nRank Computation (avg speedup over MATLAB gf):\n');
    avg_speedup_rank = mean(times_builtin_rank ./ times_mex_rank);
    fprintf('  %.2fx faster\n', avg_speedup_rank);

    fprintf('\nNull Space Computation (avg speedup over MATLAB gf):\n');
    avg_speedup_m4ri_null = mean(times_builtin_null ./ times_m4ri_null);
    avg_speedup_bp_null = mean(times_builtin_null ./ times_bp_null);
    fprintf('  M4RI: %.2fx faster\n', avg_speedup_m4ri_null);
    fprintf('  BP: %.2fx faster\n', avg_speedup_bp_null);
else
    fprintf('\nRank & Null Space Computation:\n');
    fprintf('  MEX-only benchmarks (gf not available)\n');
end

%% Detailed Results Table
fprintf('\n========================================\n');
fprintf('Detailed Results\n');
fprintf('========================================\n');

fprintf('\nMatrix Multiplication:\n');
fprintf('%-10s %-15s %-15s %-15s %-10s\n', 'Size', 'MEX (ms)', 'M4RI (ms)', 'MATLAB (ms)', 'Speedup (M4RI)');
fprintf('%-10s %-15s %-15s %-15s %-10s\n', '----', '--------', '---------', '-----------', '--------------');
for i = 1:length(sizes)
    fprintf('%-10d %-15.4f %-15.4f %-15.4f %-10.2fx\n', ...
        sizes(i), times_mex_matmul(i)*1000, times_m4ri_matmul(i)*1000, times_builtin_matmul(i)*1000, ...
        times_builtin_matmul(i)/times_m4ri_matmul(i));
end

if has_gf
    fprintf('\nRank Computation:\n');
    fprintf('%-10s %-15s %-15s %-10s\n', 'Size', 'MEX (ms)', 'MATLAB (ms)', 'Speedup');
    fprintf('%-10s %-15s %-15s %-10s\n', '----', '--------', '-----------', '-------');
    for i = 1:length(sizes)
        fprintf('%-10d %-15.4f %-15.4f %-10.2fx\n', ...
            sizes(i), times_mex_rank(i)*1000, times_builtin_rank(i)*1000, ...
            times_builtin_rank(i)/times_mex_rank(i));
    end

    fprintf('\nNull Space Computation:\n');
    fprintf('%-10s %-15s %-15s %-15s %-10s\n', 'Size', 'M4RI (ms)', 'BP (ms)', 'MATLAB (ms)', 'Speedup (M4RI)');
    fprintf('%-10s %-15s %-15s %-15s %-10s\n', '----', '---------', '-------', '-----------', '--------------');
    for i = 1:length(sizes)
        fprintf('%-10s %-15.4f %-15.4f %-15.4f %-10.2fx\n', ...
            sprintf('%dx%d', sizes(i), round(1.5*sizes(i))), ...
            times_m4ri_null(i)*1000, times_bp_null(i)*1000, times_builtin_null(i)*1000, ...
            times_builtin_null(i)/times_m4ri_null(i));
    end
end

fprintf('\nBenchmark complete!\n');

end
