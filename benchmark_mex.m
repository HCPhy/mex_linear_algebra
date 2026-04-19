function results = benchmark_mex(varargin)
% BENCHMARK_MEX Benchmark MEX functions against MATLAB GF(2) baselines.
%
% This compares the custom bit-packed and M4RI MEX implementations against
% MATLAB Communications Toolbox gf arrays when available.
%
% Usage:
%   benchmark_mex
%   results = benchmark_mex('sizes', [32 64 128], 'nullSizes', [32 64])
%   results = benchmark_mex('numThreads', 8, 'sizes', 2048)
%
% The MATLAB gf timings are reported two ways:
%   - gfComputeMs: operates on preconstructed gf arrays.
%   - gfWithConvertMs: includes gf(double(A),1) construction cost.

opts = parse_options(varargin{:});

% Get the directory of this script
[scriptDir, ~, ~] = fileparts(mfilename('fullpath'));
cd(scriptDir);

% Add paths
addpath(fullfile(scriptDir, 'bin'));
if exist(fullfile(scriptDir, 'gf2null'), 'dir')
    addpath(fullfile(scriptDir, 'gf2null'));
end

requiredMex = { ...
    'mela_matmul_gf2', ...
    'mela_matmul_m4ri', ...
    'mela_null_gf2', ...
    'mela_null_m4ri', ...
    'mela_rank_gf2', ...
    'mela_rank_m4ri' ...
    };
for k = 1:numel(requiredMex)
    if exist(requiredMex{k}, 'file') ~= 3 && exist(requiredMex{k}, 'file') ~= 2
        error('MEX function %s not found. Please run compile_mex first.', requiredMex{k});
    end
end

hasGf = exist('gf', 'file') == 2;
hasGf2Null = exist('gf2null', 'file') == 2;
hasGfRank = exist('gfrank', 'builtin') || exist('gfrank', 'file');
[threadInfo, threadCleanup] = setup_threads(opts, requiredMex); %#ok<ASGLU>
parallelInfo = setup_parallel(opts, scriptDir, threadInfo);

fprintf('========================================\n');
fprintf('MEX Function Benchmark Suite\n');
fprintf('========================================\n');
fprintf('Matrix sizes: %s\n', mat2str(opts.sizes));
fprintf('Null sizes:   %s\n', mat2str(opts.nullSizes));
fprintf('MATLAB gf:    %s\n', yes_no(hasGf));
fprintf('gfrank:       %s\n', yes_no(hasGfRank));
fprintf('gf2null:      %s\n', yes_no(hasGf2Null));
fprintf('Threads:      %s\n', threadInfo.description);
fprintf('Parallel:     %s\n', parallelInfo.description);
fprintf('Timing:       min %.3g s, rounds %d, slow-call rounds %d after %.3g s\n\n', ...
    opts.minSeconds, opts.rounds, opts.slowRounds, opts.slowCallSeconds);

if ~hasGf
    fprintf('Communications Toolbox gf is not available; gf baselines will be NaN.\n\n');
elseif ~hasGf2Null
    fprintf('gf2null helper is not available; gf null-space baseline will be NaN.\n\n');
end

fprintf('Benchmarking GF(2) matrix multiplication...\n');
results.matmul = bench_matmul(opts.sizes, hasGf, opts, parallelInfo);

fprintf('\nBenchmarking GF(2) rank...\n');
results.rank = bench_rank(opts.sizes, hasGf, hasGfRank, opts, parallelInfo);

fprintf('\nBenchmarking GF(2) null space...\n');
results.null = bench_null(opts.nullSizes, hasGf, hasGf2Null, opts, parallelInfo);

print_summary(results, hasGf, hasGf2Null, hasGfRank);
plot_results(results, scriptDir, hasGf, hasGf2Null, opts.savePlot);

fprintf('\nBenchmark complete!\n');

end

function opts = parse_options(varargin)
opts.sizes = [32 64 128 256 512 1024 2048];
opts.nullSizes = [];
opts.savePlot = true;
opts.seed = 2;
opts.minSeconds = 0.10;
opts.rounds = 5;
opts.slowCallSeconds = 0.50;
opts.slowRounds = 2;
opts.numThreads = "auto";
opts.useParallel = "off";
opts.workers = [];
opts.parallelProfile = "";

if mod(nargin, 2) ~= 0
    error('Options must be name-value pairs.');
end

for k = 1:2:nargin
    key = lower(string(varargin{k}));
    value = varargin{k + 1};
    switch key
        case "sizes"
            opts.sizes = value(:).';
        case "nullsizes"
            opts.nullSizes = value(:).';
        case "saveplot"
            opts.savePlot = logical(value);
        case "seed"
            opts.seed = value;
        case "minseconds"
            opts.minSeconds = value;
        case "rounds"
            opts.rounds = value;
        case "slowcallseconds"
            opts.slowCallSeconds = value;
        case "slowrounds"
            opts.slowRounds = value;
        case {"numthreads", "threads"}
            opts.numThreads = value;
        case {"useparallel", "parallel"}
            opts.useParallel = value;
        case {"workers", "numworkers"}
            opts.workers = value;
        case "parallelprofile"
            opts.parallelProfile = string(value);
        otherwise
            error('Unknown option: %s', key);
    end
end

if isempty(opts.nullSizes)
    opts.nullSizes = opts.sizes;
end
end

function [threadInfo, cleanupObj] = setup_threads(opts, requiredMex)
threadCount = resolve_thread_count(opts.numThreads);
threadInfo.description = "current MATLAB/OpenMP defaults";
cleanupObj = [];

if isempty(threadCount)
    return;
end

envNames = {'OMP_NUM_THREADS', 'OMP_DYNAMIC'};
oldEnv = cell(size(envNames));
for k = 1:numel(envNames)
    oldEnv{k} = getenv(envNames{k});
end

oldMaxThreads = [];
try
    oldMaxThreads = maxNumCompThreads;
    maxNumCompThreads(threadCount);
catch ME
    warning('benchmark_mex:ThreadSetup', ...
        'Could not set MATLAB maxNumCompThreads(%d): %s', threadCount, ME.message);
end

setenv('OMP_NUM_THREADS', num2str(threadCount));
setenv('OMP_DYNAMIC', 'FALSE');
clear_mex_functions(requiredMex);

threadInfo.description = sprintf('sequential, %d threads requested', threadCount);
cleanupObj = onCleanup(@() restore_threads(oldMaxThreads, envNames, oldEnv, requiredMex));
end

function threadCount = resolve_thread_count(value)
if isempty(value)
    threadCount = [];
    return;
end

if islogical(value)
    if value
        threadCount = default_thread_count();
    else
        threadCount = [];
    end
    return;
end

if isnumeric(value)
    threadCount = round(value);
    validateattributes(threadCount, {'numeric'}, {'scalar', 'integer', 'positive'});
    return;
end

mode = lower(string(value));
switch mode
    case {"auto", "max"}
        threadCount = default_thread_count();
    case {"current", "default", "off"}
        threadCount = [];
    otherwise
        threadCount = str2double(mode);
        if isnan(threadCount)
            error('Unknown numThreads value: %s', mode);
        end
        threadCount = round(threadCount);
        validateattributes(threadCount, {'numeric'}, {'scalar', 'integer', 'positive'});
end
end

function threadCount = default_thread_count()
try
    threadCount = feature('numcores');
catch
    threadCount = maxNumCompThreads;
end
threadCount = max(1, round(threadCount));
end

function clear_mex_functions(requiredMex)
for k = 1:numel(requiredMex)
    clear(requiredMex{k});
end
end

function restore_threads(oldMaxThreads, envNames, oldEnv, requiredMex)
clear_mex_functions(requiredMex);

for k = 1:numel(envNames)
    setenv(envNames{k}, oldEnv{k});
end

if ~isempty(oldMaxThreads)
    try
        maxNumCompThreads(oldMaxThreads);
    catch
    end
end
end

function parallelInfo = setup_parallel(opts, scriptDir, threadInfo)
mode = normalize_parallel_mode(opts.useParallel);
workItems = max(numel(opts.sizes), numel(opts.nullSizes));
parallelInfo.enabled = false;
parallelInfo.description = "off";

if mode == "off"
    parallelInfo.description = "off";
    return;
end

if mode == "auto" && workItems < 2
    parallelInfo.description = "off (auto: one size)";
    return;
end

if ~parallel_available()
    parallelInfo.description = "off (Parallel Computing Toolbox unavailable)";
    if mode == "on"
        warning('Parallel benchmark requested, but Parallel Computing Toolbox is unavailable. Running serially.');
    end
    return;
end

try
    pool = gcp('nocreate');
    if isempty(pool)
        if strlength(string(opts.parallelProfile)) > 0 && ~isempty(opts.workers)
            pool = parpool(char(opts.parallelProfile), opts.workers);
        elseif strlength(string(opts.parallelProfile)) > 0
            pool = parpool(char(opts.parallelProfile));
        elseif ~isempty(opts.workers)
            pool = parpool(opts.workers);
        else
            pool = parpool;
        end
    elseif ~isempty(opts.workers) && pool.NumWorkers ~= opts.workers
        warning('Using existing parallel pool with %d workers; requested %d workers.', ...
            pool.NumWorkers, opts.workers);
    end

    add_paths_on_workers(scriptDir, opts);
    parallelInfo.enabled = true;
    parallelInfo.description = sprintf('on (%d workers, sizes distributed)', pool.NumWorkers);
    if ~strcmp(threadInfo.description, "current MATLAB/OpenMP defaults")
        fprintf('Note: useParallel plus numThreads can oversubscribe CPU cores; for per-call threading, keep useParallel=false.\n');
    end
catch ME
    parallelInfo.description = "off (parallel startup failed)";
    if mode == "on"
        warning('benchmark_mex:ParallelStartupFailed', ...
            'Parallel benchmark requested but could not start a pool: %s. Running serially.', ME.message);
    else
        fprintf('Parallel auto mode could not start a pool: %s\n', ME.message);
    end
end
end

function mode = normalize_parallel_mode(value)
if islogical(value) || isnumeric(value)
    if value
        mode = "on";
    else
        mode = "off";
    end
    return;
end

mode = lower(string(value));
switch mode
    case {"auto", "on", "off"}
        return;
    case {"true", "yes", "1"}
        mode = "on";
    case {"false", "no", "0"}
        mode = "off";
    otherwise
        error('Unknown useParallel value: %s', mode);
end
end

function tf = parallel_available()
tf = exist('parpool', 'file') == 2 && exist('gcp', 'file') == 2;
try
    tf = tf && license('test', 'Distrib_Computing_Toolbox');
catch
    tf = false;
end
end

function add_paths_on_workers(scriptDir, opts)
if exist('pctRunOnAll', 'file') ~= 2
    return;
end

binDir = quote_matlab_string(fullfile(scriptDir, 'bin'));
pctRunOnAll(sprintf('addpath(''%s'')', binDir));

gf2nullDir = fullfile(scriptDir, 'gf2null');
if exist(gf2nullDir, 'dir')
    pctRunOnAll(sprintf('addpath(''%s'')', quote_matlab_string(gf2nullDir)));
end

threadCount = resolve_thread_count(opts.numThreads);
if ~isempty(threadCount)
    pctRunOnAll(sprintf('setenv(''OMP_NUM_THREADS'', ''%d''); setenv(''OMP_DYNAMIC'', ''FALSE''); maxNumCompThreads(%d);', ...
        threadCount, threadCount));
end
end

function s = quote_matlab_string(s)
s = strrep(char(s), '''', '''''');
end

function T = bench_matmul(sizes, hasGf, opts, parallelInfo)
nRows = numel(sizes);
sizeN = sizes(:);
mexBitPackedMs = nan(nRows, 1);
mexM4riMs = nan(nRows, 1);
gfComputeMs = nan(nRows, 1);
gfWithConvertMs = nan(nRows, 1);
rows = cell(nRows, 1);

if parallelInfo.enabled
    parfor i = 1:nRows
        rows{i} = bench_matmul_one(sizes(i), i, hasGf, opts);
    end
else
    for i = 1:nRows
        rows{i} = bench_matmul_one(sizes(i), i, hasGf, opts);
    end
end

for i = 1:nRows
    row = rows{i};
    mexBitPackedMs(i) = row.mexBitPackedMs;
    mexM4riMs(i) = row.mexM4riMs;
    gfComputeMs(i) = row.gfComputeMs;
    gfWithConvertMs(i) = row.gfWithConvertMs;
    fprintf('%s\n', row.log);
end

T = table(sizeN, mexBitPackedMs, mexM4riMs, gfComputeMs, gfWithConvertMs);
end

function row = bench_matmul_one(n, rowIndex, hasGf, opts)
row = struct('mexBitPackedMs', NaN, 'mexM4riMs', NaN, ...
    'gfComputeMs', NaN, 'gfWithConvertMs', NaN, 'log', '');

A = random_logical_matrix(n, n, row_seed(opts, 1000000, rowIndex));
B = random_logical_matrix(n, n, row_seed(opts, 2000000, rowIndex));

Cbp = logical(mela_matmul_gf2(A, B));
Cm4ri = logical(mela_matmul_m4ri(A, B));
if ~isequal(Cbp, Cm4ri)
    row.log = sprintf('  Size %dx%d: [MISMATCH] MEX matmul implementations disagree.', n, n);
    return;
end

if hasGf
    Agf = gf(double(A), 1);
    Bgf = gf(double(B), 1);
    Cgf = Agf * Bgf;
    if ~isequal(Cbp, logical(Cgf.x))
        row.log = sprintf('  Size %dx%d: [MISMATCH] MEX and MATLAB gf matmul disagree.', n, n);
        return;
    end
end

row.mexBitPackedMs = measure_ms(@() mela_matmul_gf2(A, B), opts);
row.mexM4riMs = measure_ms(@() mela_matmul_m4ri(A, B), opts);

if hasGf
    row.gfComputeMs = measure_ms(@() Agf * Bgf, opts);
    row.gfWithConvertMs = measure_ms(@() gf(double(A), 1) * gf(double(B), 1), opts);
end

row.log = sprintf('  Size %dx%d: BP %.4f ms, M4RI %.4f ms', ...
    n, n, row.mexBitPackedMs, row.mexM4riMs);
if hasGf
    row.log = sprintf('%s, gf %.4f ms, gf+convert %.4f ms', ...
        row.log, row.gfComputeMs, row.gfWithConvertMs);
end
end

function T = bench_rank(sizes, hasGf, hasGfRank, opts, parallelInfo)
nRows = numel(sizes);
sizeN = sizes(:);
mexBitPackedMs = nan(nRows, 1);
mexM4riMs = nan(nRows, 1);
gfComputeMs = nan(nRows, 1);
gfWithConvertMs = nan(nRows, 1);
gfrankMs = nan(nRows, 1);
rows = cell(nRows, 1);

if parallelInfo.enabled
    parfor i = 1:nRows
        rows{i} = bench_rank_one(sizes(i), i, hasGf, hasGfRank, opts);
    end
else
    for i = 1:nRows
        rows{i} = bench_rank_one(sizes(i), i, hasGf, hasGfRank, opts);
    end
end

for i = 1:nRows
    row = rows{i};
    mexBitPackedMs(i) = row.mexBitPackedMs;
    mexM4riMs(i) = row.mexM4riMs;
    gfComputeMs(i) = row.gfComputeMs;
    gfWithConvertMs(i) = row.gfWithConvertMs;
    gfrankMs(i) = row.gfrankMs;
    fprintf('%s\n', row.log);
end

T = table(sizeN, mexBitPackedMs, mexM4riMs, gfComputeMs, gfWithConvertMs, gfrankMs);
end

function row = bench_rank_one(n, rowIndex, hasGf, hasGfRank, opts)
row = struct('mexBitPackedMs', NaN, 'mexM4riMs', NaN, ...
    'gfComputeMs', NaN, 'gfWithConvertMs', NaN, 'gfrankMs', NaN, 'log', '');

A = random_logical_matrix(n, n, row_seed(opts, 3000000, rowIndex));

rbp = mela_rank_gf2(A);
rm4ri = mela_rank_m4ri(A);
if rbp ~= rm4ri
    row.log = sprintf('  Size %dx%d: [MISMATCH] MEX rank implementations disagree.', n, n);
    return;
end

if hasGf
    Agf = gf(double(A), 1);
    rgf = rank(Agf);
    if rbp ~= rgf
        row.log = sprintf('  Size %dx%d: [MISMATCH] MEX rank and rank(gf(A,1)) disagree.', n, n);
        return;
    end
end

if hasGfRank
    rgfrank = gfrank(A, 2);
    if rbp ~= rgfrank
        row.log = sprintf('  Size %dx%d: [MISMATCH] MEX rank and gfrank(A,2) disagree.', n, n);
        return;
    end
end

row.mexBitPackedMs = measure_ms(@() mela_rank_gf2(A), opts);
row.mexM4riMs = measure_ms(@() mela_rank_m4ri(A), opts);

if hasGf
    row.gfComputeMs = measure_ms(@() rank(Agf), opts);
    row.gfWithConvertMs = measure_ms(@() rank(gf(double(A), 1)), opts);
end

if hasGfRank
    row.gfrankMs = measure_ms(@() gfrank(A, 2), opts);
end

row.log = sprintf('  Size %dx%d: BP %.4f ms, M4RI %.4f ms', ...
    n, n, row.mexBitPackedMs, row.mexM4riMs);
if hasGf
    row.log = sprintf('%s, rank(gf) %.4f ms, rank(gf+convert) %.4f ms', ...
        row.log, row.gfComputeMs, row.gfWithConvertMs);
end
if hasGfRank
    row.log = sprintf('%s, gfrank %.4f ms', row.log, row.gfrankMs);
end
end

function T = bench_null(sizes, hasGf, hasGf2Null, opts, parallelInfo)
nRows = numel(sizes);
rowsM = sizes(:);
colsN = round(1.5 * rowsM);
mexBitPackedMs = nan(nRows, 1);
mexM4riMs = nan(nRows, 1);
gfComputeMs = nan(nRows, 1);
gfWithConvertMs = nan(nRows, 1);
rows = cell(nRows, 1);

if parallelInfo.enabled
    parfor i = 1:nRows
        rows{i} = bench_null_one(sizes(i), i, hasGf, hasGf2Null, opts);
    end
else
    for i = 1:nRows
        rows{i} = bench_null_one(sizes(i), i, hasGf, hasGf2Null, opts);
    end
end

for i = 1:nRows
    row = rows{i};
    mexBitPackedMs(i) = row.mexBitPackedMs;
    mexM4riMs(i) = row.mexM4riMs;
    gfComputeMs(i) = row.gfComputeMs;
    gfWithConvertMs(i) = row.gfWithConvertMs;
    fprintf('%s\n', row.log);
end

T = table(rowsM, colsN, mexBitPackedMs, mexM4riMs, gfComputeMs, gfWithConvertMs);
end

function row = bench_null_one(m, rowIndex, hasGf, hasGf2Null, opts)
n = round(1.5 * m);
row = struct('mexBitPackedMs', NaN, 'mexM4riMs', NaN, ...
    'gfComputeMs', NaN, 'gfWithConvertMs', NaN, 'log', '');

A = random_logical_matrix(m, n, row_seed(opts, 4000000, rowIndex));

Zbp = logical(mela_null_gf2(A));
Zm4ri = logical(mela_null_m4ri(A));
if size(Zbp, 1) ~= n || size(Zm4ri, 1) ~= n
    row.log = sprintf('  Size %dx%d: [MISMATCH] MEX null-space basis has wrong row count.', m, n);
    return;
end
if ~all(mela_matmul_gf2(A, Zbp) == 0, 'all') || ~all(mela_matmul_gf2(A, Zm4ri) == 0, 'all')
    row.log = sprintf('  Size %dx%d: [MISMATCH] MEX null-space verification failed.', m, n);
    return;
end

if hasGf && hasGf2Null
    Agf = gf(double(A), 1);
    Zgf = gf2null(Agf);
    if ~isempty(Zgf)
        gfProduct = Agf * Zgf;
        if ~all(gfProduct.x == 0, 'all')
            row.log = sprintf('  Size %dx%d: [MISMATCH] gf2null verification failed.', m, n);
            return;
        end
    end
end

row.mexBitPackedMs = measure_ms(@() mela_null_gf2(A), opts);
row.mexM4riMs = measure_ms(@() mela_null_m4ri(A), opts);

if hasGf && hasGf2Null
    row.gfComputeMs = measure_ms(@() gf2null(Agf), opts);
    row.gfWithConvertMs = measure_ms(@() gf2null(gf(double(A), 1)), opts);
end

row.log = sprintf('  Size %dx%d: BP %.4f ms, M4RI %.4f ms', ...
    m, n, row.mexBitPackedMs, row.mexM4riMs);
if hasGf && hasGf2Null
    row.log = sprintf('%s, gf2null %.4f ms, gf2null+convert %.4f ms', ...
        row.log, row.gfComputeMs, row.gfWithConvertMs);
end
end

function print_summary(results, hasGf, hasGf2Null, hasGfRank)
fprintf('\n========================================\n');
fprintf('Detailed Results (milliseconds)\n');
fprintf('========================================\n');

fprintf('\nMatrix multiplication:\n');
disp(results.matmul);

fprintf('\nRank:\n');
disp(results.rank);

fprintf('\nNull space:\n');
disp(results.null);

fprintf('\n========================================\n');
fprintf('Average Speedups\n');
fprintf('========================================\n');

fprintf('\nMatrix multiplication vs MATLAB gf compute-only:\n');
print_speedup('  BP', results.matmul.gfComputeMs, results.matmul.mexBitPackedMs, hasGf);
print_speedup('  M4RI', results.matmul.gfComputeMs, results.matmul.mexM4riMs, hasGf);

fprintf('\nMatrix multiplication vs MATLAB gf including conversion:\n');
print_speedup('  BP', results.matmul.gfWithConvertMs, results.matmul.mexBitPackedMs, hasGf);
print_speedup('  M4RI', results.matmul.gfWithConvertMs, results.matmul.mexM4riMs, hasGf);

fprintf('\nRank vs rank(gf(A,1)) compute-only:\n');
print_speedup('  BP', results.rank.gfComputeMs, results.rank.mexBitPackedMs, hasGf);
print_speedup('  M4RI', results.rank.gfComputeMs, results.rank.mexM4riMs, hasGf);

fprintf('\nRank vs rank(gf(A,1)) including conversion:\n');
print_speedup('  BP', results.rank.gfWithConvertMs, results.rank.mexBitPackedMs, hasGf);
print_speedup('  M4RI', results.rank.gfWithConvertMs, results.rank.mexM4riMs, hasGf);

if hasGfRank
    fprintf('\nRank vs gfrank(A,2):\n');
    print_speedup('  BP', results.rank.gfrankMs, results.rank.mexBitPackedMs, true);
    print_speedup('  M4RI', results.rank.gfrankMs, results.rank.mexM4riMs, true);
end

fprintf('\nNull space vs gf2null(gf(A,1)) compute-only:\n');
print_speedup('  BP', results.null.gfComputeMs, results.null.mexBitPackedMs, hasGf && hasGf2Null);
print_speedup('  M4RI', results.null.gfComputeMs, results.null.mexM4riMs, hasGf && hasGf2Null);

fprintf('\nNull space vs gf2null(gf(A,1)) including conversion:\n');
print_speedup('  BP', results.null.gfWithConvertMs, results.null.mexBitPackedMs, hasGf && hasGf2Null);
print_speedup('  M4RI', results.null.gfWithConvertMs, results.null.mexM4riMs, hasGf && hasGf2Null);
end

function print_speedup(label, baselineMs, candidateMs, enabled)
if ~enabled
    fprintf('%s: unavailable\n', label);
    return;
end

ratio = mean_ratio(baselineMs, candidateMs);
if isnan(ratio)
    fprintf('%s: no valid timings\n', label);
else
    fprintf('%s: %.2fx\n', label, ratio);
end
end

function ratio = mean_ratio(numerator, denominator)
valid = isfinite(numerator) & isfinite(denominator) & numerator > 0 & denominator > 0;
if ~any(valid)
    ratio = NaN;
else
    ratio = mean(numerator(valid) ./ denominator(valid));
end
end

function A = random_logical_matrix(m, n, seed)
stream = RandStream('mt19937ar', 'Seed', seed);
A = rand(stream, m, n) >= 0.5;
end

function seed = row_seed(opts, offset, rowIndex)
seed = mod(double(opts.seed) + double(offset) + double(rowIndex), 2^32 - 1);
if seed == 0
    seed = 1;
end
end

function ms = measure_ms(fn, opts)
% Use a small median loop instead of timeit so tiny MEX calls do not trigger
% warnings and each candidate is measured with the same policy.
fn();

iters = 1;
elapsed = 0;
while elapsed < 0.01 && iters < 4096
    tStart = tic;
    for k = 1:iters
        fn();
    end
    elapsed = toc(tStart);
    if elapsed < 0.01
        iters = iters * 2;
    end
end

secondsPerCall = elapsed / iters;
iters = max(1, ceil(opts.minSeconds / max(secondsPerCall, eps)));
roundCount = opts.rounds;
if opts.slowCallSeconds > 0 && secondsPerCall >= opts.slowCallSeconds
    roundCount = min(roundCount, opts.slowRounds);
end
roundCount = max(1, round(roundCount));
times = zeros(roundCount, 1);

for r = 1:roundCount
    tStart = tic;
    for k = 1:iters
        fn();
    end
    times(r) = toc(tStart) / iters;
end

ms = 1000 * median(times);
end

function plot_results(results, scriptDir, hasGf, hasGf2Null, savePlot)
if ~savePlot
    return;
end

fprintf('\nGenerating performance plots...\n');
fig = figure('Position', [100 100 1500 500]);

subplot(1, 3, 1);
plot(results.matmul.sizeN, results.matmul.mexBitPackedMs, 'o-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'MEX Bit-Packed');
hold on;
plot(results.matmul.sizeN, results.matmul.mexM4riMs, '^-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'MEX M4RI');
if hasGf
    plot(results.matmul.sizeN, results.matmul.gfComputeMs, 's-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'MATLAB gf');
    plot(results.matmul.sizeN, results.matmul.gfWithConvertMs, 'd--', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'MATLAB gf + convert');
end
hold off;
xlabel('Matrix Size (n x n)');
ylabel('Time (ms)');
title('GF(2) Matrix Multiplication');
legend('Location', 'northwest');
grid on;
set(gca, 'XScale', 'log', 'YScale', 'log');

subplot(1, 3, 2);
plot(results.rank.sizeN, results.rank.mexBitPackedMs, 'o-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'MEX Bit-Packed');
hold on;
plot(results.rank.sizeN, results.rank.mexM4riMs, '^-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'MEX M4RI');
if hasGf
    plot(results.rank.sizeN, results.rank.gfComputeMs, 's-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'rank(gf)');
    plot(results.rank.sizeN, results.rank.gfWithConvertMs, 'd--', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'rank(gf + convert)');
end
if any(isfinite(results.rank.gfrankMs))
    plot(results.rank.sizeN, results.rank.gfrankMs, 'v:', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'gfrank(A,2)');
end
hold off;
xlabel('Matrix Size (n x n)');
ylabel('Time (ms)');
title('GF(2) Rank');
legend('Location', 'northwest');
grid on;
set(gca, 'XScale', 'log', 'YScale', 'log');

subplot(1, 3, 3);
shapeLabels = arrayfun(@(m, n) sprintf('%dx%d', m, n), results.null.rowsM, results.null.colsN, 'UniformOutput', false);
x = 1:height(results.null);
plot(x, results.null.mexBitPackedMs, 'o-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'MEX Bit-Packed');
hold on;
plot(x, results.null.mexM4riMs, '^-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'MEX M4RI');
if hasGf && hasGf2Null
    plot(x, results.null.gfComputeMs, 's-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'gf2null(gf)');
    plot(x, results.null.gfWithConvertMs, 'd--', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'gf2null(gf + convert)');
end
hold off;
set(gca, 'XTick', x, 'XTickLabel', shapeLabels);
xlabel('Matrix Size (m x n)');
ylabel('Time (ms)');
title('GF(2) Null Space');
legend('Location', 'northwest');
grid on;
set(gca, 'YScale', 'log');

outPath = fullfile(scriptDir, 'benchmark_results.png');
saveas(fig, outPath);
fprintf('  Saved plot to: %s\n', outPath);
end

function s = yes_no(value)
if value
    s = 'yes';
else
    s = 'no';
end
end
