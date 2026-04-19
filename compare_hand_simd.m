function results = compare_hand_simd(varargin)
%COMPARE_HAND_SIMD Compare explicit SIMD intrinsics against plain -O3 loops.
%
% This script compiles two temporary MEX builds for each source:
%   *_hand : current source behavior, including explicit SIMD blocks.
%   *_o3   : same source, but explicit SIMD blocks disabled with
%            MELA_NO_HAND_SIMD so the compiler sees only scalar XOR loops.
%
% All build products are written under tempdir and do not touch ./bin.
%
% Usage:
%   results = compare_hand_simd
%   results = compare_hand_simd('sizes', [64 128 256 512])

opts = parse_options(varargin{:});

[scriptDir, ~, ~] = fileparts(mfilename('fullpath'));
srcDir = fullfile(scriptDir, 'src');
buildRoot = fullfile(tempdir, 'mela_simd_compare');
srcWorkDir = fullfile(buildRoot, 'src');
outDir = fullfile(buildRoot, ['mex_' computer('arch')]);

if exist(buildRoot, 'dir')
    rmdir(buildRoot, 's');
end
mkdir(srcWorkDir);
mkdir(outDir);

sources = { ...
    'mela_matmul_gf2', ...
    'mela_matmul_m4ri', ...
    'mela_rank_gf2', ...
    'mela_rank_m4ri', ...
    'mela_null_gf2', ...
    'mela_null_m4ri' ...
    };

fprintf('Build root: %s\n', buildRoot);
fprintf('Architecture: %s\n', computer('arch'));

handFlags = mex_flags('');
o3Flags = mex_flags('-DMELA_NO_HAND_SIMD');

for i = 1:numel(sources)
    name = sources{i};
    originalPath = fullfile(srcDir, [name '.c']);
    transformedPath = fullfile(srcWorkDir, [name '_simd_toggle.c']);
    write_simd_toggle_source(originalPath, transformedPath);

    fprintf('Compiling %-18s hand-SIMD...\n', name);
    mex(handFlags{:}, '-outdir', outDir, '-output', [name '_hand'], transformedPath);

    fprintf('Compiling %-18s O3-only...\n', name);
    mex(o3Flags{:}, '-outdir', outDir, '-output', [name '_o3'], transformedPath);
end

addpath(outDir);
cleanup = onCleanup(@() rmpath(outDir)); %#ok<NASGU>

fprintf('\nRunning correctness checks...\n');
run_correctness_checks();
fprintf('Correctness checks passed.\n\n');

results = run_benchmarks(opts);
print_results(results);

end

function opts = parse_options(varargin)
opts.sizes = [64 128 256 512 1024];
opts.nullSizes = [64 128 256];
opts.minSeconds = 0.15;
opts.rounds = 5;

if mod(nargin, 2) ~= 0
    error('Options must be name-value pairs.');
end

for i = 1:2:nargin
    key = lower(string(varargin{i}));
    value = varargin{i + 1};
    switch key
        case "sizes"
            opts.sizes = value;
        case "nullsizes"
            opts.nullSizes = value;
        case "minseconds"
            opts.minSeconds = value;
        case "rounds"
            opts.rounds = value;
        otherwise
            error('Unknown option: %s', key);
    end
end
end

function flags = mex_flags(extraCflags)
arch = computer('arch');

if strcmp(arch, 'maca64')
    ompPath = '/opt/homebrew/opt/libomp';
    if exist(ompPath, 'dir')
        matlabBin = fullfile(matlabroot, 'bin', 'maca64');
        flags = {'-O', ...
            sprintf('CFLAGS="$CFLAGS -O3 -Xpreprocessor -fopenmp -I%s/include %s"', ompPath, extraCflags), ...
            sprintf('LDFLAGS="$LDFLAGS -L%s -lomp"', matlabBin)};
    else
        flags = {'-O', sprintf('CFLAGS="$CFLAGS -O3 %s"', extraCflags)};
    end
elseif strcmp(arch, 'glnxa64')
    [status, cmdout] = system('grep avx512f /proc/cpuinfo');
    if status == 0 && ~isempty(cmdout)
        flags = {'-O', ...
            sprintf('CFLAGS="$CFLAGS -O3 -mavx512f -mavx512bw -mavx512dq -fopenmp %s"', extraCflags), ...
            'LDFLAGS="$LDFLAGS -fopenmp"'};
    else
        flags = {'-O', ...
            sprintf('CFLAGS="$CFLAGS -O3 -mavx2 -fopenmp %s"', extraCflags), ...
            'LDFLAGS="$LDFLAGS -fopenmp"'};
    end
else
    flags = {'-O', sprintf('CFLAGS="$CFLAGS -O3 -mavx2 %s"', extraCflags)};
end
end

function write_simd_toggle_source(originalPath, transformedPath)
txt = fileread(originalPath);
txt = strrep(txt, '#ifdef __AVX512F__', ...
    '#if defined(__AVX512F__) && !defined(MELA_NO_HAND_SIMD)');
txt = strrep(txt, '#ifdef __AVX2__', ...
    '#if defined(__AVX2__) && !defined(MELA_NO_HAND_SIMD)');
txt = strrep(txt, '#if defined(__aarch64__) || defined(__arm64__)', ...
    '#if (defined(__aarch64__) || defined(__arm64__)) && !defined(MELA_NO_HAND_SIMD)');

fid = fopen(transformedPath, 'w');
if fid < 0
    error('Could not write transformed source: %s', transformedPath);
end
fwrite(fid, txt);
fclose(fid);
end

function run_correctness_checks()
rng(1);

A = logical(randi([0 1], 97, 65));
B = logical(randi([0 1], 65, 83));
assert(isequal(mela_matmul_gf2_hand(A, B), mela_matmul_gf2_o3(A, B)));
assert(isequal(mela_matmul_m4ri_hand(A, B), mela_matmul_m4ri_o3(A, B)));

D = randi([0 7], 43, 31);
E = randi([0 7], 31, 29);
assert(isequal(mela_matmul_gf2_hand(D, E), mela_matmul_gf2_o3(D, E)));
assert(isequal(mela_matmul_m4ri_hand(D, E), mela_matmul_m4ri_o3(D, E)));

R = logical(randi([0 1], 128, 96));
assert(mela_rank_gf2_hand(R) == mela_rank_gf2_o3(R));
assert(mela_rank_m4ri_hand(R) == mela_rank_m4ri_o3(R));

Nin = logical(randi([0 1], 48, 96));
Zhand = mela_null_gf2_hand(Nin);
Zo3 = mela_null_gf2_o3(Nin);
assert(size(Zhand, 1) == size(Nin, 2));
assert(size(Zo3, 1) == size(Nin, 2));
assert(all(mela_matmul_gf2_hand(Nin, Zhand) == 0, 'all'));
assert(all(mela_matmul_gf2_hand(Nin, Zo3) == 0, 'all'));

Zhand = mela_null_m4ri_hand(Nin);
Zo3 = mela_null_m4ri_o3(Nin);
assert(size(Zhand, 1) == size(Nin, 2));
assert(size(Zo3, 1) == size(Nin, 2));
assert(all(mela_matmul_gf2_hand(Nin, Zhand) == 0, 'all'));
assert(all(mela_matmul_gf2_hand(Nin, Zo3) == 0, 'all'));
end

function results = run_benchmarks(opts)
rng(2);

results.matmul_gf2 = bench_matmul('mela_matmul_gf2', opts.sizes, opts);
results.matmul_m4ri = bench_matmul('mela_matmul_m4ri', opts.sizes, opts);
results.rank_gf2 = bench_rank('mela_rank_gf2', opts.sizes, opts);
results.rank_m4ri = bench_rank('mela_rank_m4ri', opts.sizes, opts);
results.null_gf2 = bench_null('mela_null_gf2', opts.nullSizes, opts);
results.null_m4ri = bench_null('mela_null_m4ri', opts.nullSizes, opts);
end

function T = bench_matmul(name, sizes, opts)
hand = str2func([name '_hand']);
o3 = str2func([name '_o3']);

nRows = numel(sizes);
op = repmat({name}, nRows, 1);
sizeN = sizes(:);
handMs = zeros(nRows, 1);
o3Ms = zeros(nRows, 1);

for i = 1:nRows
    n = sizes(i);
    A = logical(randi([0 1], n, n));
    B = logical(randi([0 1], n, n));
    assert(isequal(hand(A, B), o3(A, B)));

    [handMs(i), o3Ms(i)] = measure_pair(@() hand(A, B), @() o3(A, B), opts);
end

ratioO3OverHand = o3Ms ./ handMs;
T = table(op, sizeN, handMs, o3Ms, ratioO3OverHand);
end

function T = bench_rank(name, sizes, opts)
hand = str2func([name '_hand']);
o3 = str2func([name '_o3']);

nRows = numel(sizes);
op = repmat({name}, nRows, 1);
sizeN = sizes(:);
handMs = zeros(nRows, 1);
o3Ms = zeros(nRows, 1);

for i = 1:nRows
    n = sizes(i);
    A = logical(randi([0 1], n, n));
    assert(hand(A) == o3(A));

    [handMs(i), o3Ms(i)] = measure_pair(@() hand(A), @() o3(A), opts);
end

ratioO3OverHand = o3Ms ./ handMs;
T = table(op, sizeN, handMs, o3Ms, ratioO3OverHand);
end

function T = bench_null(name, sizes, opts)
hand = str2func([name '_hand']);
o3 = str2func([name '_o3']);

nRows = numel(sizes);
op = repmat({name}, nRows, 1);
sizeN = sizes(:);
handMs = zeros(nRows, 1);
o3Ms = zeros(nRows, 1);

for i = 1:nRows
    m = sizes(i);
    n = round(1.5 * m);
    A = logical(randi([0 1], m, n));

    Zhand = hand(A);
    Zo3 = o3(A);
    assert(size(Zhand, 1) == n);
    assert(size(Zo3, 1) == n);
    assert(all(mela_matmul_gf2_hand(A, Zhand) == 0, 'all'));
    assert(all(mela_matmul_gf2_hand(A, Zo3) == 0, 'all'));

    [handMs(i), o3Ms(i)] = measure_pair(@() hand(A), @() o3(A), opts);
end

ratioO3OverHand = o3Ms ./ handMs;
T = table(op, sizeN, handMs, o3Ms, ratioO3OverHand);
end

function print_results(results)
names = fieldnames(results);
for i = 1:numel(names)
    fprintf('\n%s\n', names{i});
    disp(results.(names{i}));
end

fprintf('\nRatio meaning: ratioO3OverHand > 1 means hand-SIMD was faster; < 1 means O3-only was faster.\n');
end

function [handMs, o3Ms] = measure_pair(handFn, o3Fn, opts)
handFn();
o3Fn();

roughHand = timeit(handFn);
roughO3 = timeit(o3Fn);
iters = max(1, ceil(opts.minSeconds / max([roughHand roughO3 eps])));

handTimes = zeros(opts.rounds, 1);
o3Times = zeros(opts.rounds, 1);

for r = 1:opts.rounds
    if mod(r, 2) == 1
        handTimes(r) = measure_loop(handFn, iters);
        o3Times(r) = measure_loop(o3Fn, iters);
    else
        o3Times(r) = measure_loop(o3Fn, iters);
        handTimes(r) = measure_loop(handFn, iters);
    end
end

handMs = 1000 * median(handTimes);
o3Ms = 1000 * median(o3Times);
end

function secondsPerCall = measure_loop(fn, iters)
tStart = tic;
for i = 1:iters
    fn();
end
secondsPerCall = toc(tStart) / iters;
end
