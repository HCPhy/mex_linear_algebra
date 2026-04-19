function run_ci_tests()
%RUN_CI_TESTS Compile MEX files into a temp directory and run correctness tests.

[repoDir, ~, ~] = fileparts(mfilename('fullpath'));
cd(repoDir);

buildDir = tempname;
mkdir(buildDir);
cleanup = onCleanup(@() cleanup_build_dir(buildDir)); %#ok<NASGU>

sources = { ...
    'mela_matmul_gf2', ...
    'mela_matmul_m4ri', ...
    'mela_rank_gf2', ...
    'mela_rank_m4ri', ...
    'mela_null_gf2', ...
    'mela_null_m4ri' ...
    };

fprintf('Compiling CI MEX files into %s\n', buildDir);
flags = {'-O', 'CFLAGS="$CFLAGS -O3"'};

for i = 1:numel(sources)
    srcPath = fullfile(repoDir, 'src', [sources{i} '.c']);
    fprintf('  mex %s\n', srcPath);
    mex(flags{:}, '-outdir', buildDir, srcPath);
end

addpath(buildDir);

fprintf('Running CI correctness checks...\n');
rng(42);

test_matmul();
test_rank();
test_nullspace();
test_double_inputs();

fprintf('CI tests passed.\n');
end

function test_matmul()
A = logical(randi([0 1], 37, 23));
B = logical(randi([0 1], 23, 31));
expected = logical(mod(double(A) * double(B), 2));

assert(isequal(mela_matmul_gf2(A, B), expected));
assert(isequal(mela_matmul_m4ri(A, B), expected));

A = false(5, 7);
B = logical(randi([0 1], 7, 3));
expected = false(5, 3);

assert(isequal(mela_matmul_gf2(A, B), expected));
assert(isequal(mela_matmul_m4ri(A, B), expected));
end

function test_rank()
cases = { ...
    false(8, 8), ...
    eye(12, 'logical'), ...
    logical([1 0 1 1; 0 1 1 0; 1 1 0 1; 0 0 0 0]), ...
    logical(randi([0 1], 17, 29)), ...
    logical(randi([0 1], 64, 64)) ...
    };

for i = 1:numel(cases)
    A = cases{i};
    expected = gf2_rank_reference(A);

    assert(mela_rank_gf2(A) == expected);
    assert(mela_rank_m4ri(A) == expected);
end
end

function test_nullspace()
cases = { ...
    logical([1 0 1 1; 0 1 1 0]), ...
    logical(randi([0 1], 16, 32)), ...
    logical(randi([0 1], 31, 47)) ...
    };

for i = 1:numel(cases)
    A = cases{i};
    expectedNullity = size(A, 2) - gf2_rank_reference(A);

    check_nullspace(A, mela_null_gf2(A), expectedNullity);
    check_nullspace(A, mela_null_m4ri(A), expectedNullity);
end
end

function test_double_inputs()
A = [2 3 4; 5 6 7; 8 9 10];
B = [1 2; 3 4; 5 6];

expected = logical(mod(mod(A, 2) * mod(B, 2), 2));
assert(isequal(mela_matmul_gf2(A, B), expected));
assert(isequal(mela_matmul_m4ri(A, B), expected));

R = [2 3; 4 5];
assert(mela_rank_gf2(R) == 1);
assert(mela_rank_m4ri(R) == 1);

N = mela_null_gf2(R);
check_nullspace(logical(mod(R, 2)), N, 1);
end

function check_nullspace(A, Z, expectedNullity)
assert(islogical(Z));
assert(size(Z, 1) == size(A, 2));
assert(size(Z, 2) == expectedNullity);

product = logical(mod(double(A) * double(Z), 2));
assert(~any(product(:)));
end

function rankValue = gf2_rank_reference(A)
M = logical(mod(double(A), 2));
[m, n] = size(M);
rankValue = 0;
row = 1;

for col = 1:n
    pivot = find(M(row:m, col), 1);
    if isempty(pivot)
        continue;
    end

    pivot = pivot + row - 1;
    if pivot ~= row
        tmp = M(row, :);
        M(row, :) = M(pivot, :);
        M(pivot, :) = tmp;
    end

    for r = row + 1:m
        if M(r, col)
            M(r, :) = xor(M(r, :), M(row, :));
        end
    end

    rankValue = rankValue + 1;
    row = row + 1;
    if row > m
        break;
    end
end
end

function cleanup_build_dir(buildDir)
if contains(path, buildDir)
    rmpath(buildDir);
end

if exist(buildDir, 'dir')
    rmdir(buildDir, 's');
end
end
