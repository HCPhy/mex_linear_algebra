function Z = gf2null(A)
%GF2NULL   Null space over GF(2)
%   Z = GF2NULL(A) is an orthonormal basis for the null space of A 
%   over GF(2). That is,  A*Z becomes zero matrix.
%   size(Z,2) is the nullity of A.
%
%   Example:
%
%      A = gf( randi([0 1], 3 ));
%      A = GF(2) array. 
%      Array elements = 
%        0   0   1
%        1   1   1
%        1   1   0
%
%      Z = gf2null(A)
%      Z = GF(2) array. 
%      Array elements = 
%        1
%        1
%        0
%
%      A * Z
%      ans = GF(2) array. 
%      Array elements = 
%        0
%        0
%        0
%
%   Class support for input A:
%      gf
%
%   See also GF2RREF, RANK, NULL.

%
%   Modified by bjc97r@inu.ac.kr on 2018-11-24
%   Based on gfnull.m by Mark Wilde, 201
%         at matlabcentral/fileexchange/28633-gfnull
%   Based on null.m by MathWorks
%   Copyright 1984-2004 The MathWorks, Inc.
%   $Revision: 5.12.4.2 $  $Date: 2004/04/10 23:30:03 $
%

n = size(A, 2);
assert( isa(A,'gf'), 'gfnull expects GF(2) matrix as input.');

[R, pivcol] = gf2rref(A); % reduced row-echelon form of A over GF(2)
r = length(pivcol); % rank
nopiv = 1:n;
nopiv(pivcol) = [];
Z = gf(zeros(n,n-r));
if n > r
    Z(nopiv,:) = gf(eye(n-r,n-r));
    if r > 0
        Z(pivcol,:) = -R(1:r,nopiv);
    end
end

