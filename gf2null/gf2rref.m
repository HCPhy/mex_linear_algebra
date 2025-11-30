function [A,jb] = gf2rref(A)
%GF2RREF   Reduced row echelon form over GF(2)
%   R = GF2RREF(A) produces the reduced row echelon form of A over GF(2).
%
%   [R,jb] = GFRREF(A) also returns a vector, jb, so that:
%       r = length(jb) is this algorithm's idea of the rank of A,
%       x(jb) are the bound variables in a linear system, Ax = b,
%       A(:,jb) is a basis for the range of A,
%       R(1:r,jb) is the r-by-r identity matrix.
%
%   Example:
%
%      A = gf( randi([0 1], 4 ))
%
%      A = GF(2) array. 
%      Array elements = 
%         0   0   1   1
%         0   1   0   0
%         0   1   1   1
%         0   1   0   0
%      
%      [R, jb] = gf2rref(A)
%
%      R = GF(2) array. 
%      Array elements = 
%         0   1   0   0
%         0   0   1   1
%         0   0   0   0
%         0   0   0   0
%
%      jb =
%           2     3
%
%   Class support for input A:
%      gf
%
%   See also GF2NULL, RANK, RREF.
%

%
%   Modified by bjc97r@inu.ac.kr on 2018-11-24
%
%   Based on gfrref.m by Mark Wilde, 2010
%         at matlabcentral/fileexchange/28633-gfnull
%   Based on rref.m by MathWorks
%   Copyright 1984-2017 The MathWorks, Inc. 

assert(isa(A,'gf'),'A is type %s, not gf.',class(A))
[m, n] = size(A);

% Loop over the entire matrix.
i  = 1;
j  = 1;
jb = [];
while (i <= m) && (j <= n)
   % Find index of nonzero element in the remainder of column j.
   tmp1 = A(i:m,j);
   k    = find(tmp1.x,1);
   if isempty(k)
      j = j + 1;
   else
      k = k+i-1; % absolute row index
      % Remember column index
      jb = [jb j];
      % Swap i-th and k-th rows.
      A([i k],j:n) = A([k i],j:n);
      % Add the pivot row to all the other rows where j-th element is 1.
      tmp1 = A(:,j); ks = find(tmp1.x); ks(ks==i) = [];
      A(ks,j:n) = A(ks,j:n) + repmat( A(i, j:n), length(ks), 1);
      i = i + 1;
      j = j + 1;
   end
end
