export genus, rank, det, determinant, dimension, dim, prime, symbol, representative, is_even, signature, oddity, excess, level, genera, scale, norm, discriminant_form, mass, direct_sum,quadratic_space,hasse_invariant

@doc Markdown.doc"""
    ZpGenus

Local genus symbol over a p-adic ring.

The genus symbol of a component `p^m A` for odd prime `= p` is of the
form `(m,n,d)`, where

- `m` = valuation of the component
- `n` = rank of A
- `d = det(A) \in \{1,u\}` for a normalized quadratic non-residue `u`.

The genus symbol of a component `2^m A` is of the form `(m, n, s, d, o)`,
where

- `m` = valuation of the component
- `n` = rank of `A`
- `d` = det(A) in `\{1,3,5,7\}`
- `s` = 0 (or 1) if even (or odd)
- `o` = oddity of `A` (= 0 if s = 0) in `Z/8Z`
      = the trace of the diagonalization of `A`

The genus symbol is a list of such symbols (ordered by `m`) for each
of the Jordan blocks `A_1,...,A_t`.

Reference: [Co1999]_ Conway && Sloane 3rd edition, Chapter 15, Section 7.


    INPUT:

    - ``prime`` -- a prime number
    - ``symbol`` -- the list of invariants for Jordan blocks `A_t,...,A_t` given
      as a list of lists of integers

"""
mutable struct ZpGenus
  _prime::fmpz
  _symbol::Array{Array{Int,1},1}

  function ZpGenus(prime, symbol)
    @assert isprime(prime)
    if prime == 2
      @assert all(length(g)==5 for g in symbol)
    else
      @assert all(length(g)==3 for g in symbol)
    end
    g = new()
    g._prime = prime
    g._symbol = symbol
    return g
  end
end

@doc Markdown.doc"""
    ZGenus

A collection of local genus symbols (at primes)
and a signature pair. Together they represent the genus of a
non-degenerate Zlattice.
"""
mutable struct ZGenus
  _signature_pair::Array{Int}
  _symbols::Array{ZpGenus} # assumed to be sorted by their primes
  _representative::ZLat

  function ZGenus(signature_pair, symbols)
    G = new()
    G._signature_pair = signature_pair
    G._symbols = symbols
    return G
  end

  function ZGenus(signature_pair, symbols, representative::ZLat)
    G = new()
    G._signature_pair = signature_pair
    G._symbols = symbols
    G._representative = representative
    return G
  end
end

@doc Markdown.doc"""
    _is_even(A::MatElem) -> (Bool, Int)

Determines if the integral matrix `A` has even diagonal
(i.e. represents only even numbers).  If not, then it returns the
index of an odd diagonal entry.  If it is even, then return the
index -1.
"""
function _is_even(A::MatElem)
  for i in 1:nrows(A)
    if isodd(ZZ(A[i,i]))
      return false, i
    end
  end
  return true, -1
end

@doc Markdown.doc"""
    _split_odd(A::MatElem) -> (fmpz, fmpz_mat)

Given a non-degenerate Gram matrix `A (\mod 8)`, return a splitting
``[u] + B`` such that u is odd && `B` is not even.
Return `(u,B)`.
"""
function _split_odd(A::MatElem)
  n0 = nrows(A)
  if n0 == 1
    return A[1, 1], zero_matrix(ZZ, 0, ncols(A))
  end
  even, i = _is_even(A)
  R = base_ring(A)
  C = zero_matrix(R, n0 - 1, n0)
  u = A[i,i]
  for j in 1:n0-1
    if j < i
      C[j,j] = 1
      C[j,i] = -A[j,i] * u
    else
      C[j,j+1] = 1
      C[j,i] = -A[j+1,i] * u
    end
  end
  B = C*A*C'
  even, j = _is_even(B)
  if even
    I = parent(A)(1)
    # TODO: we could manually (re)construct the kernel here...
    if i == 1
      I[2,1] = 1 - A[2,1]*u
      i = 2
    else
      I[1,i] = 1 - A[1,i]*u
      i = 1
    end
    A = I * A * I'
    u = A[i,i]
    C = zero_matrix(R, n0-1, n0)
    for j in 1:n0-1
      if j < i
        C[j,j] = 1
        C[j,i] = -A[j,i] * u
      else
        C[j,j+1] = 1
        C[j,i] = -A[j+1,i] * u
    end
  end
  B = C * A * C'
  end
  even, j = _is_even(B)
  @assert !even
  return u, B
end

@doc Markdown.doc"""
    _trace_diag_mod_8(A::MatElem) -> fmpz

Return the trace of the diagonalised form of `A` of an integral
symmetric matrix which is diagonalizable `\mod 8`.  (Note that since
the Jordan decomposition into blocks of size `<=` 2 is not unique
here, this is not the same as saying that `A` is always diagonal in
any `2`-adic Jordan decomposition!)

INPUT:

- ``A`` -- symmetric matrix with coefficients in `\ZZ` which is odd in
  `\ZZ/2\ZZ` && has determinant not divisible by `8`.
"""
function _trace_diag_mod_8(A::MatElem)
  R = ResidueRing(ZZ, 8)
  A8 = change_base_ring(R, A)
  tr = R(0)
  while nrows(A8) > 0
    u, A8 = _split_odd(A8)
    tr += u
  end
  tr = lift(tr)
  return mod(tr, 8)
end

@doc Markdown.doc"""
    _p_adic_symbol(A::MatElem) -> Array{Array{Int64,1},1}

Given a symmetric matrix `A` && prime `p`, return the Conway Sloane
genus symbol at `p` represented as a list of lists.

The genus symbol of a component `p^m f` is of the form ``(m,n, d)``,
where

- `m` = valuation of the component
- `n` = dimension of `f`
- `d = det(f)` in `{1,-1}`
"""
function _p_adic_symbol(A::MatElem, p, val)
    if p == 2
        return _two_adic_symbol(A, val)
    end
    if nrows(A)>0
      m0 = minimum(valuation(c, p) for c in A if c!=0)
    else
      m0 = 0
    end
    q = p^m0
    n = nrows(A)
    A = divexact(A, q)
    Fp = GF(p)
    A_p = change_base_ring(Fp, A)
    bp, B_p = left_kernel(A_p)
    rref!(B_p)
    B_p = B_p[1:bp, 1:end]
    if nrows(B_p) == 0
      e0 = _kronecker_symbol(lift(det(A_p)),p)
      n0 = nrows(A)
      return [ [m0, n0, e0] ]
    else
      C_p = _basis_complement(B_p)
      e0 = _kronecker_symbol(lift(det(C_p * A_p * C_p')), p)
      n0 = nrows(C_p)
      sym = [ [0, n0, e0] ]
    end
    r = nrows(B_p)
    B = map_entries(lift, B_p)
    C = map_entries(lift, C_p)
    # Construct the blocks for the Jordan decomposition [F,X;X,A_new]
    F = change_base_ring(QQ, C * A * C')
    U = F^-1
    d = denominator(U)
    R = ResidueRing(ZZ, p^(val + 3))
    u = R(d)^-1

    U = change_base_ring(ZZ, U * d *lift(u))

    X = C * A
    A = B * (A - X'*U*X) * B'
    return [vcat([s[1]+m0] , s[2:end]) for s in vcat(sym,_p_adic_symbol(A, p, val)) ]
end


@doc Markdown.doc"""
    _two_adic_symbol(A::MatElem) -> Array{Array{Int64,1},1}

Given a symmetric matrix `A` over `Z`, return the Conway Sloane
genus symbol at `2` represented as a list of lists.

The genus symbol of a component `2^m f` is of the form ``(m,n,s,d[,o])``,
where

- `m` = valuation of the component
- `n` = dimension of `f`
- `d` = det(f) in {1,3,5,7}`
- `s` = 0` (or `1`) if even (or odd)
- `o` = oddity of `f` (`= 0` if `s = 0`) in `Z/8Z`

INPUT:

- ``A`` -- symmetric matrix with integer coefficients, non-degenerate
- ``val`` -- non-negative integer; valuation of maximal `2`-elementary divisor

OUTPUT:

a list of lists of integers (representing a Conway-Sloane `2`-adic symbol)
"""
function _two_adic_symbol(A::MatElem, val)
  n = nrows(A)
  # deal with the empty matrix
  if n == 0
    return [[0, 0, 1, 0, 0]]
  end
  m0 = minimum([ valuation(c,2) for c in A if c!=0])
  q = ZZ(2)^m0
  A = divexact(A, q)
  A_2 = change_base_ring(GF(2), A)
  k2, B_2 = left_kernel(A_2)
  rref!(B_2)
  B_2 = B_2[1:k2,1:end]
  R_8 = ResidueRing(ZZ, 8)
  ## Deal with the matrix being non-degenerate mod 2.
  if k2 == 0
    n0 = nrows(A)
    d0 = mod(det(A),8)
    @assert d0 != 0    ## SANITY CHECK: The mod 8 determinant shouldn't be zero.
    even, i = _is_even(A)    ## Determine whether the matrix is even || odd.
    if even
      return [[m0, n0, d0, 0, 0]]
    else
      tr8 = _trace_diag_mod_8(A)  ## Here we already know that A_8 is odd && diagonalizable mod 8.
      return [[m0, n0, d0, 1, tr8]]
    end
  ## Deal with the matrix being degenerate mod 2.
  else
    C_2 = _basis_complement(B_2)
    n0 = nrows(C_2)
    C = map_entries(lift, C_2)
    A_new = C * A * C'
    # compute oddity modulo 8:
    d0 = mod(det(A_new), 8)
    @assert d0 != 0
    even, i = _is_even(A_new)
    if even
      sym = [[0, n0, d0, 0, 0]]
    else
      tr8 = _trace_diag_mod_8(A_new)
      sym = [[0, n0, d0, 1, tr8]]
    end
  end
  r = nrows(B_2)
  B = map_entries(lift, B_2)
  C = map_entries(lift, C_2)
  F = change_base_ring(QQ, C * A * C')
  U = F^-1
  d = denominator(U)
  R = ResidueRing(ZZ,2^(val + 3))
  u = lift(R(d)^-1)
  U = change_base_ring(ZZ,U * d * u)
  X = C * A

  A = B * (A - X'*U*X) * B'
  return [ vcat([s[1]+m0], s[2:end]) for s in vcat(sym, _two_adic_symbol(A, val)) ]
end


@doc Markdown.doc"""
    _basis_complement(B::MatElem) -> MatElem

Given an echelonized basis matrix `B` (over a field), calculate a
matrix whose rows form a basis complement of the rows of `B`.

julia> B = matrix(ZZ, 1, 2, [1,0])
[1  0]
julia> Hecke.basis_complement(B)
[0  1]
"""
function _basis_complement(B::MatElem)
    F = base_ring(B)
    m = nrows(B)
    n = ncols(B)
    C = zero_matrix(F, n - m, n)
    k = 1
    l = 1
    for i in 1:m
      for j in k:n
        if B[i,j] == 0
          C[l,j] = 1
          l += 1
        else
          k = j+1
          break
        end
      end
    end
    for j in k:n
        C[l + j - k, j] = 1
    end
    return C
end

@doc Markdown.doc"""
    Return a list of all global genera with the given conditions.

Here a genus is called global if it is non-empty.

INPUT:

- ``sig_pair`` -- a pair of non-negative integers giving the signature
- ``determinant`` -- an integer; the sign is ignored
- ``max_scale`` -- (default: ``Nothing``) an integer; the maximum scale of a
      jordan block
- ``even`` -- boolean (default: ``false``)

OUTPUT:

A list of all (non-empty) global genera with the given conditions.
"""
function genera(sig_pair, determinant; max_scale=Nothing, even=false)
  determinant = ZZ(determinant)
  if !all(s >= 0 for s in sig_pair)
    raise(error("the signature vector must be a pair of non negative integers."))
  end
  if max_scale == Nothing
    max_scale = determinant
  else
    max_scale = ZZ(max_scale)
  end
  rank = sig_pair[1] + sig_pair[2]
  genera = ZGenus[]
  local_symbols = Vector{ZpGenus}[]
  # every global genus has a 2-adic symbol
  if mod(determinant, 2) == 1
    push!(local_symbols, _local_genera(2, rank, 0, 0, even))
  end
  # collect the p-adic symbols
  for p in prime_divisors(determinant)
    det_val = valuation(determinant, p)
    mscale_p = valuation(max_scale, p)
    local_symbol_p = _local_genera(p, rank, det_val, mscale_p, even)
    push!(local_symbols,local_symbol_p)
  end
  # take the cartesian product of the collection of all possible
  # local genus symbols one for each prime
  # && check which combinations produce a global genus
  # TODO:
  # we are overcounting. Find a more
  # clever way to directly match the symbols for different primes.
  for g in cartesian_product_iterator(local_symbols)
    # create a Genus from a list of local symbols
    G = ZGenus(sig_pair, g)
    # discard the empty genera
    if _is_global_genus(G)
      push!(genera, G)
    end
  end
  # render the output deterministic for testing
  sort!(genera,by=x -> [s._symbol for s in x._symbols])
    return genera
end

@doc Markdown.doc"""
    _local_genera(p, rank, det_val, max_scale, even)

Return all `p`-adic genera with the given conditions.

This is a helper function for `genera`.
No input checks are done.

INPUT:
- ``p`` -- a prime number
- ``rank`` -- the rank of this genus
- ``det_val`` -- valuation of the determinant at p
- ``max_scale`` -- an integer the maximal scale of a jordan block
- ``even`` -- ``bool``; is ignored if `p` is not `2`
    """
function _local_genera(p, rank, det_val, max_scale, even)
  scales_rks = [] # contains possibilities for scales && ranks
  rank = Int64(rank)
  for rkseq in _integer_lists(rank, max_scale+1)
    # rank sequences
    # sum(rkseq) = rank
    # length(rkseq) = max_scale + 1
    # now assure that we get the right determinant
    d = 0
    pgensymbol = Vector{Int}[]
    for i in 0:max_scale
      d += i * rkseq[i+1]
      # blocks of rank 0 are omitted
      if rkseq[i+1] != 0
        push!(pgensymbol,[i, rkseq[i+1], 0])
      end
    end
    if d == det_val
      push!(scales_rks,pgensymbol)
    end
  end
  # add possible determinant square classes
  symbols = Vector{ZpGenus}()
  if p != 2
    for g in scales_rks
      n = length(g)
      for v in cartesian_product_iterator([[-1, 1] for i in 1:n], inplace=false)
        g1 = deepcopy(g)
        for k in 1:n
          g1[k][3] = v[k]
        end
        g1 = ZpGenus(p, g1)
        push!(symbols, g1)
      end
    end
  end
  # for p == 2 we have to include determinant, even/odd, oddity
  # further restrictions apply && are deferred to _blocks
  # (brute force sieving is too slow)
  # TODO: If this is too slow, enumerate only the canonical symbols.
  # as a drawback one has to reconstruct the symbol from the canonical symbol
  # this is more work for the programmer
  if p == 2
    for g in scales_rks
      poss_blocks = Vector{Vector{Vector{Int}}}()
      for b in g
        append!(b,[0, 0])
        push!(poss_blocks,_blocks(b, (even && b[1] == 0)))
      end
      for g1 in cartesian_product_iterator(poss_blocks,inplace=false)
        if _is_2_adic_genus(g1)
          g1 = ZpGenus(p, g1)
          # some of our symbols have the same canonical symbol
          # thus they are equivalent - we want only one in
          # each equivalence class
          if !(g1 in symbols)
            push!(symbols, g1)
          end
        end
      end
    end
  end
  return symbols
end

@doc Markdown.doc"""
    _blocks(b::Array{Int}, even_only=false) -> Vector{Vector{Int}}

Return all viable `2`-adic jordan blocks with rank && scale given by ``b``

This is a helper function for `_local_genera`.
It is based on the existence conditions for a modular `2`-adic genus symbol.

INPUT:

- ``b`` -- a list of `5` non-negative integers the first two are kept
and all possibilities for the remaining `3` are enumerated

- ``even_only`` -- bool (default: ``true``) if set, the blocks are even
"""
function _blocks(b::Array{Int}, even_only=false)
  blocks = Array{Array{Int,1},1}()
  rk = b[2]
  # recall: 2-genus_symbol is [scale, rank, det, even/odd, oddity]
  if rk == 0
    @assert b[3] == 1
    @assert b[4] == 0
    @assert b[5] == 0
    push!(blocks, copy(b))
  elseif rk == 1 && !even_only
    for det in [1, 3, 5, 7]
      b1 = copy(b)
      b1[3] = det
      b1[4] = 1
      b1[5] = det
      push!(blocks, b1)
    end
  elseif rk == 2
    b1 = copy(b)
    # even case
    b1[4] = 0
    b1[5] = 0
    b1[3] = 3
    push!(blocks, b1)
    b1 = copy(b1)
    b1[3] = 7
    push!(blocks, b1)
    # odd case
    if !even_only
      # format (det, oddity)
      for s in [(1,2), (5,6), (1,6), (5,2), (7,0), (3,4)]
        b1 = copy(b)
        b1[3] = s[1]
        b1[4] = 1
        b1[5] = s[2]
        push!(blocks, b1)
      end
    end
  elseif rk % 2 == 0
    # the even case has even rank
    b1 = copy(b)
    b1[4] = 0
    b1[5] = 0
    d = mod((-1)^(rk//2), 8)
    for det in [d, mod(d * (-3) , 8)]
      b1 = copy(b1)
      b1[3] = det
        push!(blocks, b1)
    end
    # odd case
    if !even_only
      for s in [(1,2), (5,6), (1,6), (5,2), (7,0), (3,4)]
        b1 = copy(b)
        b1[3] = s[1]*mod((-1)^(rk//2 -1) , 8)
        b1[4] = 1
        b1[5] = s[2]
        push!(blocks, b1)
      end
      for s in [(1,4), (5,0)]
        b1 = copy(b)
        b1[3] = s[1]*mod((-1)^(rk//2 - 2) , 8)
        b1[4] = 1
        b1[5] = s[2]
        push!(blocks, b1)
      end
    end
  elseif rk % 2 == 1 && !even_only
    # odd case
    for t in [1, 3, 5, 7]
      d = mod((-1)^div(rk, 2) * t , 8)
      for det in [d, mod(-3*d, 8)]
        b1 = copy(b)
        b1[3] = det
        b1[4] = 1
        b1[5] = t
        push!(blocks, b1)
      end
    end
  end
  # convert ints to integers
  return blocks
end

function genus(A::MatElem)
  return genus(Zlattice(gram=A))
end

function genus(L::ZLat)
  A = gram_matrix(L)
  denom = denominator(A)
  A = change_base_ring(ZZ, denom^2 * A)
  symbols = ZpGenus[]
  el = lcm(diagonal(hnf(A)))
  primes = prime_divisors(el)
  if !(2 in primes)
    prepend!(primes, 2)
  end
  for p in primes
    val = valuation(el, p)
    if p == 2
      val += 3
    end
    push!(symbols, genus(A, p, val, offset=2*valuation(denom,p)))
  end
  DA = diagonal(quadratic_space(QQ, A))
  neg = Int(count(x<0 for x in DA))
  pos = Int(count(x>0 for x in DA))
  if neg+pos != ncols(A)
    raise(error("QuadraticForm is degenerate"))
  end
  return ZGenus([pos, neg], symbols, L)
end

@doc Markdown.doc"""
    _is_global_genus(G::ZGenus) -> Bool

Return if `S` is the symbol of of a global quadratic form || lattice.
"""
function _is_global_genus(G::ZGenus)
  D = determinant(G)
  r, s = signature_pair(G)
  oddi = r - s
  for loc in G._symbols
    p = loc._prime
    sym = loc._symbol
    v = sum([ss[1] * ss[2] for ss in sym])
    a = divexact(D, p^v)
    b = prod([ss[3] for ss in sym])
    if p == 2
      if !_is_2_adic_genus(sym)
        return false
      end
      if _kronecker_symbol(a*b, p) != 1
        return false
      end
      oddi -= oddity(loc)
    else
      if _kronecker_symbol(a, p) != b
        return false
      end
      oddi += excess(loc)
    end
  end
  if oddi != 0
    return false
  end
  return true
end


@doc Markdown.doc"""
    _is_2_adic_genus(symbol::Array{Array{Int,1},1})

Given a `2`-adic local symbol (as the underlying list of quintuples)
check whether it is the `2`-adic symbol of a `2`-adic form.

INPUT:

- ``genus_symbol_quintuple_list`` -- a quintuple of integers (with certain
  restrictions).
  """
function _is_2_adic_genus(symbol::Array{Array{Int,1},1})
  for s in symbol
    ## Check that we have a quintuple (i.e. that p=2 && not p >2)
    if size(s)[1] != 5
      raise(error("The genus symbols are not quintuples, so it's not a genus symbol for the prime p=2."))
    end
    ## Check the Conway-Sloane conditions
    if s[2] == 1
      if s[4] == 0 || s[3] != s[5]
        return false
      end
    end
    if s[2] == 2 && s[4] == 1
      if mod(s[3], 8) in (1, 7)
        if !(s[5] in (0, 2, 6))
          return false
        end
      end
      if mod(s[3], 8) in (3, 5)
        if !(s[5] in (2, 4, 6))
          return false
        end
      end
    end
    if mod(s[2] - s[5], 2) == 1
      return false
    end
    if s[4] == 0 && s[5] != 0
      return false
    end
  end
  return true
end


function prime(S::ZpGenus)
  return S._prime
end

function symbol(S::ZpGenus)
  return copy(S._symbol)
end

function genus(L::ZLat, p)
  return genus(gram_matrix(L), p)
end

function genus(A::fmpz_mat, p, val; offset=0)
  @assert base_ring(A)==ZZ
  p = ZZ(p)
  symbol = _p_adic_symbol(A, p, val)
  for i in 1:size(symbol)[1]
    symbol[i][1] = symbol[i][1] - offset
  end
  return ZpGenus(p, symbol)
end

function genus(A::MatElem, p)
  offset = 0
  if base_ring(A) == QQ
    d = denominator(A)
    val = valuation(d, p)
    A = change_base_ring(ZZ, A*divexact(d^2, p^val))
    offset = valuation(d, p)
  end
  val = valuation(det(A), p)
  if p == 2
    val += 3
  end
  return genus(A, p, val, offset=offset)
end

function Base.show(io::IO, G::ZpGenus)
  p = G._prime
  CS_string = ""
  if p == 2
    for sym in G._symbol
      s, r, d, e, o = sym
      d = _kronecker_symbol(d, 2)
      if s>=0
        CS_string *= " $(p^s)^$(d * r)"
      else
        CS_string *="(1/$(p^-s))^$(d * r)"
      end
      if e == 1
        CS_string *= "_$o"
      end
    end
  else
    for sym in G._symbol
      s,r ,d = sym
      CS_string *= " $(p^s)^$(d * r)"
    end
  end
  rep = "Genus symbol at $p:  $CS_string"
  print(io, rstrip(rep))
end

function Base.:(==)(G1::ZpGenus, G2::ZpGenus)
  # This follows p.381 Chapter 15.7 Theorem 10 in Conway Sloane's book
  if G1._prime != G2._prime
    raise(error("Symbols must be over the same prime to be comparable"))
  end
  if G1._prime != 2
    return G1._symbol == G2._symbol
  end
  sym1 = symbol(G1)
  sym2 = symbol(G2)
  n = length(sym1)
  @assert all(g[2]!=0 for g in sym1)
  @assert all(g[2]!=0 for g in sym2)
  # scales && ranks
  s1 = [g[1:2] for g in sym1]
  s2 = [g[1:2] for g in sym2]
  if s1!=s2
    return false
  end
  # parity
  s1 = [g[4] for g in sym1]
  s2 = [g[4] for g in sym2]
  if s1 != s2
    return false
  end
  push!(sym1,[sym1[end][1]+1,0,1,0,0])
  push!(sym2,[sym1[end][1]+1,0,1,0,0])
  prepend!(sym1,[[-1,0,1,0,0]])
  prepend!(sym1,[[-2,0,1,0,0]])
  prepend!(sym2,[[-1,0,1,0,0]])
  prepend!(sym2,[[-2,0,1,0,0]])
  n = length(sym1)
  # oddity && sign walking conditions
  for m in 1:n
    # "for each integer m for which f_{2^m} has type II, we have..."
    if sym1[m][4] == 1
      continue # skip if type I
    end
    # sum_{q<2^m}(t_q-t'_q)
    l = sum(fmpz[sym1[i][5]-sym2[i][5] for i in 1:m-1])
    # 4 (min(a,m)+min(b,m)+...)
    # where 2^a, 2^b are the values of q for which e_q!=e'_q
    det_differs = [i for i in 1:n if _kronecker_symbol(sym1[i][3],2)!=_kronecker_symbol(sym2[i][3],2)]
    r = 4*sum(fmpz[min(sym1[m][1], sym1[i][1]) for i in det_differs])
    if 0 != mod(l-r, 8)
      return false
    end
  end
  return true
end

#=
    function automorphous_numbers(self)
        r"""
        Return generators of the automorphous square classes at this prime.

        A `p`-adic square class `r` is called automorphous if it is
        the spinor norm of a proper `p`-adic integral automorphism of this form.
        These classes form a group. See [Co1999]_ Chapter 15, 9.6 for details.

        OUTPUT:

        - a list of integers representing the square classes of generators of
          the automorphous numbers

        EXAMPLES:

        The following examples are given in
        [Co1999]_ 3rd edition, Chapter 15, 9.6 pp. 392::

            sage: A = matrix.diagonal([3, 16])
            sage: G = Genus(A)
            sage: sym2 = G.local_symbols()[0]
            sage: sym2
            Genus symbol at 2:    [1^-1]_3:[16^1]_1
            sage: sym2.automorphous_numbers()
            [3, 5]

            sage: A = matrix(ZZ,3,[2,1,0, 1,2,0, 0,0,18])
            sage: G = Genus(A)
            sage: sym = G.local_symbols()
            sage: sym[0]
            Genus symbol at 2:    1^-2 [2^1]_1
            sage: sym[0].automorphous_numbers()
            [1, 3, 5, 7]
            sage: sym[1]
            Genus symbol at 3:     1^-1 3^-1 9^-1
            sage: sym[1].automorphous_numbers()
            [1, 3]

        Note that the generating set given is not minimal.
        The first supplementation rule is used here::

            sage: A = matrix.diagonal([2, 2, 4])
            sage: G = Genus(A)
            sage: sym = G.local_symbols()
            sage: sym[0]
            Genus symbol at 2:    [2^2 4^1]_3
            sage: sym[0].automorphous_numbers()
            [1, 2, 3, 5, 7]

        but not there::

            sage: A = matrix.diagonal([2, 2, 32])
            sage: G = Genus(A)
            sage: sym = G.local_symbols()
            sage: sym[0]
            Genus symbol at 2:    [2^2]_2:[32^1]_1
            sage: sym[0].automorphous_numbers()
            [1, 2, 5]

        Here the second supplementation rule is used::

            sage: A = matrix.diagonal([2, 2, 64])
            sage: G = Genus(A)
            sage: sym = G.local_symbols()
            sage: sym[0]
            Genus symbol at 2:    [2^2]_2:[64^1]_1
            sage: sym[0].automorphous_numbers()
            [1, 2, 5]
        """
        from .normal_form import collect_small_blocks
        automorphs = []
        sym = self.symbol_tuple_list()
        G = self.gram_matrix().change_ring(ZZ)
        p = self.prime()
        if p != 2:
            up = ZZ(_min_nonsquare(p))
            I = G.diagonal()
            for r in I:
                # We need to consider all pairs in I
                # since at most 2 elements are part of a pair
                # we need need at most 2 of each type
                if I.count(r) > 2:
                    I.remove(r)
            # products of all pairs
            for r1 in I:
                for r2 in I:
                    automorphs.append(r1*r2)
            # supplement (i)
            for block in sym:
                if block[1] >= 2:
                    automorphs.append(up)
                    break
            # normalize the square classes && remove duplicates
            automorphs1 = set()
            for s in automorphs:
                u = 1
                if s.prime_to_m_part(p).kronecker(p) == -1:
                    u = up
                v = (s.valuation(p) % 2)
                sq = u * p^v
                automorphs1.add(sq)
            return list(automorphs1)

        # p = 2
        I = []
        II = []
        for block in collect_small_blocks(G)
            if block.ncols() == 1:
                u = block[0,0]
                if I.count(u) < 2:
                    I.append(block[0,0])
            else # rank2
                q = block[0,1]
                II += [2*q, 3*2*q, 5*2*q, 7*2*q]

        L = I + II
        # We need to consider all pairs in L
        # since at most 2 elements are part of a pair
        # we need need at most 2 of each type
        for r in L:     # remove triplicates
            if L.count(r) > 2:
                L.remove(r)
        n = length(L)
        for i in range(n)
            for j in range(i)
                r = L[i] * L[j]
                automorphs.append(r)

        # supplement (i)
        for k in range(length(sym))
            s = sym[k:k+3]
            if sum([b[1] for b in s if b[0] - s[0][0] < 4]) >= 3:
                automorphs += [ZZ(1), ZZ(3), ZZ(5), ZZ(7)]
            break

        # supplement (ii)
        I.sort(key=lambda x: x.valuation(2))
        n = length(I)
        for i in range(n)
            for j in range(i)
                r = I[i] / I[j]
                v, u = r.val_unit(ZZ(2))
                u = u % 8
                assert v >= 0
                if v==0 && u==1:
                    automorphs.append(ZZ(2))
                if v==0 && u==5:
                    automorphs.append(ZZ(6))
                if v in [0, 2, 4]:  # this overlaps with the first two cases!
                    automorphs.append(ZZ(5))
                if v in [1, 3] && u in [1, 5]:
                    automorphs.append(ZZ(3))
                if v in [1, 3] && u in [3, 7]:
                    automorphs.append(ZZ(7))

        # normalize the square classes && remove duplicates
        automorphs1 = set()
        for s in automorphs:
            v, u = s.val_unit(ZZ(2))
            v = v % 2
            u = u % 8
            sq = u * 2^v
            automorphs1.add(sq)
        return list(automorphs1)

=#
@doc Markdown.doc"""
    representative(S::ZpGenus) -> MatElem

Return a gram matrix of some representative of this local genus.
    """
function representative(S::ZpGenus)
  G = fmpq_mat[]
  p = prime(S)
  for block in S._symbol
    push!(G, _gram_from_jordan_block(p, block))
  end
  G = diagonal_matrix(G)
  return change_base_ring(QQ, G)
end

@doc Markdown.doc"""
    is_even(S::ZpGenus) -> Bool

Return if the underlying `p`-adic lattice is even.

If `p` is odd, every lattice is even.
"""
function is_even(S::ZpGenus)
  if prime(S) != 2 || rank(S) == 0
    return true
  end
  sym = S._symbol[1]
  return sym[1] > 0 || sym[3] == 0
end

@doc Markdown.doc"""
    symbol(S::ZpGenus, scale) -> Array{Array{Int64,1},1}

Return a copy of the underlying lists of integers
for the Jordan block of the given scale
"""
function symbol(S::ZpGenus, scale::Int)
  sym = S._symbol
  for s in sym
    if s[1] == scale
      return copy(s)
    end
  end
  if S._prime != 2
    return [scale,0,1]
  else
    return [scale, 0,1,0,0]
  end
end

@doc Markdown.doc"""
  hasse_invariant(S::ZpGenus) -> Int

Return the Hasse invariant of a representative.
If the representative is diagonal (a_1,...a_n)
Then the Hasse invariant is

$\prod_{i<j}(a_i,a_j)_p$.
"""
function hasse_invariant(S::ZpGenus)
  # Conway Sloane Chapter 15 5.3
  n = dimension(S)
  f0 = [squarefree_part(determinant(S))]
  append!(f0, [1 for i in 2:n])
  f0 = diagonal_matrix(f0)
  f0 = genus(f0, prime(S))
  if excess(S) == excess(f0)
    return 1
  else
    return -1
  end
end

@doc Markdown.doc"""
    quadratic_space(G::ZGenus)

Return the quadratic space defined by this genus.
"""
function quadratic_space(G::ZGenus)
  dim = dimension(G)
  det = determinant(G)
  prime_neg_hasse = [prime(s) for s in G._symbols if hasse_invariant(s)==-1]
  neg = G._signature_pair[2]
  qf =_quadratic_form_with_invariants(dim, det, prime_neg_hasse, neg)
  return quadratic_space(QQ, qf)
end

rational_representative(G::ZGenus) = quadratic_space(G::ZGenus)

#=
    function represents(self,other)
        r"""
        Return if self is represents other.

        WARNING:

        For p == 2 the statement of O Meara is wrong.
        """
        self, other = other, self
        if self.prime() != other.prime()
            raise ValueError("different primes")
        p = self.prime()
        s1 = self.symbol_tuple_list()
        s2 = other.symbol_tuple_list()
        level = max(s1[-1][0],s2[-1][0])
        #notation
        function delta(pgenus,i)
            # O'Meara pp.
            if pgenus.symbol(i+1)[3]==1:
                return ZZ(2)^(i+1)
            if pgenus.symbol(i+2)[3]==1:
                return ZZ(2)^(i+2)
            return ZZ(0)

        genus1 = self
        genus2 = other
        gen1 = []
        gen2 = []

        for i in range(level+3)
            g1 = [s for s in s1 if s[0]<=i]
            g2 = [s for s in s2 if s[0]<=i]
            gen1.append(Genus_Symbol_p_adic_ring(p,g1))
            gen2.append(Genus_Symbol_p_adic_ring(p,g2))
            if p!=2 && not gen1[i].space()<=gen2[i].space()
                return false

        if p != 2:
            return true

        # additional conditions for p==2
        for i in range(level+1)
            d = QQ(gen1[i].det()*gen2[i].det())
            # Lower Type following O'Meara Page 858
            # (7)
            if gen1[i].rank() > gen2[i].rank()
                return false
            # (8)
            if gen1[i].rank() == gen2[i].rank()
                if d.valuation(2)%2!=0:
                    return false
            # (9)
            if gen1[i].rank() == gen2[i].rank()
                l = delta(genus1,i)
                r = delta(genus2,i).gcd(2^(i+2))
                if not r.divides(l)
                    return false
                l = delta(genus2,i-1)
                r = delta(genus1,i-1).gcd(2^(i+1))
                if not r.divides(l)
                    return false
            v = d.valuation(2)
            cond = (gen1[i].rank() + 1 == gen2[i].rank()
                    && gen1[i].rank()>0
                   )
            # (10)
            if cond && (i+1-v) % 2 == 0:
                l = delta(genus2,i-1)
                r = delta(genus1,i-1).gcd(2^(i+1))
                if not r.divides(l)
                    return false
            # (11)
            if cond && (i-v) % 2 == 0:
                l = delta(genus1,i)
                r = delta(genus2,i).gcd(2^(i+2))
                if not r.divides(l)
                    return false

        gen2_round = []
        for i in range(level+3)
            g2 = [s for s in s2 if s[0]<i || s[0]==i && s[3]==1]
            gen2_round.append(Genus_Symbol_p_adic_ring(p,g2))

        gen1_square = []
        for i in range(level+1)
            g1 = [s for s in s1 if s[0]<=i || s[0]==i+1 && s[3]==0]
            gen1_square.append(Genus_Symbol_p_adic_ring(p,g1))

        FH = LocalGenusSymbol(matrix(QQ,2,[0,1,1,0]),p).space()
        for i in range(level+1)
            # I
            d = delta(genus2,i)
            L = gen2_round[i+2].space()-gen1_square[i].space()
            if not any(u*d<=L for u in [1,3,5,7])
                return false
            # II
            d = delta(genus1,i)
            L = gen2_round[i+2].space()-gen1_square[i].space()
            if not any(u*d<=L for u in [1,3,5,7])
                return false
            # III
            S1 = gen2_round[i+2].space()
            S2 = gen1_square[i].space()
            if  S1 - S2 == FH:
                if not 2*delta(genus1,i).valuation(2) <= delta(genus1,i).valuation(2) + delta(genus2,i).valuation(2)
                    return false
            # IV
            ti1 = LocalGenusSymbol(matrix([2^i]),p).space()
            ti2 = LocalGenusSymbol(matrix([5*2^i]),p).space()
            S = (ti1 + gen2_round[i+1].space())-gen1[i].space()
            if not (ti1<=S || ti2<=S)
                return false
            # V
            # there is a typo in O'Meara
            # the reason is that
            # (ti1 + gen2_round[i+1])-gen1_square[i]
            # can have negative dimension
            # even if l = L .... && surely
            # L is represented by itsself
            S = (ti1 + gen2[i+1].space())-gen1_square[i].space()
            if not (ti1<=S || ti2<=S)
                return false
        return true
=#
@doc Markdown.doc"""
    determinant(S::ZpGenus) -> fmpz

Return an integer representing the determinant of this genus.
    """
function determinant(S::ZpGenus)
  p = S._prime
  e = prod(s[3] for s in S._symbol)
  if p == 2
    e = e % 8
  elseif e==-1
    e = _min_nonsquare(p)
  end
  return e*prod([ p^(s[1]*s[2]) for s in S._symbol ])
end

function det(S::ZpGenus)
 return determinant(S)
end
#=
=#
@doc Markdown.doc"""
    dimension(S::ZpGenus) -> fmpz

Return the dimension of this genus.
"""
function dimension(S::ZpGenus)
  return sum(s[2] for s in S._symbol)
end

function dim(S::ZpGenus)
  return dimension(S)
end

function rank(S::ZpGenus)
  return dimension(S)
end


@doc Markdown.doc"""
    direct_sum(S1::ZpGenus, S2::ZpGenus)

Return the local genus of the direct sum of two representatives.
"""
function direct_sum(S1::ZpGenus, S2::ZpGenus)
  if prime(S1) != prime(S2)
    throw(ValueError("the local genus symbols must be over the same prime"))
  end
  sym1 = S1._symbol
  sym2 = S2._symbol
  m = max(sym1[end][1], sym2[end][1])
  sym1 = Dict([[s[1], s] for s in sym1])
  sym2 = Dict([[s[1], s] for s in sym2])
  symbol = []
  for k in 0:m
    if prime(S1) == 2
      b = [k, 0, 1, 0, 0]
    else
      b = [k, 0, 1]
    end
    for sym in [sym1, sym2]
      try
        s = sym[k]
        b[2] += s[2]
        b[3] *= s[3]
        if prime(S1) == 2
          b[3] = mod(b[3], 8)
          if s[4] == 1
            b[4] = s[4]
          end
          b[5] = mod(b[5] + s[5], 8)
        end
      catch KeyError
      end
    end
    if b[2] != 0
      push!(symbol, b)
    end
  end
  if rank(S1) == rank(S2) == 0
    symbol = S1._symbol
  end
  return ZpGenus(prime(S1), symbol)
end

@doc Markdown.doc"""
    excess(S::ZpGenus) -> Nemo.fmpz_mod

Return the p-excess of the quadratic form whose Hessian
matrix is the symmetric matrix A.

When p = 2 the p-excess is
called the oddity.
The p-excess is allways even && is divisible by 4 if
p is congruent 1 mod 4.

REFERENCE:
Conway && Sloane Lattices && Codes, 3rd edition, pp 370-371.
"""
function excess(S::ZpGenus)
  R = ResidueRing(ZZ, 8)
  p = S._prime
  if S._prime == 2
    return dimension(S) - oddity(S)
  end
  k = 0
  for s in S._symbol
    if isodd(s[1]) && s[3] == -1
      k += 1
    end
  end
  return R(sum(s[2]*(p^s[1]-1) for s in S._symbol) + 4*k)
end

@doc Markdown.doc"""
    signature(S::ZpGenus) -> Nemo.fmpz_mod

Return the p-signature of this p-adic form.
"""
function signature(S::ZpGenus)
  R = ResidueRing(ZZ, 8)
  if S._prime == 2
    return oddity(S)
  else
    return R(dimension(S)) - excess(S)
  end
end

@doc Markdown.doc"""
    oddity(S::ZpGenus) -> Nemo.fmpz_mod

Return the oddity of this even form.

The oddity is also called the 2-signature
"""
function oddity(S::ZpGenus)
  R = ResidueRing(ZZ, 8)
  p = S._prime
  if p != 2
    raise(error("the oddity is only defined for p=2"))
  end
  k = 0
  for s in S._symbol
    if mod(s[1], 2) == 1 && s[3] in (3, 5)
      k += 1
    end
  end
  return R(sum([s[5] for s in S._symbol]) + 4*k)
end

@doc Markdown.doc"""
    scale(S::ZpGenus) -> fmpz

Return the scale of this local genus.

Let `L` be a lattice with bilinear form `b`.
The scale of `(L,b)` is defined as the ideal
`b(L,L)`.
"""
function scale(S::ZpGenus)
  if rank(S) == 0
    return ZZ(0)
  end
  return S._prime^S._symbol[1][1]
end

@doc Markdown.doc"""
    norm(S::ZpGenus) -> fmpz

Return the norm of this local genus.

Let `L` be a lattice with bilinear form `b`.
The norm of `(L,b)` is defined as the ideal
generated by `\{b(x,x) | x \in L\}`.
"""
function norm(S::ZpGenus)
  if rank(S) == 0
    return ZZ(0)
  end
  p = prime(S)
  if p == 2
    fq = S._symbol[1]
    return S._prime^(fq[1] + 1 - fq[4])
  else
    return scale(S)
  end
end
@doc Markdown.doc"""
    level(S::ZpGenus) -> fmpz

Return the maximal scale of a jordan component.
"""
function level(S::ZpGenus)
  if rank(S) == 0
    return ZZ(1)
  end
  return prime(S)^S._symbol[end][1]
end

function Base.show(io::IO, G::ZGenus)
  rep = "ZGenus\nSignature: $(G._signature_pair)"
  for s in G._symbols
    rep *= "\n$s"
  end
  print(io, rep)
end
#=
    function _latex_(self)
        r"""
        The Latex representation of this lattice.

        EXAMPLES::

            sage: D4 = QuadraticForm(Matrix(ZZ, 4, 4, [2,0,0,-1, 0,2,0,-1, 0,0,2,-1, -1,-1,-1,2]))
            sage: G = D4.global_genus_symbol()
            sage: latex(G)
            \mbox{Genus of}\\ \left(\begin{array}{rrrr}
            2 & 0 & 0 & -1 \\
            0 & 2 & 0 & -1 \\
            0 & 0 & 2 & -1 \\
            -1 & -1 & -1 & 2
            \end{array}\right)\\ \mbox{Signature: } (4, 0)\\ \mbox{Genus symbol at } 2\mbox{: }1^{-2}  :2^{-2}
        """
        rep = r"\mbox{Genus"
        if self.dimension() <= 20:
            rep += r" of}\\ %s" %self._representative._latex_()
        else
            rep +=r"}"
        rep += r"\\ \mbox{Signature: } %s"%(self._signature,)
        for s in self._symbols:
            rep += r"\\ " + s._latex_()
        return rep
=#

function Base.:(==)(G1::ZGenus, G2::ZGenus)
  t = length(G1._symbols)
  if t != length(G2._symbols)
    return false
  end
  for i in 1:t
    if G1._symbols[i] != G2._symbols[i]
      return false
    end
  end
  return true
end

@doc Markdown.doc"""is_even(G::ZGenus)
    Return if this genus is even.
"""
function is_even(G::ZGenus)
  if rank(G) == 0
    return true
  end
  return is_even(G._symbols[1])
end
#=
    function _proper_spinor_kernel(self)
        r"""
        Return the proper spinor kernel.

        OUTPUT:

        A pair ``(A, K)`` where

        .. MATH::

            A = \prod_{p \mid 2d} ZZ_p^\times / ZZ_p^{\times2},

        `d` is the determinant of this genus && `K` is a subgroup of `A`.

        EXAMPLES::

            sage: gram = matrix(ZZ, 4, [2,0,1,0, 0,2,1,0, 1,1,5,0, 0,0,0,16])
            sage: genus = Genus(gram)
            sage: genus._proper_spinor_kernel()
            (Group of SpinorOperators at primes (2,),
            Subgroup of Group of SpinorOperators at primes (2,) generated by (1, 1, f2))
            sage: gram = matrix(ZZ, 4, [3,0,1,-1, 0,3,-1,-1, 1,-1,6,0, -1,-1,0,6])
            sage: genus = Genus(gram)
            sage: genus._proper_spinor_kernel()
            (Group of SpinorOperators at primes (2,),
            Subgroup of Group of SpinorOperators at primes (2,) generated by (1, 1, f2))
        """
        from sage.quadratic_forms.genera.spinor_genus import SpinorOperators
        syms = self.local_symbols()
        primes = tuple([sym.prime() for sym in syms])
        A = SpinorOperators(primes)
        kernel_gens = []
        # -1 adic contribution
        sig = self.signature_pair_of_matrix()
        if sig[0] * sig[1] > 1:
            kernel_gens.append(A.delta(-1, prime=-1))
        for sym in syms:
            for r in sym.automorphous_numbers()
                kernel_gens.append(A.delta(r, prime=sym.prime()))
        K = A.subgroup(kernel_gens)
        return A, K

    function _improper_spinor_kernel(self)
        r"""
        Return the improper spinor kernel.

        OUTPUT:

        A pair ``(A, K)`` where

        .. MATH::

            A = \prod_{p \mid 2d} ZZ_p^\times / ZZ_p^{\times2},

        `d` is the determinant of this genus && `K` is a subgroup of `A`.

        EXAMPLES::

            sage: gram = matrix(ZZ, 4, [2,0,1,0, 0,2,1,0, 1,1,5,0, 0,0,0,16])
            sage: genus = Genus(gram)
            sage: genus._proper_spinor_kernel()
            (Group of SpinorOperators at primes (2,),
            Subgroup of Group of SpinorOperators at primes (2,) generated by (1, 1, f2))
            sage: gram = matrix(ZZ, 4, [3,0,1,-1, 0,3,-1,-1, 1,-1,6,0, -1,-1,0,6])
            sage: genus = Genus(gram)
            sage: genus._improper_spinor_kernel()
            (Group of SpinorOperators at primes (2,),
            Subgroup of Group of SpinorOperators at primes (2,) generated by (1, 1, f2, f1))
        """
        A, K = self._proper_spinor_kernel()
        if A.order() == K.order()
            return A, K
        b, j = self._proper_is_improper()
        if b:
            return A, K
        else
            K = A.subgroup(K.gens() + (j,))
            return A, K


    function spinor_generators(self, proper)
        r"""
        Return the spinor generators.

        INPUT:

        - ``proper`` -- boolean

        OUTPUT:

        a list of primes not dividing the determinant

        EXAMPLES::

            sage: g = matrix(ZZ, 3, [2,1,0, 1,2,0, 0,0,18])
            sage: gen = Genus(g)
            sage: gen.spinor_generators(false)
            [5]
        """
        from sage.sets.primes import Primes
        if proper:
            A, K = self._proper_spinor_kernel()
        else
            A, K = self._improper_spinor_kernel()
        Q = A.quotient(K)
        q = Q.order()
        U = Q.subgroup([])

        spinor_gens = []
        P = Primes()
        p = ZZ(2)
        while not U.order() == q:
            p = P.next(p)
            if p.divides(self.determinant())
                continue
            g = Q(A.delta(p))
            if g.gap() in U.gap() # containment in sage is broken
                continue
            else
                spinor_gens.append(p)
                U = Q.subgroup((g,) + Q.gens())
        return spinor_gens

    function _proper_is_improper(self)
        r"""
        Return if proper && improper spinor genus coincide.

        EXAMPLES::

            sage: gram = matrix(ZZ, 4, [2,0,1,0, 0,2,1,0, 1,1,5,0, 0,0,0,16])
            sage: genus = Genus(gram)
            sage: genus._proper_is_improper()
            (true, [2:1])

        This genus consists of only on (improper) class, hence spinor genus and
        improper spinor genus differ::

            sage: gram = matrix(ZZ, 4, [3,0,1,-1, 0,3,-1,-1, 1,-1,6,0, -1,-1,0,6])
            sage: genus = Genus(gram)
            sage: genus._proper_is_improper()
            (false, [2:7])
        """
        G = self.representative()
        d = self.dimension()
        V = ZZ^d
        # TODO:
        # this is a potential bottleneck
        # find a more clever way
        # with just the condition q != 0
        # even better would be a
        # version which does not require a representative
        norm = self.norm()
        P = [s.prime() for s in self._symbols]
        while true:
            x = V.random_element()
            q = x * G* x
            if q != 0 && all(q.valuation(p) == norm.valuation(p) for p in P)
                break
        Q = [p for p in q.prime_factors() if (norm.valuation(p) + q.valuation(p)) % 2 != 0]
        r = ZZ.prod(Q)
        # M = \tau_x(L)
        # q = [L: L & M]
        A, K = self._proper_spinor_kernel()
        j = A.delta(r) # diagonal embedding of r
        return j in K, j


=#
@doc Markdown.doc"""
    signature(G::ZGenus)

Return the signature of this genus.

The signature is `p - n` where `p` is the number of positive eigenvalues
and `n` the number of negative eigenvalues.
"""
function signature(G::ZGenus)
  p, n = G._signature_pair
  return p - n
end

@doc Markdown.doc"""
    signature_pair(G::ZGenus)

Return the signature_pair of this genus.

The signature is `[p, n]` where `p` is the number of positive eigenvalues
and `n` the number of negative eigenvalues.
"""
function signature_pair(G::ZGenus)
  return G._signature_pair
end

@doc Markdown.doc"""
    determinant(G::ZGenus)

Return the determinant of this genus.
"""
function determinant(G::ZGenus)
  p, n = G._signature_pair
  return (-1)^n*prod( prime(g)^sum(s[1]*s[2] for s in g._symbol) for g in G._symbols)
end

function det(G::ZGenus)
  return determinant(G)
end

function dimension(G::ZGenus)
  return sum(G._signature_pair)
end

function dim(G::ZGenus)
  return dimension(G)
end

function rank(G::ZGenus)
  return dimension(G)
end

#=
function represents(self, other)
        p1, m1 = self.signature_pair()
        p2, m2 = other.signature_pair()
        if not p1>=p2 && m1>=m2:
            return false
        primes = [s.prime() for s in self.local_symbols()]
        primes += [s.prime() for s in other.local_symbols()
                   if s.prime() not in primes]
        for p in primes:
            sp = self.local_symbols(p)
            op = other.local_symbols(p)
            if not sp.represents(op)
                   return false
        return true

=#
@doc Markdown.doc"""
    direct_sum(G1::ZGenus, G2::ZGenus)

Return the genus of the direct sum of ``G1`` and ``G2``.

The direct sum is defined via representatives.
"""
function direct_sum(G1::ZGenus, G2::ZGenus)
  p1, n1 = G1._signature_pair
  p2, n2 = G2._signature_pair
  signature_pair = [p1 + p2, n1 + n2]
  primes = [prime(s) for s in G1._symbols]
  append!(primes, [prime(s) for s in G2._symbols if !(prime(s) in primes)])
  sort(primes)
  local_symbols = []
  for p in primes
    sym_p = direct_sum(local_symbol(G1, p), local_symbol(G2, p))
    push!(local_symbols, sym_p)
  end
  return ZGenus(signature_pair, local_symbols)
end

@doc Markdown.doc"""
    discriminant_form(G::ZGenus)

Return the discriminant form associated to this genus.

  sage: A = matrix.diagonal(ZZ, [2, -4, 6, 8])
  sage: GS = Genus(A)
  sage: GS.discriminant_form()
  Finite quadratic module over Integer Ring with invariants (2, 2, 4, 24)
  Gram matrix of the quadratic form with values in Q/2Z:
  [ 1/2    0    0    0]
  [   0  3/2    0    0]
  [   0    0  7/4    0]
  [   0    0    0 7/24]
  sage: A = matrix.diagonal(ZZ, [1, -4, 6, 8])
  sage: GS = Genus(A)
  sage: GS.discriminant_form()
  Finite quadratic module over Integer Ring with invariants (2, 4, 24)
  Gram matrix of the quadratic form with values in Q/Z:
  [ 1/2    0    0]
  [   0  3/4    0]
  [   0    0 7/24]
"""
function discriminant_form(G::ZGenus)
  qL = fmpq_mat[]
  for gs in G._symbols
    p = gs._prime
    for block in gs._symbol
      q = _gram_from_jordan_block(p, block, true)
      push!(qL, q)
    end
  end
  q = diagonal_matrix(qL)
  return TorQuadMod(q)
end


@doc Markdown.doc"""
Compute a representative of this genus && cache it.


TESTS::

    sage: from sage.quadratic_forms.genera.genus import genera
    sage: for det in range(1, 5)
    ....:     G = genera((4,0), det, even=false)
    ....:     assert all(g==Genus(g.representative()) for g in G)
    sage: for det in range(1, 5)
    ....:     G = genera((1,2), det, even=false)
    ....:     assert all(g==Genus(g.representative()) for g in G)
    sage: for det in range(1, 9) # long time (8s, 2020)
    ....:     G = genera((2,2), det, even=false) # long time
    ....:     assert all(g==Genus(g.representative()) for g in G) # long time
"""
function representative(G::ZGenus)
  V = quadratic_space(G)
  L = lattice(V)
  L = maximal_integral_lattice(L)
  for sym in G._symbols
    p = prime(sym)
    L = local_modification(L, representative(sym), p)
  end
  # confirm the computation
  @hassert genus(L) == G
  G._representative = L
  return L
end


#=

    function representatives(self, backend=Nothing, algorithm=Nothing)
        r"""
        Return a list of representatives for the classes in this genus

        INPUT:

        - ``backend`` -- (default:``Nothing``)
        - ``algorithm`` -- (default:``Nothing``)

        OUTPUT:

        - a list of gram matrices

        EXAMPLES::

            sage: from sage.quadratic_forms.genera.genus import genera
            sage: G = Genus(matrix.diagonal([1, 1, 7]))
            sage: G.representatives()
            (
            [1 0 0]  [1 0 0]
            [0 2 1]  [0 1 0]
            [0 1 4], [0 0 7]
            )

        Indefinite genera work as well::

            sage: G = Genus(matrix(ZZ, 3, [6,3,0, 3,6,0, 0,0,2]))
            sage: G.representatives()
            (
            [2 0 0]  [ 2 -1  0]
            [0 6 3]  [-1  2  0]
            [0 3 6], [ 0  0 18]
            )

        For positive definite forms the magma backend is available::

            sage: G = Genus(matrix.diagonal([1, 1, 7]))
            sage: G.representatives(backend="magma")  # optional - magma
            (
            [1 0 0]  [ 1  0  0]
            [0 1 0]  [ 0  2 -1]
            [0 0 7], [ 0 -1  4]
            )
        """
        try:
            return self._representatives
        except AttributeError:
            pass
        n = self.dimension()
        representatives = []
        if n == 0:
            return (self.representative(), )
        if backend is Nothing:
            if n > 6 && prod(self.signature_pair_of_matrix()) == 0:
                backend = 'magma'
            else
                backend = 'sage'
        if backend == 'magma':
            if prod(self.signature_pair_of_matrix()) != 0:
                if n <= 2:
                    raise NotImplementedError()
                K = magma.RationalsAsNumberField()
                gram = magma.Matrix(K, n, self.representative().list())
                L = gram.NumberFieldLatticeWithGram()
                representatives = L.GenusRepresentatives()
                representatives = [r.GramMatrix().ChangeRing(magma.Rationals()).sage() for r in representatives]
            else
                e = 1
                if self.signature_pair_of_matrix()[1] != 0:
                    e = -1
                K = magma.Rationals()
                gram = magma.Matrix(K, n, (e*self.representative()).list())
                L = gram.LatticeWithGram()
                representatives = L.GenusRepresentatives()
                representatives = [e*r.GramMatrix().sage() for r in representatives]
        elseif backend == "sage":
            if n == 1:
                return [self.representative()]
            if n == 2:
                # Binary forms are considered positive definite take care of that.
                e = ZZ(1)
                if self.signature_pair()[0] == 0:
                    e = ZZ(-1)
                d = - 4 * self.determinant()
                from sage.quadratic_forms.binary_qf import BinaryQF_reduced_representatives
                for q in BinaryQF_reduced_representatives(d, proper=false)
                    if q[1] % 2 == 0:  # we want integrality of the gram matrix
                        m = e*matrix(ZZ, 2, [q[0], q[1] // 2, q[1] // 2, q[2]])
                        if Genus(m) == self:
                            representatives.append(m)
            if n > 2:
                from sage.quadratic_forms.quadratic_form import QuadraticForm
                from sage.quadratic_forms.quadratic_form__neighbors import neighbor_iteration
                e = ZZ(1)
                if not self.is_even()
                    e = ZZ(2)
                if self.signature_pair()[0] == 0:
                    e *= ZZ(-1)
                Q = QuadraticForm(ZZ,e*self.representative())
                seeds = [Q]
                for p in self.spinor_generators(proper=false)
                    v = Q.find_primitive_p_divisible_vector__next(p)
                    seeds.append(Q.find_p_neighbor_from_vec(p, v))
                if ZZ.prod(self.signature_pair()) != 0:
                    # indefinite genus && improper spinor genus agree
                    representatives = seeds
                else
                    # we do a neighbor iteration
                    from sage.sets.primes import Primes
                    P = Primes()
                    # we need a prime with L_p isotropic
                    # this is certainly the case if the lattice is even
                    # && p does not divide the determinant
                    if self.is_even()
                        p = ZZ(2)
                    else
                        p = ZZ(3)
                    det = self.determinant()
                    while p.divides(det)
                        p = P.next(p)
                    representatives = neighbor_iteration(seeds, p, mass=Q.conway_mass(), algorithm=algorithm)
                representatives = [g.Hessian_matrix() for g in representatives]
                representatives = [(g/e).change_ring(ZZ) for g in representatives]
        else
            raise ValueError("unknown algorithm")
        for g in representatives:
            g.set_immutable()
        self._representatives = tuple(representatives)
        assert length(representatives) > 0, self
        return self._representatives

=#
@doc Markdown.doc"""
        local_symbols(G::ZGenus)

Return a copy of the local symbols.
"""
function local_symbols(G::ZGenus)
  return deepcopy(G._symbols)
end

@doc Markdown.doc"""
        local_symbol(G::ZGenus, p)

Return the local symbol at `p`.
"""
function local_symbol(G::ZGenus, p)
  p = ZZ(p)
  for sym in G._symbols
    if p == prime(sym)
      return deepcopy(sym)
    end
  end
  @assert p!=2
  sym_p = [[0, rank(G), _kronecker_symbol(det(G),p)]]
  return ZpGenus(p, sym_p)
end



@doc Markdown.doc"""
    level(G::ZGenus)

Return the level of this genus.

This is the denominator of the inverse gram matrix
of a representative.
"""
function level(G::ZGenus)
  return prod(level(sym) for sym in G._symbols)
end

@doc Markdown.doc"""
    scale(G::ZGenus)

Return the scale of this genus.

Let `L` be a lattice with bilinear form `b`.
The scale of `(L,b)` is defined as the ideal
`b(L,L)`.
"""
function scale(G::ZGenus)
  return prod([scale(s) for s in G._symbols])
end

@doc Markdown.doc"""
    norm(G::ZGenus)


Return the norm of this genus.

Let `L` be a lattice with bilinear form `b`.
The norm of `(L,b)` is defined as the ideal
generated by `\{b(x,x) | x \in L\}`.
"""
function norm(G::ZGenus)
  return prod([norm(s) for s in G._symbols])
end
#=
=#

@doc Markdown.doc"""
    _gram_from_jordan_block(p, block, discr_form=false) -> MatElem

Return the gram matrix of this jordan block.

This is a helper for `discriminant_form` && `representative`.
No input checks.

INPUT:

- ``p`` -- a prime number

- ``block`` -- a list of 3 integers || 5 integers if `p` is `2`

- ``discr_form`` -- bool (default: ``false``); if ``true`` invert the scales
  to obtain a gram matrix for the discriminant form instead.

    EXAMPLES::

        sage: from sage.quadratic_forms.genera.genus import _gram_from_jordan_block
        sage: block = [1, 3, 1]
        sage: _gram_from_jordan_block(5, block)
        [5 0 0]
        [0 5 0]
        [0 0 5]
        sage: block = [1, 4, 7, 1, 2]
        sage: _gram_from_jordan_block(2, block)
        [0 2 0 0]
        [2 0 0 0]
        [0 0 2 0]
        [0 0 0 2]

    For the discriminant form we obtain::

        sage: block = [1, 3, 1]
        sage: _gram_from_jordan_block(5, block, true)
        [4/5   0   0]
        [  0 2/5   0]
        [  0   0 2/5]
        sage: block = [1, 4, 7, 1, 2]
        sage: _gram_from_jordan_block(2, block, true)
        [  0 1/2   0   0]
        [1/2   0   0   0]
        [  0   0 1/2   0]
        [  0   0   0 1/2]
"""
function _gram_from_jordan_block(p::fmpz, block, discr_form=false)
  level = block[1]
  rk = block[2]
  det = block[3]
  if p == 2
    o = block[4]
    t = block[5]
    U = QQ[0 1; 1 0]
    V = QQ[2 1; 1 2]
    W = QQ[1;]
    if o == 0
      if det in [1, 7]
        qL = [U for i in 1:div(rk, 2)]
      else
        qL = [U for i in 1:(div(rk, 2) - 1)]
        push!(qL, V)
      end
    elseif o == 1
      if rk % 2 == 1
        qL = [U for i in 1:max(0, div(rk - 3, 2))]
        if t*det % 8 in [3, 5]
          push!(qL,V)
        elseif rk >= 3
          push!(qL, U)
        end
        push!(qL, t * W)
      else
        if det in [3, 5]
          det = -1
        else
          det = 1
        end
        qL = [U for i in 1:max(0, div(rk - 4, 2))]
        if (det , t) == (1, 0)
          append!(qL, [U, 1 * W, 7 * W])
        elseif (det , t) == (1, 2)
          append!(qL, [U, 1 * W, 1 * W])
        elseif (det , t) == (1, 4)
          append!(qL , [V, 1 * W, 3 * W])
        elseif (det , t) == (1, 6)
          append!(qL, [U, 7 * W, 7 * W])
        elseif (det , t) == (-1, 0)
          append!(qL, [V, 1 * W, 7 * W])
        elseif (det , t) == (-1, 2)
          append!(qL, [U, 3 * W, 7 * W])
        elseif (det , t) == (-1, 4)
          append!(qL, [U, 1 * W, 3 * W])
        elseif (det , t) == (-1, 6)
          append!(qL, [U, 1 * W, 5 * W])
        else
          raise(error("invalid symbol $block"))
        end
          # if the rank is 2 there is a U too much
        if rk == 2
          qL = qL[end-1:end]
        end
      end
    end
    if size(qL)[1] != 0
      q = diagonal_matrix(qL)
    else
      q = zero_matrix(QQ, 0, 0)
    end
    if discr_form
      q = q * (1//2^level)
    else
      q = q * 2^level
    end
  elseif p != 2 && discr_form
    q = identity_matrix(QQ, rk)
    d = 2^(rk % 2)
    if _kronecker_symbol(d, p) != det
      u = _min_nonsquare(p)
      q[1,1] = u
    end
    q = q * (2 // p^level)
  end
  if p != 2 && !discr_form
    q = identity_matrix(QQ, rk)
    if det != 1
      u = _min_nonsquare(p)
      q[1,1] = u
    end
    q = q * p^level
  end
  return q
end






##################################################
# The mass formula
##################################################
@doc Markdown.doc"""
    _M_p(species, p)

Return the diagonal factor `M_p` as a function of the species.
"""
function _M_p(species, p)
  if species == 0
    return QQ(1)
  end
  p = QQ(p)
  n = abs(species)
  s = Int(div(n + 1,2))
  mp = 2 * prod(fmpq[1 - p^(-2*k) for k in 1:s-1])
  if n % 2 == 0
    mp *= ZZ(1) - sign(species) * p^(-s)
  end
  return QQ(1) // mp
end

@doc Markdown.doc"""
    _standard_mass_squared(G::ZGenus) -> fmpq

Return the standard mass of this genus.
It depends only on the dimension and determinant.
"""
function _standard_mass_squared(G::ZGenus)
  n = dimension(G)
  if n % 2 == 0
    s = div(n, 2)
  else
    s = div(n, 2) + 1
  end
  std = QQ(2)^2
  std *= prod(fmpq[_gamma_exact(j // 2) for j in 1:n])^2
  std *= prod(fmpq[_zeta_exact(2*k) for k in 1:s-1])^2
  if n % 2 == 0
    D = ZZ(-1)^(s) * determinant(G)
    std *= _quadratic_L_function_squared(ZZ(s), D)
    d = fundamental_discriminant(D)
    # since quadratic_L_function__exact is different
    # from \zeta_D as defined by Conway && Sloane
    # we have to compensate
    # the missing Euler factors
    for sym in G._symbols
      p = sym._prime
      std *= (1 - _kronecker_symbol(d, p)*QQ(p)^(-s))^2
    end
  end
  return std
end

@doc Markdown.doc"""
    mass(G::ZGenus) -> fmpq

Return the mass of this genus.

The genus must be definite.
Let `L_1, ... L_n` be a complete list of representatives
of the isometry classes in this genus.
Its mass is defined as

.. MATH::

    \sum_{i=1}^n \frac{1}{|O(L_i)|}.

INPUT:

- ``backend`` -- default: ``'sage'``, || ``'magma'``

OUTPUT:

a rational number

EXAMPLES::

    sage: from sage.quadratic_forms.genera.genus import genera
    sage: G = genera((8,0), 1, even=true)[0]
    sage: G.mass()
    1/696729600
    sage: G.mass(backend='magma')  # optional - magma
    1/696729600

The `E_8` lattice is unique in its genus::

    sage: E8 = QuadraticForm(G.representative())
    sage: E8.number_of_automorphisms()
    696729600

TESTS:

Check a random genus with magma::

    sage: d = ZZ.random_element(1, 1000)
    sage: n = ZZ.random_element(2, 10)
    sage: L = genera((n,0), d, d, even=false)
    sage: k = ZZ.random_element(0, length(L))
    sage: G = L[k]
    sage: G.mass()==G.mass(backend='magma')  # optional - magma
    true
"""
function mass(G::ZGenus)
  pos, neg = G._signature_pair
  @req pos * neg == 0 "the genus must be definite."
  if pos + neg == 1
    return QQ(1//2)
  end
  mass1 = _standard_mass_squared(G)
  mass1 *= prod(fmpq[_mass_squared(sym) for sym in G._symbols])
  mass1 //= prod(fmpq[_standard_mass(sym) for sym in G._symbols])^2
  return sqrt(mass1)
end


@doc Markdown.doc"""
    _mass_squared(G::ZpGenus) -> fmpq

Return the local mass `m_p` of this genus as defined by Conway.

See Equation (3) in [CS1988]_.

EXAMPLES::

  sage: G = Genus(matrix.diagonal([1, 3, 9]))
  sage: G.local_symbol(3).mass()
  9/8

  TESTS::

  sage: G = Genus(matrix([1]))
  sage: G.local_symbol(2).mass()
  Traceback (most recent call last)
  ....
  ValueError: the dimension must be at least 2
"""
function _mass_squared(G::ZpGenus)
  @req dimension(G) > 1 "the dimension must be at least 2"
  p = G._prime
  sym = G._symbol
  #diagonal product

  # diagonal factors
  m_p = prod(_M_p(species, p) for species in _species_list(G))^2
  # cross terms
  r = length(sym)
  ct = 0
  for j in 1:r
    for i in 1:j
        ct += (sym[j][1] - sym[i][1]) * sym[i][2] * sym[j][2]
    end
  end
  m_p *= p^ct
  if p != 2
    return m_p
  end
  # type factors
  nII = sum(fmpz[fq[2] for fq in sym if fq[4] == 0])
  nI_I = ZZ(0)   # the total number of pairs of adjacent constituents f_q,
  # f_2q that are both of type I (odd)
  for k in 1:r-1
    if sym[k][4] == sym[k+1][4] == 1 && sym[k][1] + 1 == sym[k+1][1]
      nI_I += ZZ(1)
    end
  end
  return m_p * QQ(2)^(2*(nI_I - nII))
end

@doc Markdown.doc"""
    _standard_mass(G::ZpGenus)

Return the standard p-mass of this local genus.

See Equation (6) of [CS1988]_.
"""
function _standard_mass(G::ZpGenus)
  n = dimension(G)
  p = G._prime
  s = div(n + 1, 2)
  std = 2*prod(fmpq[1 - QQ(p)^(-2*k) for k in 1:s-1])
  if n % 2 == 0
    D = ZZ(-1)^s * ZZ(determinant(G))
    epsilon = _kronecker_symbol(4*D, p)
    std *= (1 - epsilon*QQ(p)^(-s))
  end
  return QQ(1) // std
end

@doc Markdown.doc"""
    _species_list(G::ZpGenus)

Return the species list.
See Table 1 in [CS1988]_.
"""
function _species_list(G::ZpGenus)
  p = prime(G)
  species_list = []
  sym = G._symbol
  if p != 2
    for k in 1:length(sym)
      n = ZZ(sym[k][2])
      d = sym[k][3]
      if n % 2 == 0 && d != _kronecker_symbol(-1, p)^(div(n, 2))
        species = -n
      else
        species = n
      end
      push!(species_list, species)
    end
    return species_list
  end
  #  p == 2
  # create a dense list of symbols
  symbols = Vector{Int}[]
  s = 1
  for k in 0:sym[end][1]
    if sym[s][1] == k
      push!(symbols, sym[s])
      s +=1
    else
      push!(symbols,[k, 0, 1, 0, 0])
    end
  end
  # avoid a case distinction
  sym = [[-2, 0, 1, 0, 0],[-1, 0, 1, 0, 0]]
  append!(sym, symbols)
  push!(sym, [sym[end-1][1] + 1, 0, 1, 0, 0])
  push!(sym, [sym[end-1][1] + 2, 0, 1, 0, 0])
  for k in 2:length(sym)-1
    free = true
    if sym[k-1][4]==1 || sym[k+1][4]==1
      free = false
    end
    n = sym[k][2]
    o = sym[k][5]
    if _kronecker_symbol(sym[k][3], 2) == -1
      o = mod(o + 4, 8)
    end
    if sym[k][4] == 0 || n % 2 == 1
      t = div(n, 2)
    else
      t = div(n, 2) - 1
    end
    if free && (o == 0 || o == 1 || o == 7)
      species = 2*t
    elseif free && (o == 3 || o == 5 || o == 4)
      species = -2*t
    else
      species = 2*t + 1
    end
    push!(species_list, species)
  end
  return species_list
end




@doc Markdown.doc"""
    _gamma_exact(n)

Evaluates the exact value of the `\Gamma^2` function at an integer or
half-integer argument. Ignoring factors of pi
"""
function _gamma_exact(n)
  n = QQ(n)
  if denominator(n) == 1
    @req (n > 0) "not in domain"
    return factorial(ZZ(n) - 1)
  end
  @req (denominator(n) == 2) "n must be in (1/2)ZZ"
  a = QQ(1)
  while n != 1//2
    if n < 0
      a //= n
      n += 1
    elseif n > 0
      n += -1
      a *= n
    end
  end
  return a
end

@doc Markdown.doc"""
    _zeta_exact(n)

Returns the exact value of the Riemann Zeta function
ignoring factors of pi.

The argument must be a critical value, namely either positive even
or negative odd.

See for example [Iwa1972]_, p13, Special value of `\zeta(2k)`
REFERENCES:

- [Iwa1972]_
- [IR1990]_
- [Was1997]_
"""
function _zeta_exact(n)
  if n < 0
    return bernoulli(1-n)//(n-1)
  elseif n > 1
    if (n % 2 == 0)
      return QQ(-1)^(div(n, 2) + 1) * QQ(2)^(n-1) * bernoulli(n) // factorial(ZZ(n))
    else
      raise(error("n must be a critical value (i.e. even > 0 or odd < 0)"))
    end
  elseif n == 1
    raise(error("Here is a pole"))
  elseif n == 0
    return QQ(-1// 2)
  end
end

@doc Markdown.doc"""
    _quadratic_L_function_squared(n, d)

Returns the square of the exact value of a quadratic twist of the Riemann Zeta function by `\chi_d(x) = \left(\frac{d}{x}\right)`.
We take the square and ignore multiples of pi so that the output is rational.

The input `n` must be a critical value.

- [Iwa1972]_, pp 16-17, Special values of `L(1-n, \chi)` and `L(n, \chi)`
- [IR1990]_
- [Was1997]_
"""
function _quadratic_L_function_squared(n, d)
  if n <= 0
      return _bernoulli_kronecker(1-n, d)^2//(n-1)^2
  end
  @req (n >= 1) "n must be a critical value (i.e. even > 0 or odd < 0)"
  # Compute the kind of critical values (p10)
  if _kronecker_symbol(fundamental_discriminant(d), -1) == 1
    delta = 0
  else
    delta = 1
  end
  # Compute the positive special values (p17)
  @req (mod(n - delta, 2) == 0) "not in domain"
  f = abs(fundamental_discriminant(d))
  if delta == 0
      GS = f
  else
      GS = -f
  end
  a = ZZ(-1)^(2 + (n - delta))
  a *= (2//f)^(2*n)
  a *= GS     # Evaluate the Gauss sum here! =0
  a *= 1//(4 * (-1)^delta)
  a *= _bernoulli_kronecker(Int(n),d)^2//factorial(ZZ(n))^2
  return a
end
