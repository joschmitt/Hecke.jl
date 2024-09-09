@doc raw"""
    rref(M::SMat{T}; truncate = false) where {T <: FieldElement} -> (Int, SMat{T})

Return a tuple $(r, A)$ consisting of the rank $r$ of $M$ and a reduced row echelon
form $A$ of $M$.
If the function is called with `truncate = true`, the result will not contain zero
rows, so `nrows(A) == rank(M)`.
"""
function rref(A::SMat{T}; truncate::Bool = false) where {T <: FieldElement}
  B = deepcopy(A)
  r = rref!(B, truncate = truncate)
  return r, B
end

# This does not really work in place, but it certainly changes A
function rref!(A::SMat{T}; truncate::Bool = false) where {T <: FieldElement}
  B = sparse_matrix(base_ring(A))
  B.c = A.c
  number_of_rows = A.r

  # Remove empty rows, so they don't get into the way when we sort
  i = 1
  while i <= length(A.rows)
    if iszero(A.rows[i])
      deleteat!(A.rows, i)
    else
      i += 1
    end
  end
  A.r = length(A.rows)

  # Prefer rows with more zeros in front and, if the position of the first
  # non-zero entry is equal, sparse rows.
  # This way, we should discover the pivots "bottom up" and need to reduce
  # less when adding a row to the rref.
  rows = sort!(A.rows, lt = (x, y) -> x.pos[1] > y.pos[1] || (x.pos[1] == y.pos[1] && length(x) < length(y)))
  #rows = sort!(A.rows, lt = (x, y) -> length(x) < length(y) || (length(x) == length(y) && x.pos[1] > y.pos[1]))

  for r in rows
    b = _add_row_to_rref!(B, r)
    if nrows(B) == ncols(B)
      break
    end
  end

  A.nnz = B.nnz
  A.rows = B.rows
  rankA = B.r
  if !truncate
    while length(A.rows) < number_of_rows
      push!(A.rows, sparse_row(base_ring(A)))
    end
    A.r = number_of_rows
  else
    A.r = B.r
  end
  return rankA
end

# Reduce v by M and if the result is not zero add it as a row (and then reduce
# M to maintain the rref).
# Return true iff v is not in the span of the rows of M.
# M is supposed to be in rref and both M and v are changed in place.
function _add_row_to_rref!(M::SMat{T}, v::SRow{T}) where { T <: FieldElem }
  if iszero(v)
    return false
  end

  pivot_found = false
  s = one(base_ring(M))
  i = 1
  new_row = 1
  while i <= length(v)
    c = v.pos[i]
    r = find_row_starting_with(M, c)
    if r > nrows(M) || M.rows[r].pos[1] > c
      # We found an entry in a column of v, where no other row of M has the pivot.
      @assert !iszero(v.values[i])
      i += 1
      if pivot_found
        # We already found a pivot
        continue
      end

      @assert i == 2 # after incrementing
      pivot_found = true
      new_row = r
      continue
    end

    # Reduce the entries of v by M.rows[r]
    t = -v.values[i] # we assume M.rows[r].pos[1] == 1 (it is the pivot)
    v = add_scaled_row!(M.rows[r], v, t)
    # Don't increase i, we deleted the entry
  end
  if !pivot_found
    return false
  end

  # Multiply v by inv(v.values[1])
  if !isone(v.values[1])
    t = inv(v.values[1])
    for j = 2:length(v)
      v.values[j] = mul!(v.values[j], v.values[j], t)
    end
    v.values[1] = one(base_ring(M))
  end
  insert!(M, new_row, v)

  # Reduce the rows above the newly inserted one
  for i = 1:new_row - 1
    c = M.rows[new_row].pos[1]
    j = searchsortedfirst(M.rows[i].pos, c)
    if j > length(M.rows[i].pos) || M.rows[i].pos[j] != c
      continue
    end

    t = -M.rows[i].values[j]
    l = length(M.rows[i])
    M.rows[i] = add_scaled_row!(M.rows[new_row], M.rows[i], t)
    if length(M.rows[i]) != l
      M.nnz += length(M.rows[i]) - l
    end
  end
  return true
end

### Sparse RREF using Markowitz pivoting to produce minimal fill-in
# References:
# * Joux: Algorithmic cryptanalysis, Chapman & Hall/CRC, 2009,
#   Section 3.4.2.2
# * Duff, Erisman, Reid: Direct methods for sparse matrices,
#   Oxford University Press, 2nd edition, 2017
#   Sections 7.2 and 10.2

# Implements doubly-linked lists with headers to keep track of the length of
# rows and columns (by length we mean "number of non-zero entries")
# See Duff, Erisman, Reid: Direct methods for sparse matrices,
#     Oxford University Press, 2nd edition, 2017
#     Section 10.2
#
# If we store lengths of rows of a matrix A, then forward_links and backward_links
# have length nrows(A) and headers has length ncols(A).
# * headers[l] is a row index with a row of length l; if headers[l] == 0, there
#   are no rows of length l
# * forward_links[i] gives the next row of length(A.rows[i]) (or 0 if there is
#   no further row)
# * backward_links[i] gives the previous row of length(A.rows[i]) (or 0 if there
#   is no such row)
# That means one can discover the indices of all rows of a given length l with
# the loop:
#   r = headers[l]
#   while r != 0
#     r = forward_links[r]
#   end
#
# If lengths of columns are stored, the above holds true with the words "row"
# and "column" swapped.
struct MarkowitzStorage
  forward_links::Vector{Int}
  backward_links::Vector{Int}
  headers::Vector{Int}

  function MarkowitzStorage(n::Int, l::Int)
    return new(zeros(Int, n), zeros(Int, n), zeros(Int, l))
  end
end

# Add the entry (row or column) of index i with length l
function _add_entry!(S::MarkowitzStorage, i::Int, l::Int)
  @assert S.forward_links[i] == 0 && S.backward_links[i] == 0
  j = S.headers[l]
  @assert i != j
  S.forward_links[i] = j
  if j != 0
    S.backward_links[j] = i
  end
  S.headers[l] = i
  return nothing
end

# Delete the entry (row or column) of index i with length l, that is, set all
# links to 0
function _delete_entry!(S::MarkowitzStorage, i::Int, l::Int)
  if S.backward_links[i] == 0
    if S.headers[l] != i
      # i (of length l) is not an entry, so nothing to delete
      @assert S.forward_links[i] == 0
      return nothing
    end
    S.headers[l] = S.forward_links[i]
  end
  if S.backward_links[i] != 0
    S.forward_links[S.backward_links[i]] = S.forward_links[i]
  end
  if S.forward_links[i] != 0
    S.backward_links[S.forward_links[i]] = S.backward_links[i]
  end
  S.forward_links[i] = 0
  S.backward_links[i] = 0
  return nothing
end

# For a sparse matrix A with (pseudo) transpose AT, produce the lists of row and
# column lengths
function _initialize_markowitz_storage(A::SMat, AT::Vector{Vector{Int}})
  row_storage = MarkowitzStorage(nrows(A), ncols(A))
  col_storage = MarkowitzStorage(ncols(A), nrows(A))
  @inbounds for r in 1:nrows(A)
    is_zero(length(A.rows[r])) && continue
    _add_entry!(row_storage, r, length(A.rows[r]))
  end
  @inbounds for c in 1:ncols(A)
    is_zero(length(AT[c])) && continue
    _add_entry!(col_storage, c, length(AT[c]))
  end
  return row_storage, col_storage
end

# Find an entry (r, c) of A such that the product (R - 1) * (C - 1) is minimized,
# where R is the number of entries in the row r and C the number of entries in
# the column c
function _find_next_pivot(A::SMat, AT::Vector{Vector{Int}}, row_counts::MarkowitzStorage, col_counts::MarkowitzStorage, pivot_rows::BitVector)
  r_min = 0
  c_min = 0
  w_min = nrows(A)*ncols(A)

  l = 1
  @inbounds while l <= min(nrows(A), ncols(A))
    l1 = l - 1
    # First, search through the rows of length l
    r = row_counts.headers[l]
    # We already search through all rows and columns of length <= l - 1,
    # so the best we can get is (l - 1)^2
    break_min = l1^2
    w_min <= break_min && return r_min, c_min
    while r != 0
      for c in A.rows[r].pos
        # column c cannot have a pivot as the columns with a pivot are reduced
        w = l1 * (length(AT[c]) - 1)
        if w < w_min
          r_min = r
          c_min = c
          w_min = w
          w_min <= break_min && return r_min, c_min
        end
      end
      r = row_counts.forward_links[r]
    end

    # Now search through the columns of length l
    c = col_counts.headers[l]
    # We already search through all rows of length <= l and columns of
    # length <= l - 1, so the best we can get is (l - 1) * l
    break_min = l1 * l
    w_min <= break_min && return r_min, c_min
    while c != 0
      for r in AT[c]
        pivot_rows[r] && continue
        w = l1 * (length(A.rows[r]) - 1)
        if w < w_min
          r_min = r
          c_min = c
          w_min = w
          w_min <= break_min && return r_min, c_min
        end
      end
      c = col_counts.forward_links[c]
    end
    l += 1
  end
  return r_min, c_min
end

# Add t*A.rows[l] to A.rows[k] and update AT accordingly
# WARNING: the ordering of the arguments is different than in add_scaled_row!
function _add_scaled_row_with_transpose!(A::SMat{T}, k::Int, l::Int, t::T, AT::Vector{Vector{Int}}, t1::T = base_ring(A)()) where {T <: FieldElement}
  a = A.rows[l]
  b = A.rows[k]

  i = 1
  j = 1
  @inbounds while i <= length(a) && j <= length(b)
    if a.pos[i] < b.pos[j]
      t1 = mul!(t1, t, a.values[i])
      if !is_zero(t1)
        insert!(b.pos, j, a.pos[i])
        insert!(b.values, j, deepcopy(t1))
        j += 1
        A.nnz += 1
        jj = searchsortedfirst(AT[a.pos[i]], k)
        @assert jj > length(AT[a.pos[i]]) || AT[a.pos[i]][jj] > k
        insert!(AT[a.pos[i]], jj, k)
      end
      i += 1
    elseif a.pos[i] > b.pos[j]
      j += 1
    else
      t1 = mul!(t1, t, a.values[i])
      b.values[j] = addeq!(b.values[j], t1)

      if is_zero(b.values[j])
        deleteat!(b.values, j)
        deleteat!(b.pos, j)
        A.nnz -= 1
        jj = searchsortedfirst(AT[a.pos[i]], k)
        @assert AT[a.pos[i]][jj] == k
        deleteat!(AT[a.pos[i]], jj)
      else
        j += 1
      end
      i += 1
    end
  end
  @inbounds while i <= length(a)
    t1 = mul!(t1, t, a.values[i])
    if !is_zero(t1)
      push!(b.pos, a.pos[i])
      push!(b.values, deepcopy(t1))
      A.nnz += 1
      jj = searchsortedfirst(AT[a.pos[i]], k)
      @assert jj > length(AT[a.pos[i]]) || AT[a.pos[i]][jj] > k
      insert!(AT[a.pos[i]], jj, k)
    end
    i += 1
  end
  return nothing
end

function rref_markowitz!(A::SMat{T}) where {T <: FieldElement}
  # "Pseudo" transpose of A: AT[c] is the list of indices r such that A[r, c] is
  # non-zero
  AT = Vector{Vector{Int}}()
  for i in 1:ncols(A)
    push!(AT, Vector{Int}())
  end
  for i in 1:nrows(A)
    for j in A.rows[i].pos
      push!(AT[j], i)
    end
  end

  # If pivot_cols[c] == true, then we found a pivot in column c
  # If pivot_rows[r] == true, then we found a pivot in row r
  pivot_cols = falses(ncols(A))
  pivot_rows = falses(nrows(A))

  # If pivots[r] == c with c != 0, then the pivot in row r is in column c
  pivots = zeros(Int, nrows(A))

  row_counts, col_counts = _initialize_markowitz_storage(A, AT)

  t = base_ring(A)()
  t1 = base_ring(A)()
  @inbounds while true
    r_pivot, c_pivot = _find_next_pivot(A, AT, row_counts, col_counts, pivot_rows)
    r_pivot == 0 && break
    @assert !pivot_cols[c_pivot]
    pivot_cols[c_pivot] = true
    @assert !pivot_rows[r_pivot]
    pivot_rows[r_pivot] = true
    @assert pivots[r_pivot] == 0
    pivots[r_pivot] = c_pivot
    _delete_entry!(row_counts, r_pivot, length(A.rows[r_pivot]))
    _delete_entry!(col_counts, c_pivot, length(AT[c_pivot]))
    a = A.rows[r_pivot]

    # Scale the pivot to 1
    p = searchsortedfirst(a.pos, c_pivot)
    if !is_one(a.values[p])
      t = inv(a.values[p])
      for j = 1:length(a)
        j == p && continue
        a.values[j] = mul!(a.values[j], a.values[j], t)
      end
      a.values[p] = one(base_ring(A))
    end

    # Delete all columns with an entry in a; the lengths of these columns will
    # most likely be changed during the reduction
    for c in a.pos
      _delete_entry!(col_counts, c, length(AT[c]))
    end

    # Reduce the rows that have an entry in position c_pivot
    for r in copy(AT[c_pivot])
      r == r_pivot && continue
      b = A.rows[r]
      pb = searchsortedfirst(b.pos, c_pivot)
      @assert pb <= length(b) && b.pos[pb] == c_pivot

      # Delete all columns with an entry in b; the lengths of these columns will
      # most likely be changed during the reduction
      _delete_entry!(row_counts, r, length(b))
      for c in b.pos
        _delete_entry!(col_counts, c, length(AT[c]))
      end

      # Reduce b by a
      t = -b.values[pb]
      _add_scaled_row_with_transpose!(A, r, r_pivot, t, AT, t1)

      is_empty(b) && continue

      # Update the column counts for all columns of b that do not appear in a
      i = 1
      j = 1
      while j <= length(b)
        if i > length(a) || b.pos[j] < a.pos[i]
          c = b.pos[j]
          !pivot_cols[c] && _add_entry!(col_counts, c, length(AT[c]))
          j += 1
        elseif a.pos[i] < b.pos[j]
          i += 1
        else
          i += 1
          j += 1
        end
      end
      !pivot_rows[r] && _add_entry!(row_counts, r, length(b))
    end

    # Update the column counts for all columns that appear in a
    for c in a.pos
      !pivot_cols[c] && _add_entry!(col_counts, c, length(AT[c]))
    end
  end

  # At this point, we found all the pivots
  # Now go over the entries of A and make sure that every entry is to the right
  # of its pivot (this is only relevant if A does not have full rank)

  # Sort the pivots by decreasing column number
  p_sorted = sort!([(r, pivots[r]) for r in 1:nrows(A)], lt = (x, y) -> x[2] > y[2])
  @inbounds for (r, c) in p_sorted
    c == 0 && break

    a = A.rows[r]
    c_new = a.pos[1]
    c_new == c && continue

    # We messed up: the pivot in row r is in column c, but there is an entry in
    # column c_new. We now declare (r, c_new) to be the pivot (and reduce again)
    @assert c_new < c

    pivot_cols[c_new] = true
    pivot_cols[pivots[r]] = false
    pivots[r] = c_new

    # Rescale a so that the "new" pivot is 1
    if !is_one(a.values[1])
      t = inv(a.values[1])
      for j = 2:length(a)
        a.values[j] = mul!(a.values[j], a.values[j], t)
      end
      a.values[1] = one(base_ring(A))
    end

    # Reduce all relevant rows again (we don't need to update row_counts
    # or col_counts anymore)
    for i in copy(AT[c_new])
      i == r && continue
      j = searchsortedfirst(A.rows[i].pos, c_new)
      @assert A.rows[i].pos[j] == c_new

      t = -A.rows[i].values[j]
      _add_scaled_row_with_transpose!(A, i, r, t, AT, t1)
    end
  end

  # Final step: sort the rows
  @inbounds A.rows = A.rows[sortperm(pivots)]
  while length(A.rows) != 0 && isempty(A.rows[1])
    deleteat!(A.rows, 1)
    A.r -= 1
  end

  return A
end

###############################################################################
#
#   Kernel
#
###############################################################################

@doc raw"""
    nullspace(M::SMat{T}) where {T <: FieldElement}

Return a tuple $(\nu, N)$ consisting of the nullity $\nu$ of $M$ and
a basis $N$ (consisting of column vectors) for the right nullspace of $M$,
i.e. such that $MN$ is the zero matrix. If $M$ is an $m\times n$ matrix
$N$ will be a $n\times \nu$ matrix in dense representation. The columns of $N$
are in upper-right reduced echelon form.
"""
function nullspace(M::SMat{T}) where {T <: FieldElement}
  rank, A = rref(M, truncate = true)
  nullity = ncols(M) - rank
  K = base_ring(M)
  X = zero_matrix(K, ncols(M), nullity)
  if rank == 0
    for i = 1:nullity
      X[i, i] = one(K)
    end
  elseif nullity != 0
    r = 1
    k = 1
    for c = 1:ncols(A)
      if r <= rank && A.rows[r].pos[1] == c
        r += 1
        continue
      end

      for i = 1:r - 1
        j = searchsortedfirst(A.rows[i].pos, c)
        if j > length(A.rows[i].pos) || A.rows[i].pos[j] != c
          continue
        end
        X[A.rows[i].pos[1], k] = -A.rows[i].values[j]
      end
      X[c, k] = one(K)
      k += 1
    end
  end
  return nullity, X
end

@doc raw"""
    _left_kernel(M::SMat{T}) where {T <: FieldElement}

Return a tuple $\nu, N$ where $N$ is a matrix whose rows generate the
left kernel of $M$, i.e. $NM = 0$ and $\nu$ is the rank of the kernel.
If $M$ is an $m\times n$ matrix $N$ will be a $\nu\times m$ matrix in dense
representation. The rows of $N$ are in lower-left reduced echelon form.
"""
function _left_kernel(M::SMat{T}) where T <: FieldElement
  n, N = nullspace(transpose(M))
  return n, transpose(N)
end

@doc raw"""
    _right_kernel(M::SMat{T}) where {T <: FieldElement}

Return a tuple $\nu, N$ where $N$ is a matrix whose columns generate the
right kernel of $M$, i.e. $MN = 0$ and $\nu$ is the rank of the kernel.
If $M$ is an $m\times n$ matrix $N$ will be a $n \times \nu$ matrix in dense
representation. The columns of $N$ are in upper-right reduced echelon form.
"""
function _right_kernel(M::SMat{T}) where T <: FieldElement
  return nullspace(M)
end

@doc raw"""
    kernel(M::SMat{T}; side::Symbol = :left) where {T <: FieldElement}

Return a matrix $N$ containing a basis of the kernel of $M$.
If `side` is `:left` (default), the left kernel is
computed, i.e. the matrix of rows whose span gives the left kernel
space. If `side` is `:right`, the right kernel is computed, i.e. the matrix
of columns whose span is the right kernel space.
"""
function kernel(M::SMat{T}; side::Symbol = :left) where T <: FieldElement
  Solve.check_option(side, [:right, :left], "side")
  if side == :right
    return _right_kernel(M)[2]
  elseif side == :left
    return _left_kernel(M)[2]
  end
end
