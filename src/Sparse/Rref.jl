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

# Adds t*A.rows[l] to A.rows[k] (the ordering is different than in add_scaled_row!!!!)
function _add_scaled_row_with_transpose!(A::SMat{T}, k::Int, l::Int, t::T, AT::Vector{Vector{Int}}, t1::T = base_ring(A)()) where {T <: FieldElement}
  a = A.rows[l]
  b = A.rows[k]

  i = 1
  j = 1
  while i <= length(a) && j <= length(b)
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
  while i <= length(a)
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
  # "Pseudo" transpose of A: AT[c] is the list indices r such that A[r, c] is
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

  # For a column c, weights[c] is the list of pairs (r, w) where w gives the
  # weight of the entry A[r, c].
  # The weight of A[r, c] is defined as (R - 1)*(C - 1), where R is the number
  # of non-zero entries in row r and C the number of non-zero entries in column
  # C.
  # The Vector weights[c] is sorted by increasing weight.
  # TODO: Possibly, replace this with a fancier data structure.
  weights = Vector{Vector{Tuple{Int, Int}}}()
  for c in 1:ncols(A)
    push!(weights, Vector{Tuple{Int, Int}}())
    for r in AT[c]
      lr = length(A.rows[r])
      lc = length(AT[c])
      push!(weights[c], (r, (lr - 1)*(lc - 1)))
    end
    sort!(weights[c], lt = (x, y) -> x[2] < y[2])
  end

  # If pivot_cols[c] == r with r != 0, then the pivot of column c is in row r
  # If pivot_rows[r] == c with c != 0, then the pivot of row r is in column c
  pivot_cols = zeros(Int, ncols(A))
  pivot_rows = zeros(Int, nrows(A))

  t = base_ring(A)()
  t1 = base_ring(A)()
  t2 = base_ring(A)()
  while true
    c_min = 0
    min_weight = (nrows(A) - 1)*(ncols(A) - 1)
    for c in 1:ncols(A)
      !is_zero(pivot_cols[c]) && continue
      is_empty(weights[c]) && continue
      if min_weight >= weights[c][1][2]
        c_min = c
        min_weight = weights[c][1][2]
      end
    end
    c_min == 0 && break
    r_min = weights[c_min][1][1]
    @assert pivot_cols[c_min] == 0
    pivot_cols[c_min] = r_min
    @assert is_zero(pivot_rows[r_min])
    pivot_rows[r_min] = c_min
    weights[c_min] = Vector{Tuple{Int, Int}}()
    a = A.rows[r_min]
    for c in a.pos
      c == c_min && continue
      j = findfirst(x -> x[1] == r_min, weights[c])
      @assert !isnothing(j)
      deleteat!(weights[c], j)
    end

    p = searchsortedfirst(a.pos, c_min)
    if !is_one(a.values[p])
      t = inv(a.values[p])
      for j = 1:length(a)
        j == p && continue
        a.values[j] = mul!(a.values[j], a.values[j], t)
      end
      a.values[p] = one(base_ring(A))
    end
    for r in copy(AT[c_min])
      r == r_min && continue
      b = A.rows[r]
      pb = searchsortedfirst(b.pos, c_min)
      @assert pb <= length(b) && b.pos[pb] == c_min

      for c in b.pos
        j = findfirst(x -> x[1] == r, weights[c])
        if !isnothing(j)
          deleteat!(weights[c], j)
        end
      end

      t = -b.values[pb]
      _add_scaled_row_with_transpose!(A, r, r_min, t, AT, t1)
      !is_zero(pivot_rows[r]) && continue

      for c in b.pos
        w = (length(b) - 1)*(length(AT[c]) - 1)
        rw = (r, w)
        j = searchsortedfirst(weights[c], rw, lt = (x, y) -> x[2] < y[2])
        insert!(weights[c], j, rw)
      end
    end
  end

  # At this point, we found all the pivots there
  # Now go over the entries of A and make sure that every entry is to the right
  # of its pivot (this is only relevant if A does not have full rank)

  for c in 1:ncols(A)
    for r in copy(AT[c])
      pivot_rows[r] <= c && continue

      a = A.rows[r]
      c_new = a.pos[1]
      pivot_cols[c_new] = r
      pivot_cols[pivot_rows[r]] = 0
      pivot_rows[r] = c_new
      # We messed up: the pivot in row r has to be in column c_new

      if !is_one(a.values[1])
        t = inv(a.values[1])
        for j = 2:length(a)
          a.values[j] = mul!(a.values[j], a.values[j], t)
        end
        a.values[1] = one(base_ring(A))
      end

      for i in copy(AT[c_new])
        i == r && continue
        j = searchsortedfirst(A.rows[i].pos, c_new)
        @assert A.rows[i].pos[j] == c_new

        t = -A.rows[i].values[j]
        _add_scaled_row_with_transpose!(A, i, r, t, AT, t1)
      end
    end
  end

  # Final step: sort the rows
  A.rows = A.rows[sortperm(pivot_rows)]
  while isempty(A.rows[1])
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
