#function algebra(M::Vector{T}) where {T <: MatElem}
#  @assert length(M) > 0
#  A = M[1]
#  n = nrows(A)
#  n2 = n^2
#  @assert n == ncols(A)
#  K = base_ring(A)
#  Mprod = M
#  Morig = copy(M)
#
#  current_dim = -1
#
#  B = zero_matrix(K, length(Mprod) + 1, n2)
#
#  l = 0
#  while true
#    if l != 0
#      B = zero_matrix(K, length(Mprod), n2)
#    end
#    for k in 1:length(Mprod)
#      for i in 1:n
#        for j in 1:n
#          B[k, (i - 1)* n  + j] = Mprod[k][i, j]
#        end
#      end
#    end
#    # Add the identity
#    if l == 0
#      for i in 1:n
#        B[length(M) + 1, (i - 1)*n + i] = one(K)
#      end
#    end
#    new_dim = rref!(B)
#    if new_dim == current_dim
#      break
#    end
#    current_dim = new_dim
#    M = [ matrix(K, n, n, [B[k, (i - 1)*n + j] for i in 1:n for j in 1:n]) for k in 1:new_dim]
#    Mprod = [ M[i] * M[j] for i in 1:length(M) for j in 1:length(M) ]
#    l = l + 1
#  end
#
#  dim = current_dim
#  B = sub(B, 1:dim, 1:ncols(B))
#
#  basis = [ matrix(K, n, n, [B[k, (i - 1)*n + j] for i in 1:n for j in 1:n]) for k in 1:dim]
#
#  @assert isone(basis[1])
#
#  v = zero_matrix(K, 1, n2)
#
#  structure = Array{elem_type(K), 3}(dim, dim, dim)
#
#  for k in 1:dim
#    for l in 1:dim
#      N = basis[k] * basis[l]
#      for i in 1:n
#        for j in 1:n
#          v[1, (i - 1)* n  + j] = N[i, j]
#        end
#      end
#      b, u = can_solve_with_solution(B, v, side = :left)
#      error("NOT HERE!")
#      @assert b
#      @assert N == sum(u[i]*basis[i] for i in 1:dim)
#      for m in 1:dim
#        structure[k, l, m] = u[m, 1]
#      end
#    end
#  end
#
#  A = AlgAss(K, structure)
#
#  gens = Vector{AlgAssElem{elem_type(K)}}(length(Morig))
#
#  for l in 1:length(Morig)
#    N = Morig[l]
#    for i in 1:n
#      for j in 1:n
#        v[1, (i - 1)* n  + j] = N[i, j]
#      end
#    end
#    b, u = can_solve_with_solution(B, v, side = :left)
#    gens[l] =  A([u[1, m] for m in 1:dim])
#  end
#
#  A.gens = gens
#
#  return A
#end

#function gens(A::AlgAss{T}) where {T}
#  #return A.gens::Vector{AlgAssElem{T}}
#end

##

include("Lattices/Types.jl")
include("Lattices/Basics.jl")
include("Lattices/Reduction.jl")
include("Lattices/Morphisms.jl")

################################################################################
#
#  Local isomorphism
#
################################################################################

function _lift_to_Q(K::gfp_mat)
  z = zero_matrix(QQ, nrows(K), ncols(K))
  for i in 1:nrows(K)
    for j in 1:ncols(K)
      u = K[i, j].data
      if iszero(u)
        continue
      else
        z[i, j] = u
      end
    end
  end
  return z
end

function _lift_to_Q!(z, K::gfp_mat)
  for i in 1:nrows(K)
    for j in 1:ncols(K)
      u = K[i, j].data
      if iszero(u)
        continue
      else
        z[i, j] = u
      end
    end
  end
  return z
end

function addmul!(A::gfp_mat, B::gfp_mat, C::gfp_elem, D::gfp_mat)
  ccall((:nmod_mat_scalar_addmul_ui, libflint), Cvoid, (Ref{gfp_mat}, Ref{gfp_mat}, Ref{gfp_mat}, UInt), A, B, D, C.data)
  return A
end

function mul!(A::gfp_mat, B::gfp_elem, D::gfp_mat)
  ccall((:nmod_mat_scalar_mul_ui, libflint), Cvoid, (Ref{gfp_mat}, Ref{gfp_mat}, UInt), A, C, B.data)
end

function pmaximal_sublattices(L::ModAlgAssLat, p::Int; filter = nothing, composition_factors = nothing)
  res = typeof(L)[]
  if composition_factors === nothing
    V = reduction(L, p)
    F = V.base_ring
    comp_fac = Hecke.composition_factors(V)
  else
    comp_fac = composition_factors
    F = coefficient_ring(comp_fac[1])
    V = change_coefficient_ring(F, L)
  end
  for C in comp_fac
    H = basis_of_hom(V, C)
    if length(H) == 0
      continue
    end
    # We have to loop over all morphisms, but we can discard multiples
    # So we do a projective space thingy
    pivs = Int[]
    Kl = zero_matrix(QQ, rank(L), rank(L))
    for v in enumerate_lines(F, length(H))
      zero!(Kl)
      empty!(pivs)
      #k = sum(H[i] * v[i] for i in 1:length(H))
      if length(H) == 1
        k = H[1]
      else
        k = v[1] * H[1]
        for i in 2:length(H)
          if iszero(v[i])
            continue
          end
          addmul!(k, k, v[i], H[i])
          #k = add!(k ,k, v[i] * H[i])
        end
      end
      r, K = left_kernel(k)
      _, K = rref(K)
      # The lattice we are looking is generated by lift(K) + pZ^n
      # The HNF of this lattice is/would be the lift(rref(K) and p's added in
      # the zero-rows such that the rank is full.
      # We scale the rows and check the pivots
      l = 1
      m = r + 1
      for i in 1:r
        while iszero(K[i, l])
          l += 1
        end
        # The rows are not normalized
        push!(pivs, l)
      end
      # We lift directly to Q
      Kl = _lift_to_Q!(Kl, K)
      # Set the zero rows correctly
      for i in 1:nrows(K)
        if !(i in pivs)
          Kl[m, i] = p
          m += 1
        end
      end
      # Kl has the same Z-span as fmpq_mat(hnf_modular_eldiv(lift(K), fmpz(p)))
      # We need the basis matrix with respect to
      _bmat = mul!(Kl, Kl, L.basis)
      LL = lattice(L.V, L.base_ring, _bmat, check = false)
      if any(LLL -> LLL.basis == LL.basis, res)
        continue
      end
      if filter === nothing ||
         (filter == :local_isomorphism && all(LLL -> !is_locally_isomorphic(LLL, LL, fmpz(p)), res))
        push!(res, LL)
      end
    end
  end
  return res
end

################################################################################
#
#  Centering algorithm
#
################################################################################

function sublattice_classes(L::ModAlgAssLat, p::Int)
  res = typeof(L)[L]
  to_check = typeof(L)[L]
  while !isempty(to_check)
    M = pop!(to_check)
    X = pmaximal_sublattices(M, p, filter = :local_isomorphism)
    for N in X
      if any(LLL -> is_locally_isomorphic(LLL, N, fmpz(p))[1], res)
        continue
      else
        push!(res, N)
        push!(to_check, N)
      end
    end
    #@show length(res)
  end
  return res
end

function is_sublattice(L::ModAlgAssLat, M::ModAlgAssLat)
  return isone(denominator(basis_matrix(L) * basis_matrix_inverse(M)))
end

function _issublattice(L::ModAlgAssLat, M::ModAlgAssLat, tmp)
  tmp = mul!(tmp, basis_matrix(L), basis_matrix_inverse(M))
  return isone(denominator(tmp))
end

function sublattices(L::ModAlgAssLat, p::Int, level = inf)
  res = typeof(L)[L]
  to_check = typeof(L)[L]
  pL = p*L
  temp = typeof(L)[]
  i = 0
  F = GF(p)
  comp_fac = composition_factors(change_coefficient_ring(F, L))
  tmp_ = zero_matrix(QQ, rank(L), rank(L))
  while !isempty(to_check)
    if i >= level
      break
    end
    i += 1
    empty!(temp)

    for M in to_check
      pmax = pmaximal_sublattices(M, p, composition_factors = comp_fac)
      for N in pmax
        if _issublattice(N, pL, tmp_)
          continue
        end
        if any(LLL -> LLL.basis == N.basis, temp) || any(LLL -> LLL.basis == N.basis, res)
          continue
        else
          push!(temp, N)
        end
      end
    end
    empty!(to_check)
    append!(to_check, temp)
    append!(res, temp)
  end
  return res
end
