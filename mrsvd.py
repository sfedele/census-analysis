"""Example of implementing an SVD via Map Reduce.

"""
from itertools import count, chain, product
from collections import defaultdict
from random import uniform
import copy

from analyze import loaddata
import mr


def identity_mapper((key, val)):
  return (key, val)


def mrsvd(ind1, ind2, keyvals, K=2):
  # Perform an SVD on the list of keyvals,
  # using ind1 and ind2 into the keys as the
  # independent variables.  Decompose into
  # K singular values.
  
  # Initial the coeffs vector with random values:
  def coeffs_initialize_mapperx((key, val)):
    return chain(( ((0, k, key[ind1]), None) for k in range(K) ),
                 ( ((1, k, key[ind2]), None) for k in range(K) ))
  def coeffs_initialize_reducerx(key, vals):
    return [uniform(-1, 1)]

  # maybe:
  # coeffs_vector :: [((CoeffsType1, SingularVectorCoeff, IndexType1), Value)]
  initial_coeffs_vector = mr.mapreducex(coeffs_initialize_mapperx, coeffs_initialize_reducerx, keyvals)
  
  coeffs_vector = copy.copy(initial_coeffs_vector)
  stepsize = 0.05
  
  # Join the coeffs vectors with the rows that need them:
  # We want in the reducer, for each row
  # This is easiest using a HashMap in Hive,etc, though for 
  # the sake of pedantricity we'll do it as a full map/reduce.
  
  # ??Would really like to avoid doing the full outter product
  #   even if this is pedantic.  I mean yeah you can do like:
  def join_mapperx(((key, val), ((c, k, i_or_j), coeff))):
    if key[ind1] == i_or_j or key[ind2] == i_or_j:
      yield  key, (c, k, i_or_j, coeff, val)
  # And then some crazy reducer that has to do like shit like this:
  def join_reducerx(key, valsx):
    vals = {}
    coeffs = [({}, {}) for k in range(K)]
    for c, k, i_or_j, coeff, val in valsx:
      vals.setdefault(val, vals.get(val, 0) + 1)
      coeffs[k][c][i_or_j] = coeff
    # Tho now how do you differentiate the unique vals?
    # Like, we could count about how many of each rows
    # we expect 2*K coeffs to match for each input record:
    i = key[ind1]
    j = key[ind2]
    for val, count in vals.iteritems():
      assert count % (2 * K) == 0
      predicted = sum(c1[i] * c2[j] for c1, c2 in coeffs)
      err = val - predicted
      for n in range(count / (2 * K)):
        yield err
  mapreduce.mapreducex(join_mapperx, join_reducerx, product(keyvals, coeffs_vector))
  # And so on and so forth.

  # This whole outer product thing is kinda lame though, plus this method requires
  # Building a whole dictionary for each record computed - so even if we solve the
  # horrible T = #keyvals, N = #(ind1), M = #(ind2) ::: T * (K*N + K*M) outer product,
  # we still have another 2K * T operations afterwards in the reducer.
  
  # So this is like T * Sqrt(T) at best, T^2 at worst
  
  
  # Now, consider the following update step.
  # The main difference is that we've created a dictionary for the coeffs:
  
  coeffs = [({}, {}) for k in range(K)]
  for (c, k, i_or_j), x in initial_coeffs_vector:
    coeffs[k][c][i_or_j] = x
  
  # Step one: compute the derivative:
  def neg_gradient_step_mapperx((key, val)):
    # Use current coefficients to compute error:
    i = key[ind1]
    j = key[ind2]
    predicted = sum(c1[i] * c2[j] for c1, c2 in coeffs)
    err = val - predicted
    for k, (c1, c2) in enumerate(coeffs):
      # Doing gradient _descent_ (minimization), so we move
      # _opposite_ the derivative, hence no minus:
      yield (0, k, i), 2 * stepsize * err * c2[j]
      yield (1, k, j), 2 * stepsize * err * c1[i]
      
  # We thus have T * K operations assuming dict lookup is O(1)
  sum_reducer = lambda (key, values): [sum(values)]
  # And a sum reducer over 2*K keys, not that that matters, summing up all T*K outputs.
  # The net result is T * K steps, compared to T * K * Sqrt(T) above.
  # Consider running on 1M = 1e6 datapoints.  If each iteration over T * K points takes 1s,
  # the factor of Sqrt(T) adds an extra 1e3s = 15 minutes.
  step_vector = mapreduce.mapreducex(neg_gradient_step_mapperx, sum_reducer, keyvals)

  # Step two: update the parameters:
  # The following takes 2 * K * (N + M) * 2 steps, so like K * Sqrt(T):
  new_coeffs = mapreduce.mapreduce(identity_mapper, sum_reducer, chain(step_vector, coeffs_vector))
  # Then if we loop this, initializing the coeffs data structure requires another 2 * K * (N + M) steps
  # Keeping this all roughly K * (T + Sqrt(T)) ~ K * T.
  
  # Of course, N & M will depend heavily on the specifics of the data involved.
  # It could be that N & M are both really small compared to T, so all that 
  # Sqrt(T) crap above doesn't really mean anything, and the first algorithm 
  # will approach the second asymptotically.  The second dominates the first,
  # however.
  
  # The limitation of the second approach is that it requires a common dictionary
  # available in every mapper.  Up to a certain point this is possible e.g. the
  # dictionary representation needs to fit in memory in each mapper.  Even if this
  # is possible, this amount of data must be serialized to each mapper as well,
  # which could lead to some absurd amounts of data transferred per iteration.
  
  # A second approach is to consider an extension to mapreduce as defined above.
  # Up until now, we've only considered mapreduces with a single mapper & single
  # reducer.  In the join_reducerx, we had to go through some serious contortions
  # to get all the data where we wanted.  What if, however, we consider MapsReduce?
  # ??Like, to join data, we allow multiple streams, each with their own mappers,
  # and then in the reducer the stream of keys is *already* guaranteed to be sorted,
  # and grouped by the streams (so reducerx gets multiple (keys, vals) streams)
  # We could then do like a mergesort whatever over the streams of keys, yielding
  # rows of data whenever the keys match (and thus achieving left, right & outer
  # joins through differing policies of key matching per stream).
  
  # Subsequent maps on the resulting data may be able to be done in place (e.g.
  # in this reducer).
  
  # SIgh, this is still heavily dependent on N & M.  Like, implementing that as
  # a join without hashmaps... well, worst case is that N = M = T (e.g. both
  # are just permutations on the dataset or whatever).  First of, using this
  # whole mergesort thing, we'd need to iterate over the second index a lot,
  # or otherwise since the rows are combinations of i & j, well, its still tough.
  
  # NOw, imagine that you have N + M > machine memory.  You might achieve
  # this by sharding both the coeffs & the records on ind1, ind2.  You
  # would pick a sharding such that the set of coeffs in each shard *does*
  # fit into memory.
  
  # The mappers shard; the reducer would slurp up the entire coeffs streams first,
  # build a hashmap, and then iterate over the rows for each.  At this point,
  # the main decision is whether to use a sorted list or hashmap.  Both would
  # be fun, with advantages and drawbacks each.
  
  # Unfortunately, this isn't exactly going according to the multi-stream mergesort
  # as described above.  Moreover, its not entirely clear how to "shard on i & j"
  # for the coeffs - they have only 1 i or j, not both.  Essentially, you would
  # need a hash function such that h(i,j) = h(i,nil) V j && h(i,j) = h(nil,i) V i
  
  # Alternatively, imagine N = [A1, A2, A3, A4] & M = [B1, B2, B3]
  # we would shard these coeffs as follows:
  #
  #  A1,B1 A1,B2 A1,B3
  #  A2,B1 A2,B2 A2,B3
  #  A3,B1 A3,B2 A3,B3
  #  A4,B1 A4,B2 A4,B3
  #
  # And then all rows would get sent to (i%4, j%3) (if you can even CONSTRUCT
  # a mapping that takes the raw string labels to an appropriate shard... ok
  # fine thats easy).  It still requires duplicating each A 3x & B 4x, so you
  # get ~#shards*(N + M) extra; there's not an easy primative for this in Hive,
  # though you could emulate it.  Essentially this is the principle of what
  # must be done, however.  So like if N = M = T, well, that's lame - why are
  # you doing an SVD on that anyway?
  
  # Roughly then in this construction, the initial mapper iterates over
  # let m = #M shards, n = #N shards
  # The we do m*N+n*M steps in distributing the coeffs to n*m shards.
  # We do T steps distributing the records.
  # We do another N/n+M/m steps reading in the coeffs per reducer,
  # plus T*(4*K) steps predicting & outputing coeff updates assuming
  # ammortized constant costs looking up the coeffs.
  
  # We're then into the final reducer above, showing that this approach
  # achieves asymptotically the T*K performance above when N,M & T are
  # large.  The disadvantage is the N/n+M/m steps *per reducer* reading
  # in the coeffs; remember that this is roughly the size of what will
  # fit into memory.  Hopefully T/(n*m) is still large enough that this
  # is worth it.  Heavy fragmentation across keys in either N or M will
  # also ruin this approach.
  
  # SO yeah, so much for pedantic.  Anyway, scaling this & industrializing
  # the problem is, well, a fun problem itself.  Ultimately, doing it
  # correctly requires an understanding of the underlying key distribution,
  # so running an initial probe mapreduce to determine the key profiles
  # & constructing an appropriate sharding mechanism (e.g. add a random
  # key column if there's heavy key skew, although this either equires
  # more reducers or decreases n & m).
  
  # One nice thing about this approach, finally, is that once the data is
  # distributed to each node once, only derivative steps need be computed
  # again.  Its probably possible to create an efficient cluster-based
  # approach to this SVD technique scaling to very very large values of
  # T, N, M, where each step is nearly embarrassingly parallel.  The
  # cluster would of course extend naturally into a parallelizable online
  # implementation of the prediction step as well (n*m nodes behind a proxy
  # that fans out based on the i, j of the incoming query).
  
  # Naturally, this construct 'begs the question' of whether learning
  # itself could be done entirely online at this point; other than perhaps
  # overloading the proxy(s), the main issue is syncing each node's updates
  # with the n + m other nodes that shares values with them.  This is, to
  # my knowledge, an outstanding avenue of research to consider.
  
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  # Step 1.5: Join the derivative 
  coeffs = 
  dict(mapreduce.mapreduce)

  # Step two: update the parameters:
  def descent_update_reducerx(key, pointwise_derivatives):
    sum
    # Doing gradient _descent_ (minimization), so we move
    # _opposite_ the derivative, hence the minus-equals:
    cx[n][k][j] -= stepsize * sum(pointwise_derivatives)
    # We're updating the components in place; this isn't very
    # mapreducy, or scalable.  We'll revisit immutability of
    # mapreduces and how to factor this out later.  For now,
    # the output of this mapreduce is nil and we rely on side-
    # -effects.
    return []
    
  # We'd also like a mapreduce that, given coeff pairs,
  # returns the sum square error of using them:
  
  def sumsqrerrs(coeff_pairs):
    def sumsqrerr_mapperx((key, val)):
      for n, (coeffs1, coeffs2) in enumerate(coeff_pairs):
        predicted = sum(c1[i] * c2[j] for c1, c2 in zip(coeffs1, coeffs2))
        err = val - predicted
        yield n, err * err
    def sumsqrerr_reducerx(index, errs):
      return [sum(errs)]
    # Note: here we're relying on mapreducex to return its result sorted by key.
    return [e for n, e in mr.mapreducex(sumsqrerr_mapperx, sumsqrerr_reducerx, keyvals)]
  
  mr.mapreducex(derivative_mapperx, descent_update_reducerx, keyvals)
    
  

def main():
  wdata, rdata, job_names = loaddata()
  
  
if __name__ == '__main__':
  main()

