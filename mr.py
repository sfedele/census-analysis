""" mr.py - tools for map-reduces.

Exercises & tests at bottom.
"""


from itertools import groupby, chain, imap
from operator import itemgetter

def mapreduce(mapper, reducer, records):
  """Basic single-threaded, in memory mapreduce algorithm.

        mapper  : record -> (key, value)
        reducer : (key, [value]) -> result
        records : [record]
     
     Returns a list of [result].
     
     Note that this is *not* how Hive defines a map/reduce.
     See `mapreducex` below for more general version.
  """
  # Naturally, there's better ways to do this.
  return [(k, reducer(k, list(vs))) for k, vs in kgroupby(map(mapper, records))]
  
  
def mapreducex(mapper, reducer, records):
  """Similar to `mapreduce` except mappers and reducers can have
     multiple or no outputs.
  
        mapper  : record -> [(key, value)]
        reducer : (key, [value]) -> [result]
        records : [record]
        
    This is how Hive/streaming work.
  """
  return [(k, v) for k, vs in kgroupby(flatten1(map(mapper, records))) for v in reducer(k, list(vs))]
  
  
def kgroupby(xs):
  """Group tuples of xs by their key, or first element.  Returns an iterable of (key, [value]) pairs."""
  return ((k, imap(itemgetter(1), ys)) for k, ys in groupby(sorted(xs, key=itemgetter(0)), itemgetter(0)))


def flatten1(xs):
  """Flatten the iterable xs by on depth, so e.g. [[A]] -> [A]"""
  return chain.from_iterable(xs)
  























## Since you're still reading,  
## Exercises/questions:
##  1) Name all the ways that the output of mapreduce & mapreducex differ
##       - `mapreducex`'s output may not be sorted by key; `mapreduce`'s will be due to the groupby/sort
##       - `mapreducex` may output the same key multiple times; `mapreduce`'s output keys are unique
##  2) Is the list(vs) really necessary?  What could happen if it were removed?
##       - if the reducer function relies on random access of vals, or iterates over it twice, it will 
##         raise an Exception.
##  3) What is the algorithmic difference if the `sorted` function in kgroupby 
##     were called without the key=itemgetter(0) argument?  Write a unittest that
##     captures this difference.
##       - without the key, sorted will sort on both the key *and* the value
##         unnecessary comparisons, and changes the order of inputs to reducer.
##  4) Analyze the run time characteristics of mapreduce, as well as the maximum memory
##     used.  What are the worst cases?  How might either of these be improved?
##       - map(mapper, records) loads the entire set of (key, values) into memory - O(N) ops if mapper is O(1)
##       - sorted(*) also loads entire set into memory, takes O(N*Log(N)) ops
##       - groupby is an iterable, loading only the current record into memory - O(N) ops to iterate over
##       - imap(itemgetter(1), ys) returns an iterable also, and is O(1)
##       - the list(vs) iterates over the imap(itemgetter(1)) records, loading each list into memory taking 
##         O(N) ops.
##       - thus, maximum memory usage is upto 2 * number of records (once to load the data set & sort it, 
##         then again if every record has either a unique key or the same key)
##       - otherwise, expected memory usage is (1 + r) * #records where r is the expected ratio of output 
##         records to input records
##       - worst case runtime is O(N) + O(N*Log(N)) ~ O(N*Log(N)) assuming Python's `sorted` is implemented properly.
##       - the factor of Log(N) could be removed by aggregating the mapper output in an intermediate hash - 
##         requires upto 2x extra memory for hash table (depends on number of unique keys), saves having to 
##         do the sort
##       - memory usage could be improved by changing map to imap, doing an on-disk sort, and removing the 
##         list(vs) such that the reducer accepts an iterable instead.
##     key hoped for takeaway: performance of mapreduce is dependent on the exact key distribution of the records
##     also, that mapreduce isn't scary and is basically a python one-liner
##  5) Create data sets that cause `mapreduce` to fail via 
##        A) running out of memory, 
##        B) taking longer than 2 minutes to execute.
##  6) What is the difference between `mapreduce` and `mapreducex`?  Would a mapper for `mapreduce` also work 
##     when passed to `mapreducex`?  What would happen if it were?
##       - it would not work; mapreduce's mapper/reducer return single objects, `mapreducex`'s return lists/iterables
##       - the (key, value) output of `mapreduce`'s mapper would be treated as an iterable in `mapreducex`,
##         flattened, and then keys/values would be mixed together in the resulting list; behavior at this point 
##         depends on the specific outputs involved.
##  7) Describe a situation where `mapreducex` is more appropriate than `mapreduce`.
##       - filtering inputs: one can skip a record by not returning anything
##       - expanding/exploding inputs; outputting multiple values for each record (e.g. word counting where records
##         are documents)
##  8) Write a mapper/reducer pair along with associated records whose `mapreducex` output changes depending on 
##     whether kgroupby's sort is stable (if you answered #3 this should be easy).  Is this type of behavior 
##     desirable?  Explain why or why not.
##


# And finally, some unittests, because unittests.

import unittest


class MapReduceTest(unittest.TestCase):
  def testmapreduce(self):
    uptofive = [1, 2, 3, 4, 5]
    xs = uptofive * 20
    def mapper(x):
      return (x, 3)
    def reducer(k, vals):
      return sum(vals)
    self.assertEquals([(n, 20 * 3) for n in uptofive], mapreduce(mapper, reducer, xs))
    self.assertEquals([], mapreduce(mapper, reducer, []))
    self.assertEquals([(1, 3)], mapreduce(mapper, reducer, [1]))

  def testmapreducex(self):
    uptofive = [1, 2, 3, 4, 5]
    xs = uptofive * 20
    def mapperx(x):
      for i in range(x):
        yield (x, i)
    def reducerx(k, vals):
      return [sum(vals)]   
    self.assertEquals([(n, 20 * n * (n - 1) / 2) for n in uptofive], mapreducex(mapperx, reducerx, xs))
    def reducerx2(k, vals):
      if k != 3:
        yield sum(vals)
    self.assertEquals([(n, 20 * n * (n - 1) / 2) for n in uptofive if n != 3], mapreducex(mapperx, reducerx2, xs))
    
  def test_kgroupby_sort_is_stable(self):
    uptofive = [1, 2, 3, 4, 5]
    xs = uptofive * 20
    def mapper(x):
      return (x % 2, (5 - x) / 2)
    def reducer(k, vals):
      return vals[0]
    mr.mapreduce(mapper, reducer, xs)


if __name__ == '__main__':
  unittest.main()
