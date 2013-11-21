"""Analyze some census data.

Some helper functions and a lot of mapreduces.
"""

import glob
import csv
import re
from itertools import islice, chain
from operator import itemgetter
from collections import defaultdict

import mr

import pdb


def sumf(row, *inds):
  """Parse the given indices of row as floats & return their sum, ignoring empty values."""
  return sum(float(row[n]) if row[n] else 0 for n in inds)


def munge_mapperx_factory(field_offset):
  """Return a mapper function that reads fields from a specified offset."""
  def mapperx(row):
    o = field_offset
    # Some gross simplifications to be better considered:
    # state = one of 54 different values
    # occupation = one of ~489 values in the data
    # sex = male | female
    # race = latino | white | black | indian | asian | other
    # Relevant column numbers were obtained manually from data + headers.
    state = row[2]
    occupation = row[3]
    
    yield (state, occupation, 'male', 'latino'), sumf(row, 38 + o, 39 + o)
    yield (state, occupation, 'male', 'white'), sumf(row, 40 + o)
    yield (state, occupation, 'male', 'black'), sumf(row, 41 + o)
    yield (state, occupation, 'male', 'native'), sumf(row, 42 + o, 44 + o)
    yield (state, occupation, 'male', 'asian'), sumf(row, 43 + o)
    yield (state, occupation, 'male', 'multi'), sumf(row, *xrange(45 + o, 52 + o))
    yield (state, occupation, 'female', 'latino'), sumf(row, 70 + o, 71 + o)
    yield (state, occupation, 'female', 'white'), sumf(row, 72 + o)
    yield (state, occupation, 'female', 'black'), sumf(row, 73 + o)
    yield (state, occupation, 'female', 'native'), sumf(row, 74 + o, 76 + o)
    yield (state, occupation, 'female', 'asian'), sumf(row, 75 + o)
    yield (state, occupation, 'female', 'multi'), sumf(row, *xrange(77 + o, 84 + o))

  return mapperx


def munge_reducerx(key, vals):
  assert len(vals) == 1, "Each key combination should only appear once: %s appears multiple: %s" % (key, vals)
  return [vals[0]]


def chaincsvs(paths, offset=2):
  """Given a list of paths, return an iterable over all records from offset onward."""
  return chain.from_iterable(islice(csv.reader(open(path)), 2, None) for path in paths)


def loaddata():
  """Load some census data.  Returns a 3-tuple of
  
         worksite_data, residence_data, job_names
     
     Where worksite_data & residence_data are lists of key, value pairs
     and job_names is a dictionary mapping occupation codes to readable
     job names.
  """
  import sys, time
  print >> sys.stderr, "loading wdata..."
  start_time = time.time()
  wpaths = glob.glob("rawdata/wpart*/*EEO_10_5YR_EEOALL1W.csv")
  wdata = mr.mapreducex(munge_mapperx_factory(2), munge_reducerx, chaincsvs(wpaths))
  print >> sys.stderr, "done, time taken: %f" % (time.time() - start_time)
  
  print >> sys.stderr, "loading rdata..."
  start_time = time.time()
  rpaths = glob.glob("rawdata/rpart*/*EEO_10_5YR_EEOALL1R.csv")
  rdata = mr.mapreducex(munge_mapperx_factory(0), munge_reducerx, chaincsvs(rpaths))
  print >> sys.stderr, "done, time taken: %f" % (time.time() - start_time)

  # While we're at it, let's create a mapping from occupation code
  # to descriptive label for use later on.
  all_records = chain(chaincsvs(wpaths), chaincsvs(rpaths))
  mapping = mr.mapreduce(lambda r: (r[3], r[4]), lambda k, vs: set(vs), all_records)
  assert all(len(v) == 1 for k, v in mapping), "Should only be one description per code."
  job_names = {}
  for occupation_code, label_set in mapping:
    label = list(label_set)[0]
    x = re.split(r' \d\d\d\d ', label)[0]
    CHAR_CUTOFF = 100
    if len(x) > CHAR_CUTOFF:
      # simple ellipses if label is too long
      x = x[:CHAR_CUTOFF-3] + "..."
    job_names[occupation_code] = x

  return wdata, rdata, job_names
  
  
def joindata(wdata, rdata):
  """Joins worksite and residence data sets on their keys, filling in missing
  values with 0's.  Essentially an outer join filling NULLS with 0s."""

  # Prepend records' value with the given label:
  def tag(label, records):
    return ((key, (label, val)) for key, val in records)

  # Chain the two data sets together, tagging each differently:
  records2 = chain(tag('work', wdata), tag('resi', rdata))

  # Identity mapper:
  def mapper((key, value)):
    return key, value

  # Each key appears at most twice: once in the set tagged with 'work',
  # once in the set tagged with 'resi'.  We use dict.get's default argument
  # to pull out the appropriate value or a 0 if it didn't join.
  def reducer(key, values):
    assert len(values) in (1, 2), "Unexpected number of values in join."
    # Each value is a 2-tuple (label, number) for easy dict building:
    xs = dict(values)
    return xs.get('work', 0), xs.get('resi', 0)

  return mr.mapreduce(mapper, reducer, records2)


def compare_totals(records):
  """For each (state, sex, race) tuple, sum up counts across all occupations.
  Extract the special '0000' 'Total' occupation specially for each tuple, out-
  -putting the total and sum per key.  The expectation is that these numbers
  should add up to be the same.
  """
  def mapper((key, val)):
    return (key[0], key[2], key[3]), ('total' if key[1] == '0000' else 'counted', val)
  def reducer(key, values):
    running_count = total = 0
    for tag, count in values:
      if tag == 'total':
        assert total == 0, "Total should only appear once per key in data set."
        total = count
      else:
        running_count += count
    return total, running_count
  return mr.mapreduce(mapper, reducer, records)
  
  
def perc_diff_per_column(joined_records, colnum):
  """Find the percentage difference between resident & worksite jobs
  in the joined data, aggregated over the given key column number.
  key is the unique value of the column, value is the % difference.
  """
  def mapperx((key, val)):
    if key[1] != '0000': 
      yield key[colnum], (val[0], val[1], val[1] - val[0])
  def reducerx(key, values):
    worksite = sum(map(itemgetter(0), values))
    resident = sum(map(itemgetter(1), values))
    difference = sum(map(itemgetter(2), values))
    assert difference == resident - worksite
    return [1.0 * difference / (worksite + 1e-5)]
  return mr.mapreducex(mapperx, reducerx, joined_records)


def sex_differences_by_occupation(data):
  """Find which occupation codes have the largest gender skews
  in either direction.  To do this, we compute P(sex | occupation)
  for each occupation, returning key = occupation code, values =
  (male_prob, female_prob)..
  """
  def mapperx((key, val)):
    # Key by occupation, output (sex, count).
    yield key[1], (key[2], val)
  def reducerx(key, values):
    total = sum(count for sex, count in values)
    probs = defaultdict(int)
    for sex, count in values:
      probs[sex] += count / total
    yield probs.get('male', 0), probs.get('female', 0)
  return mr.mapreducex(mapperx, reducerx, data)


def state_occupation_rows(joined_data, job_names):
  """For each state, generate a vector where each component is
  the count of that occupation; the columns are the keys in
  job_names, sorted by occupation code, and are the same for
  every output.  key is occupation code, value is the vector
  as a Python list.  All values have the same list length.
  """
  all_codes = sorted(job_names)
  def mapperx((key, val)):
    if key[1] != '0000':
      yield key[0], (key[1], val[0])
  def reducerx(key, values):
    xs = defaultdict(int)
    for occupation_code, count in values:
      xs[occupation_code] += count
    yield [xs.get(k, 0) for k in all_codes]
  return mr.mapreducex(mapperx, reducerx, joined_data)


def main():
  # w = worksite data, r = residence data
  wdata, rdata, job_names = loaddata()
  joined_data = joindata(wdata, rdata)
  print "data loaded, pdb started:"
  pdb.set_trace()
  
  
if __name__ == '__main__':
  main()