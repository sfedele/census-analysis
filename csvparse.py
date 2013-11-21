""" Script that takes a CSV and shows basic summary information
useful as a prelude to more specific analysis.
"""

import sys
import csv
from collections import defaultdict
import re
import datetime


import mr


import pdb

def _hist(vals):
  # question of whether you want a hashmap lookup, sorted by key,
  # sorted by count, or whatever.  let's go with freq sorting:
  return mr.mapreduce(lambda x: (x, 1), lambda k, vs: sum(vs), vals)
  
def hist(vals):
  return sorted(_hist(vals), key=lambda x: x[1], reverse=True)
  
def dhist(vals):
  return dict(_hist(vals))

identity = lambda x: x

# kinda a weak date regexp string, but 
DATE_RX_STR = r'([0-9][0-9](?:[0-9][0-9])?)[-/]?' * 3 + '$'


class PreludeAnalysis(object):
  def __init__(self, rows, field_info):
    self.rows = rows
    self.field_info = field_info
    
  @classmethod
  def analyze(self, stream):
    def mapperx(row):
      for key, val in row.iteritems():
        # val is a string still
        m = re.match(r'([-+]?[0-9]+)(\.[0-9]*)?$', val)
        if m:
          # numerical, yield int or float
          if m.group(2):
            yield key, ('float', val)
          else:
            yield key, ('int', val)
        else:
          # val may be a number with $ and , and the like.
          m = re.match(r'\$?((?:[0-9]+,?)+)(\.[0-9]*)?$', val)
          if m:
            if m.group(1):
              yield key, ('pretty_float', val)
            else:
              yield key, ('pretty_int', val)
          else:
            # val may be a date
            m = re.match(DATE_RX_STR, val)
            # needs to have two 2s and a 4 len, 4 len is year... determine the other two
            # later, in reducer; sigh
            if m and sorted(map(len, m.groups()), reverse=True) in ([4, 2, 2], [2, 2, 2]):
              # going to redo this regexp in reducer too
              yield key, ('date', val)
            else:
              # TODO: val *could* be a timestamp of somesort...
              # alright, val is probably just some string
              yield key, ('string', val)

    def reducerx(key, vals):
      # keys are attribute names here
      counts = defaultdict(int)
      tag_counts = defaultdict(int)
      for tag, val in vals:
        tag_counts[tag] += 1
        counts[val] += 1
      yield FieldInfo(counts, tag_counts)

    # TODO: handle csvs without header rows.
    rows = list(csv.DictReader(stream))
    field_info = dict(mr.mapreducex(mapperx, reducerx, rows))
    return PreludeAnalysis(rows, field_info)

  def build_parser(self):
    parser = dict((key, field_info.build_parser()) for key, field_info in self.field_info.iteritems())
    return lambda r: dict((k, parser[k](v)) for k, v in r.iteritems())
    
  def parse_iter(self, stream):
    f = csv.DictReader(stream)
    first_row = f.next()
    assert sorted(first_row) == sorted(self.field_info), "Can't parse csv with different column names"
    parser = self.build_parser()
    yield parser(first_row)
    for r in f:
      yield parser(r)
      
  def printout(self):
    raise NotImplemented


class FieldInfo(object):
  def __init__(self, counts, tag_counts):
    self.counts = counts
    self.tag_counts = tag_counts
    
  def build_parser(self):
    if len(self.tag_counts) == 1:
      # well, this is easy:
      if 'int' in self.tag_counts:
        return int
      elif 'float' in self.tag_counts:
        return float
      elif 'pretty_int' in self.tag_counts:
        return pretty_int_parser
      elif 'pretty_float' in self.tag_counts:
        return pretty_float_parser
      elif 'string' in self.tag_counts:
        return identity
      elif 'date' in self.tag_counts:
        # ok, this is less easy and probably error prone;  try to
        # guess semi-intelligently what the date format is
        fmts = determine_date_formats(self.counts)
        assert len(fmts) == 1
        fmt = fmts[0]
        return lambda x: datetime.datetime.strptime(x, fmt)
    elif sorted(self.tag_counts) == ['float', 'int']:
      # promote all ints to floats on existence of a single float lol
      return float
    elif 'string' in self.tag_counts:
      return identity
    raise NotImplementedError
      
      
def pretty_int_parser(x):
  return int(x.strip('$').replace(',', ''))

def pretty_float_parser(x):
  return float(x.strip('$').replace(',', ''))


class NotADateException(Exception):
  pass

def determine_date_formats(counts):
  # assumes that counts parse via the following regexp and that
  # all the groupings are of length 2, with max 1 of length 4
  tmp = [re.match(DATE_RX_STR, s).groups() for s in counts]
  assert list(set(map(len, tmp))) == [3]
  
  # anything that isn't an integer in these must be the date separator
  separators = set(c for s in counts for c in s) - set("0123456789")
  if len(separators) > 1:
    raise NotADateException("Multiple date separators: %s" % list(separators))
  elif separators:
    separator = list(separators)[0]
  else:
    separator = ''
  
  # each column should be the same number of characters
  # only 1 can have 4 digits, and that is unambiguously the year
  year_col = None
  month_day = []
  for n in range(3):
    lens = list(set(len(x[n]) for x in tmp))
    if lens == [4]:
      if year_col is not None:
        raise NotADateException("Multiple columns have 4 digit date: %d & %d" % (year_col, n))
      year_col = n
    elif lens == [2]:
      month_day.append(n)
    else:
      raise NotADateException("Column %d has lengths: %d" % (n, lens[:10]))
  fmts = [None, None, None]
  if year_col is not None:
    # wooo got one!  now for month & day
    fmts[year_col] = "%Y"
    uniqs = [set(int(t[n]) for t in tmp) for n in month_day]
    above_12 = [n for n, u in zip(month_day, uniqs) if max(u) > 12]
    if above_12:
      if len(above_12) > 1:
        raise NotADateException("Month Day candidate columns %d both have values > 12" % month_day)
      fmts[above_12[0]] = "%d"
      # the one remaining column left is the month
      # and we're done
      return [separator.join(f or '%m' for f in fmts)]
    else:
      # neither goes above 12... fmt is ambiguous.
      # if year_col is the first one, lets say its YYYYMMDD
      # if year_col is the last one, its probably MMDDYYYY
      # if its the middle one, we notice that removing YYYY
      # in the above answers leaves MMDD, so lets just always
      # say that its MMDD for now
      i = iter(('%m', '%d'))
      return [separator.join(f if f else i.next() for f in fmts)]
  else:
    # Bah, they all have 2 digits.... lets try some things
    # yymmdd or mmddyy, which seem most natural to me
    # First, are any above 12?  they could be either years or days,
    # not months though.  if this occurs in one of the last two columns,
    # have our answer with these defaults:
    uniqs = [set(int(t[n]) for t in tmp) for n in range(3)]
    above_12 = [n for n in range(3) if max(uniqs[n]) > 12]
    if above_12:
      if above_12[0] == 0:
        # the first column went above 12 so we know it can't be month
        # thus its yymmdd by the discussion above
        return [separator.join(('%y', '%m', '%d'))]
      else:
        # first column didn't go above 12, one of the latter ones did
        # doesn't matter which, we're going to say its mmddyy
        return [separator.join(('%m', '%d', '%y'))]
    else:
      pass
    
  raise NotImplementedError


def iparse(csv_path):
  stream = lambda: open(csv_path)
  result = PreludeAnalysis.analyze(stream())
  return result.parse_iter(stream())


def main():
  if len(sys.argv) != 2:
    sys.exit('Usage: %s <path-to-csv>' % self.argv[0])
  result = PreludeAnalysis.analyze(sys.argv[1])
  print "not implemented"


if __name__ == '__main__':
  main()
    