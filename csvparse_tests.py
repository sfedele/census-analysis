import csvparse as pa
import unittest

from StringIO import StringIO
import datetime

import pdb


class PreludeAnalysisTests(unittest.TestCase):
  def do_analyze(self, data):
    f = StringIO(data)
    return pa.PreludeAnalysis.analyze(f)
    
  def do_parse_iter(self, data):
    return self.do_analyze(data).parse_iter(StringIO(data))
  
  def testBasicIntegerTagging(self):
    data = '\n'.join(['x,y,z', '1,2,3', '4,5,6', '1,7,6'])
    result = self.do_analyze(data)
    self.assertEquals(['x', 'y', 'z'], sorted(result.field_info.keys()))
    self.assertEquals(['int'], list(result.field_info['x'].tag_counts))
    self.assertEquals(['int'], list(result.field_info['y'].tag_counts))
    self.assertEquals(['int'], list(result.field_info['z'].tag_counts))
    
  def testMixedTagging(self):
    data = '\n'.join(['x,y,z,dt', 'aaa,2,3.14,2013-07-01', 'bbb,5,6,2013-08-01', 'ccc,7,6.,2013-09-01'])
    result = self.do_analyze(data)
    self.assertEquals(['dt', 'x', 'y', 'z'], sorted(result.field_info.keys()))
    self.assertEquals(['string'], list(result.field_info['x'].tag_counts))
    self.assertEquals(['int'], list(result.field_info['y'].tag_counts))
    self.assertEquals(['float', 'int'], sorted(result.field_info['z'].tag_counts))
    self.assertEquals(['date'], list(result.field_info['dt'].tag_counts))
    
  def testBasicParsing(self):
    data = '\n'.join(['x,y,z', '1,2,3', '4,5,6', '1,7,6'])
    f = self.do_parse_iter(data)
    self.assertEquals(f.next(), dict(x=1, y=2, z=3))
    self.assertEquals(f.next(), dict(x=4, y=5, z=6))
    self.assertEquals(f.next(), dict(x=1, y=7, z=6))
    
  def testMixedParsing(self):
    data = '\n'.join(['x,y,z,dt', 'aaa,2,3.14,2013-07-01', 'bbb,5,6,2013-08-01', 'ccc,7,6.,2013-09-01'])
    f = self.do_parse_iter(data)
    self.assertEquals(f.next(), dict(x='aaa', y=2, z=3.14, dt=datetime.datetime(2013,7,1)))
    self.assertEquals(f.next(), dict(x='bbb', y=5, z=6.00, dt=datetime.datetime(2013,8,1)))
    self.assertEquals(f.next(), dict(x='ccc', y=7, z=6.00, dt=datetime.datetime(2013,9,1)))
    
  def testPrettyParsing(self):
    data = '\n'.join(['x,y,z,w', '"$1,234.00",$40,$40.,"1,234,567,890"'])
    f = self.do_parse_iter(data)
    self.assertEquals(f.next(), dict(x=1234.00, y=40, z=40.00, w=1234567890))

  def testStringColumnWithASingleIntIsString(self):
    data = '\n'.join(['x'] + ['apple', 'banana', 'pear', 'kiwi'] * 10 + ['2', 'foo', 'bar', 'baz'])
    f = self.do_parse_iter(data)
    self.assert_(all(isinstance(r['x'], str) for r in f))
     

class DetermineDateFormatTests(unittest.TestCase):
  def testDefaultMonthDay(self):
    # Neither month nor day go above 12 here
    dates = pa.dhist(['2013/01/01', '2013/01/02', '2013/01/03'])
    ret = pa.determine_date_formats(dates)
    self.assertEquals(['%Y/%m/%d'], ret)
    dates2 = [d.replace('/', '-') for d in dates]
    self.assertEquals(['%Y-%m-%d'], pa.determine_date_formats(dates2))
    dates3 = [d.replace('/', '') for d in dates]
    self.assertEquals(['%Y%m%d'], pa.determine_date_formats(dates3))

  def testDayGoesAbove12(self):
    # THe non standard one goes above 12 - it should be day
    dates = pa.dhist(['2013/01/01', '2013/01/02', '2013/31/03'])
    self.assertEquals(['%Y/%d/%m'], pa.determine_date_formats(dates))
    dates2 = [d.replace('/', '-') for d in dates]
    self.assertEquals(['%Y-%d-%m'], pa.determine_date_formats(dates2))
    dates3 = [d.replace('/', '') for d in dates]
    self.assertEquals(['%Y%d%m'], pa.determine_date_formats(dates3))
    
  def testAll2Digit1(self):
    # THe non standard one goes above 12 - it should be day
    dates = pa.dhist(['10/01/13', '09/17/13', '06/05/12'])
    self.assertEquals(['%m/%d/%y'], pa.determine_date_formats(dates))
    dates2 = [d.replace('/', '-') for d in dates]
    self.assertEquals(['%m-%d-%y'], pa.determine_date_formats(dates2))
    dates3 = [d.replace('/', '') for d in dates]
    self.assertEquals(['%m%d%y'], pa.determine_date_formats(dates3))
    

if __name__ == '__main__':
  unittest.main()