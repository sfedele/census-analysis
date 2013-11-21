"""Compute a state/occupation SVD, and output some data
about the state/occupation coefficients of the first
several principle components to some CSVs.
"""

import csv

from prelude import *
import census_analyze


def main():
  wdata, rdata, job_names = census_analyze.loaddata()
  joined_data = census_analyze.joindata(wdata, rdata)
  occupation_rows = census_analyze.state_occupation_rows(joined_data, job_names)

  # Build a matrix NxM, N = #jobs, M=#States
  arr = np.array([vals for code, vals in occupation_rows])
  U, s, V = np.linalg.svd(arr, full_matrices=False)
  states = map(itemgetter(0), occupation_rows)
  assert sorted(states) == states
  jobs = map(itemgetter(1), sorted(job_names.iteritems()))
  
  abbreviations = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', '', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']

  # Write first 4 components out:
  for k in range(4):
    # Write out state csv
    with open('output/component_%s_by_state.csv' % (k + 1), 'w') as f:
      writer = csv.writer(f)
      writer.writerow(['State', 'Abbreviation', 'Coefficient'])
      writer.writerows(zip(states, abbreviations, U[:,k]))
      
    # Write out jobs csv:
    with open('output/component_%s_by_job.csv' % (k + 1), 'w') as f:
      writer = csv.writer(f)
      writer.writerow(['Code', 'Job', 'Coefficient'])
      writer.writerows(zip(sorted(job_names.keys()), jobs, V[k]))

  # Eigenvalues
  plt.plot(s); plt.title("Eigenvalues"); plt.grid(); plt.show()


if __name__ == '__main__':
  main()
  