"""Very simple k-means clustering implementation for reference/toying with."""

import numpy as np
import pdb

def kmc(data, num_clusters, labels=None):
  """Assume data is an NxM thing with the major axis
  being the one with the items we wish to cluster.
  Distance metric is L^2 on R^M.
  """
  # First, generate some arbitrary initial centers:
  N, M = data.shape
  centers = np.random.ranf((num_clusters, M))
  old_closest = None
  
  # Consider adding up to 10 centers:
  for new_center in range(25):
    for step in range(100):
      # Now, for each data point, find the closest cluster:
      dists = ((data[:,:,np.newaxis] - centers.T[np.newaxis,:,:])**2).sum(axis=1)
      closest = np.argmin(dists, axis=1)
  
      # Now, update the centers to be the average of all points
      # whose closest center is the given one:
      new_centers = []
    
      for k in range(num_clusters):
        nx = np.where(closest == k)
        num_members = len(nx[0])
        if num_members == 0:
          # Cluster has no members; create a random one
          # by picking 3 points at random and using their average:
          new_center = data[np.argsort(np.random.ranf(N))[:3]].mean(axis=0)
          five_closest = 'created new center'
        else:
          new_center = data[nx].mean(axis=0)
          five_closest = '; '.join(labels[nx[0][n]] for n in np.argsort(dists[nx,k])[0,:5]) if labels else ''
        new_centers.append(new_center)
        print "Cluster %3d has %6d members %s" % (k, num_members, five_closest)
      print
      centers = np.array(new_centers)
    
      # Keep iterating until the clusters don't change:
      if old_closest is not None and np.all(old_closest == closest):
        print "clusters didn't change"
        break
      old_closest = closest
  
    # Now, let's take the biggest cluster and try to break it up by picking 3 random points
    # inside it and introducing those as a new point.
    # Let's also replace any size-1 cluster via the same approach:
    num_members_per_cluster = [(closest == k).sum() for k in range(num_clusters)]
    biggest = np.argmax(num_members_per_cluster)
    biggest_nx = np.where(closest == biggest)
    biggest_members = len(biggest_nx[0])
    new_centers = []
    for k, (num_members, center) in enumerate(zip(num_members_per_cluster, centers)):
      if k == biggest or num_members == 1:
        new_center = data[biggest_nx][np.argsort(np.random.ranf(biggest_members))[:3]].mean(axis=0)
        if k == biggest:
          new_centers.append(center)
        new_centers.append(new_center)
      else:
        new_centers.append(center)
    assert len(new_centers) == num_clusters + 1
    new_centers = list(centers)
    new_centers.append(new_center)
    centers = np.array(new_centers)
    num_clusters = len(centers)
  