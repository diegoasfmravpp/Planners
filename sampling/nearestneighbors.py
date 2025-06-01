from . import kdtree

class NearestNeighbors:
    def __init__(self,metric,method='bruteforce'):
        self.metric = metric
        self.method = method
        if self.method == 'kdtree':
            self.kdtree = kdtree.KDTree(self.metric)
        else:
            self.nodes = []

    def reset(self):
        if self.method == 'kdtree':
            self.kdtree = kdtree.KDTree(self.metric)
        else:
            self.nodes = []

    def add(self,point,data=None):
        """Adds a point with an associated datum."""
        if self.method == 'kdtree':
            self.kdtree.add(point,data)
            self.kdtree.rebalance()
        else:
            self.nodes.append((point,data))

    def remove(self,point,data=None):
        """Removes a point, optionally matching the data too.
        Time is O(nearest).  Returns the number of points removed.
        (TODO: can only be 0 or 1 at the moment)."""
        if self.method == 'kdtree':
            res = self.kdtree.remove(point,data)
            if res == 0:
                print("KDTree: Unable to remove",point,"does not exist")
            return res
        else:
            for i,(p,pd) in enumerate(self.nodes):
                if p == point and (data == None or pd==data):
                    del self.nodes[i]
                    return 1
            print("ERROR REMOVING POINT FROM BRUTE-FORCE NN STRUCTURE")
            for p,pd in self.nodes:
                print(p,pd)
        return 0
            

    def set(self,points,datas=None):
        """Sets the point set to a list of points."""
        if datas==None:
            datas = [None]*len(points)
        if hasattr(self,'kdtree'):
            print("Resetting KD tree...")
            self.kdtree.set(points,datas)
        else:
            self.nodes = list(zip(points,datas))

    def nearest(self,pt,filter=None):
        """Nearest neighbor lookup, possibly with filter"""
        if self.method == 'kdtree':
            res = self.kdtree.nearest(pt,filter)
            return res
        else:
            #brute force
            res = None    
            dbest = float('inf')
            for p,data in self.nodes:
                d = self.metric(p,pt)
                if d < dbest and (filter == None or not filter(p,data)):
                    res = (p,data)
                    dbest = d
            return res
    
    def neighbors(self,pt,radius):
        """Range query, all points within pt.  Filtering can be done
        afterward by the user."""
        if self.method == 'kdtree':
            res = self.kdtree.neighbors(pt,radius)
            return res
        else:
            #brute force
            res = []
            for p,data in self.nodes:
                d = self.metric(p,pt)
                if d < radius:
                    res.append((p,data))
            return res
      

