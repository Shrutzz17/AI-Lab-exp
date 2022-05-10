from collections import defaultdict
 
class Graph:
 
    def __init__(self):
 
        self.graph = defaultdict(list)
 
    def addEdge(self, u, v, w):
        self.graph[u].append([v,w])
 
    def DFSUtil(self, v, visited):
 
        visited.add(v[0])
        print(v[0], end=' ')
 

        for neighbour in self.graph[v[0]]:
            if neighbour[0] not in visited:
                self.DFSUtil(neighbour, visited)
                
 
    def DFS(self, v):
        visited = set()
        self.DFSUtil(v, visited)
 

g = Graph()
g.addEdge('A', 'B', 4)
g.addEdge('A', 'C', 1)
g.addEdge('B', 'D', 3)
g.addEdge('B', 'E', 8)
g.addEdge('C', 'D', 2)
g.addEdge('C', 'F', 6)
g.addEdge('D', 'E', 4)
g.addEdge('E', 'G', 2)
g.addEdge('F', 'G', 8)
 
print ("The depth first traversal of node is: ")
g.DFS('A')
