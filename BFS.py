from collections import defaultdict
class Graph:
 
    def __init__(self):
        self.graph = defaultdict(list)
 
    def addEdge(self,u,v,w):
        self.graph[u].append([v,w])
 
    def BFS(self, s):
 
        visited = [False] * (len(self.graph) + 1)
        queue = []
        queue.append(s)
        visited[int(ord(s) - ord('A'))] = True
 
        while queue:
            s = queue.pop(0)
            print (s, end = " ")

            for i in self.graph[s]:
                if visited[int(ord(i[0]) - ord('A'))] == False:
                    queue.append(i[0])
                    visited[int(ord(i[0]) - ord('A'))] = True

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
 
print ("The breadth first traversal of node is: ")
g.BFS('A')
 
