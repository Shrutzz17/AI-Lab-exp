from queue import PriorityQueue
v = 9
graph = [[] for i in range(v)]
def best_first_search(source, target, n):
  visited = [0] * n
  visited[0] = True
  pq = PriorityQueue()
  pq.put((0, source))
  while pq.empty() == False:
    u = pq.get()[1]
    print(u, end=" ")
    if u == target:
      break
    for v, c in graph[u]:
      if visited[v] == False:
        visited[v] = True
        pq.put((c, v))
  print()
def addedge(x, y, cost):
  graph[x].append((y, cost))
  graph[y].append((x, cost))
addedge(0, 1, 4)
addedge(0, 2, 1)
addedge(1, 3, 3)
addedge(1, 4, 8)
addedge(2, 3, 2)
addedge(2, 5, 6)
addedge(3, 4, 4)
addedge(4, 6, 2)
addedge(5, 6, 8)
source = 0
target = 6
best_first_search(source, target, v)
