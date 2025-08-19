import pandas as pd
import heapq as hq

# points = [point_1, point_2 , ... , point_n]
# point_i = [feature_1, feature_2, ... , feature_m , class_label]
def kNearestNeighbors_classifier(k: int , datapoint: list , points: list):
    if k > len(points):
        raise ValueError("k must not be larger than the total number of data points")
    
    sup = float('inf')
    knns = []
    hq.heapify(knns)
    for point in points:
        dist = 0.0
        for i in range(len(point)-1):
            dist += (datapoint[i] - point[i])**2
        if dist < sup:
            if len(knns) < k:
                hq.heappush(knns,(-dist,point))
                if len(knns) == k:
                    sup = -knns[0][0]
            else:
                hq.heappushpop(knns,(-dist,point))
                sup = -knns[0][0]

    # majority vote count
    classes = {}
    majority_class = 0
    max_count = 0
    for neighbor in knns:
        if neighbor[1][-1] in classes:
            classes[neighbor[1][-1]] += 1
        else:
            classes[neighbor[1][-1]] = 1

        if max_count < classes[neighbor[1][-1]]:
            majority_class , max_count = neighbor[1][-1] , classes[neighbor[1][-1]] 
    
    return majority_class


# testing
data = pd.read_csv("ObesityClassification.csv")
points = [point for point in data[["Age","Gender","Height","Weight","Label"]].values.tolist()]
for p in points:
    p[1] = 1 if p[1] == "Male" else 0

print(kNearestNeighbors_classifier(15,[16,1,160,50],points))
