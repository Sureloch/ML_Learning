
import math

def point_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def resample_path(points, n=20):
    # Compute total length
    total_length = 0
    for i in range(len(points) - 1):
        total_length += point_distance(points[i], points[i+1])

    # Desired spacing between resampled points
    average_space = total_length / (n - 1)

    resampled = [points[0]]
    distance_accum = 0
    target = average_space
    i = 0

    while len(resampled) < n:
        p1 = points[i]
        p2 = points[i+1]
        seg_len = point_distance(p1, p2)

        if distance_accum + seg_len >= target:
            # How far along the segment the target lies
            ratio = (target - distance_accum) / seg_len

            # Linear interpolation
            new_x = p1[0] + ratio * (p2[0] - p1[0])
            new_y = p1[1] + ratio * (p2[1] - p1[1])
            resampled.append((new_x, new_y))
            distance_accum = target - average_space
            target += average_space
        else:
            distance_accum += seg_len
            i += 1

            if i >= len(points) - 1:
                break

    # Ensure last point is exactly the last original point
    if len(resampled) < n:
        resampled.append(points[-1])

    return resampled

def dtw_cost_matrix(path_a, path_b):
    rows = len(path_a)
    cols = len(path_b)

    cost = [[0.0 for _ in range(cols)] for _ in range(rows)]

    for i in range(rows):
        for j in range(cols):
            cost[i][j] = point_distance(path_a[i], path_b[j])

    return cost
def dtw_accumulated_matrix(cost):
    rows = len(cost)
    cols = len(cost[0])

    # Create matrix of same size
    acc = [[0.0 for _ in range(cols)] for _ in range(rows)]

    # --- 1. Initialize the top-left corner ---
    acc[0][0] = cost[0][0]

    # --- 2. First row (can only come from the left) ---
    for j in range(1, cols):
        acc[0][j] = cost[0][j] + acc[0][j-1]

    # --- 3. First column (can only come from above) ---
    for i in range(1, rows):
        acc[i][0] = cost[i][0] + acc[i-1][0]

    # --- 4. Main DP loop ---
    for i in range(1, rows):
        for j in range(1, cols):
            acc[i][j] = cost[i][j] + min(
                acc[i-1][j],     # from above
                acc[i][j-1],     # from left
                acc[i-1][j-1]    # from diagonal
            )

    return acc

def dtw(path_a, path_b):
    cost = dtw_cost_matrix(path_a, path_b)
    acc = dtw_accumulated_matrix(cost)
    return acc[-1][-1]


path_a = [(0,0), (1,2), (2,5)]
path_b = [(0,0), (1,3), (2,5)]
print("Similar paths:", dtw(path_a, path_b))

# Test 2 - different paths, should be HIGH score
path_c = [(0,0), (3,1), (6,2)]
print("Different paths:", dtw(path_a, path_c))