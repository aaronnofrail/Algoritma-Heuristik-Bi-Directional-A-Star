"""
Implementasi Bidirectional A* Search
Studi Kasus: Rute Optimal Pete-Pete Unhas (Tamalanrea) -> Pantai Losari
Kelompok 5B - Kecerdasan Buatan
"""

import heapq

# Definisi graf dasar (waktu tempuh sebelum penalti transit), dalam menit
BASE_EDGES = {
    ("Unhas", "Sudiang"): 33.60,
    ("Unhas", "BTP"): 9.60,
    ("Unhas", "Panakkukang"): 22.50,
    ("BTP", "Daya"): 16.20,
    ("BTP", "Panakkukang"): 27.90,
    ("Daya", "Sudiang"): 18.00,
    ("Panakkukang", "Karebosi"): 20.70,
    ("Daya", "Karebosi"): 55.50,
    ("Karebosi", "Losari"): 6.90,
    ("Panakkukang", "Losari"): 18.60,
}

START = "Unhas"
GOAL = "Losari"
TRANSIT_PENALTY = 5.0

# Nilai heuristik dua arah (jarak garis lurus x 3 menit/km)
H_FORWARD = {   # estimasi node -> Losari (goals)
    "Unhas": 27.33, "Sudiang": 45.69, "Daya": 33.75, "BTP": 31.23,
    "Panakkukang": 15.09, "Karebosi": 4.02, "Losari": 0.0,
}
H_BACKWARD = {  # estimasi node -> Unhas (start)
    "Unhas": 0.0, "Sudiang": 23.10, "Daya": 9.75, "BTP": 8.61,
    "Panakkukang": 13.32, "Karebosi": 26.01, "Losari": 27.33,
}


def build_adjacency():
    """Bangun adjacency list undirected + waktu final (dengan penalti transit)."""
    adj = {}

    def add(u, v, w):
        adj.setdefault(u, []).append((v, w))
        adj.setdefault(v, []).append((u, w))

    for (u, v), base_w in BASE_EDGES.items():
        penalty = 0.0 if (u == GOAL or v == GOAL) else TRANSIT_PENALTY
        add(u, v, base_w + penalty)
    return adj


ADJ = build_adjacency()


def reconstruct_path(meet_node, parent_f, parent_b):
    """Gabungkan jalur dari sisi forward dan backward lewat meet_node."""
    path_f = []
    node = meet_node
    while node is not None:
        path_f.append(node)
        node = parent_f.get(node)
    path_f.reverse()

    path_b = []
    node = parent_b.get(meet_node)
    while node is not None:
        path_b.append(node)
        node = parent_b.get(node)

    return path_f + path_b


def bidirectional_astar(start=START, goal=GOAL, verbose=True):
    open_f = [(H_FORWARD[start], 0.0, start)]
    open_b = [(H_BACKWARD[goal], 0.0, goal)]

    g_f = {start: 0.0}
    g_b = {goal: 0.0}
    closed_f = {}
    closed_b = {}
    parent_f = {start: None}
    parent_b = {goal: None}

    best = float("inf")
    meet_node = None
    step = 0

    while open_f and open_b:
        f_min_f = open_f[0][0]
        f_min_b = open_b[0][0]
        if f_min_f + f_min_b >= best:
            break

        step += 1
        f_val, g_val, node = heapq.heappop(open_f)
        if node in closed_f and closed_f[node] <= g_val:
            continue
        closed_f[node] = g_val
        if verbose:
            print(f"[Forward #{step}] proses {node:<12} g={g_val:6.2f}  f={f_val:6.2f}")

        if node in closed_b:
            cand = g_val + g_b[node]
            if cand < best:
                best, meet_node = cand, node

        for nb, w in ADJ.get(node, []):
            new_g = g_val + w
            if new_g < g_f.get(nb, float("inf")):
                g_f[nb] = new_g
                parent_f[nb] = node
                heapq.heappush(open_f, (new_g + H_FORWARD[nb], new_g, nb))

        f_val, g_val, node = heapq.heappop(open_b)
        if node in closed_b and closed_b[node] <= g_val:
            continue
        closed_b[node] = g_val
        if verbose:
            print(f"[Backward #{step}] proses {node:<12} g={g_val:6.2f}  f={f_val:6.2f}")

        if node in closed_f:
            cand = g_val + g_f[node]
            if cand < best:
                best, meet_node = cand, node

        for nb, w in ADJ.get(node, []):
            new_g = g_val + w
            if new_g < g_b.get(nb, float("inf")):
                g_b[nb] = new_g
                parent_b[nb] = node
                heapq.heappush(open_b, (new_g + H_BACKWARD[nb], new_g, nb))

    path = reconstruct_path(meet_node, parent_f, parent_b) if meet_node else None
    return best, path, meet_node


if __name__ == "__main__":
    total_cost, route, meet = bidirectional_astar()
    print("\n=== HASIL PENCARIAN ===")
    print(f"Simpul pertemuan (meet node) : {meet}")
    print(f"Total path cost optimal      : {total_cost:.2f} menit")
    print(f"Rute optimal                 : {' -> '.join(route)}")