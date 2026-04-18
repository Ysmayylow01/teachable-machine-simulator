from flask import Flask, render_template, jsonify, request
from sklearn.datasets import make_classification, make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (silhouette_score, accuracy_score,
                              classification_report)
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import base64, io
import json

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────
#  SUPERVISED
# ─────────────────────────────────────────────────────────────
@app.route('/api/supervised/generate', methods=['POST'])
def supervised_generate():
    body   = request.get_json(silent=True) or {}
    n_cls  = max(2, int(body.get('n_classes', 3)))
    n_samp = int(body.get('n_samples', 200))

    X, y = make_classification(
        n_samples=n_samp, n_features=2, n_informative=2,
        n_redundant=0, n_clusters_per_class=1,
        n_classes=n_cls, random_state=42
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # scale to canvas coords (0-400 x 0-300)
    X[:, 0] = (X[:, 0] - X[:, 0].min()) / (np.ptp(X[:, 0]) or 1) * 360 + 20
    X[:, 1] = (X[:, 1] - X[:, 1].min()) / (np.ptp(X[:, 1]) or 1) * 260 + 20

    points = [{"x": float(X[i,0]), "y": float(X[i,1]), "label": int(y[i])}
              for i in range(len(X))]
    return jsonify({"points": points, "n_classes": n_cls})


@app.route('/api/supervised/train', methods=['POST'])
def supervised_train():
    body      = request.get_json(silent=True) or {}
    points    = body.get('points', [])
    algorithm = body.get('algorithm', 'knn')
    epochs    = int(body.get('epochs', 50))

    if not points:
        return jsonify({"error": "No data"}), 400

    X = np.array([[p['x'], p['y']] for p in points])
    y = np.array([p['label'] for p in points])

    # normalise back
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_s, y, test_size=0.2, random_state=42)

    models = {
        'knn':      KNeighborsClassifier(n_neighbors=5),
        'tree':     DecisionTreeClassifier(max_depth=5, random_state=42),
        'logistic': LogisticRegression(max_iter=1000, random_state=42),
        'svm':      SVC(kernel='rbf', probability=True, random_state=42),
    }
    clf = models.get(algorithm, models['knn'])
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))

    # simulate epoch loss curve
    np.random.seed(42)
    losses = []
    for e in range(epochs):
        t = e / epochs
        base  = 1.0 - t * 0.85
        noise = np.random.normal(0, 0.03 * (1 - t))
        losses.append(round(float(max(0.01, base + noise)), 4))

    # decision boundary grid
    nx, ny = 40, 30
    xs = np.linspace(X_s[:,0].min()-0.5, X_s[:,0].max()+0.5, nx)
    ys = np.linspace(X_s[:,1].min()-0.5, X_s[:,1].max()+0.5, ny)
    xx, yy = np.meshgrid(xs, ys)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(ny, nx)

    # back to canvas coords
    def to_canvas_x(v):
        return float((v - X_s[:,0].min()) / ((X_s[:,0].max()-X_s[:,0].min()) or 1) * 360 + 20)
    def to_canvas_y(v):
        return float((v - X_s[:,1].min()) / ((X_s[:,1].max()-X_s[:,1].min()) or 1) * 260 + 20)

    boundary = []
    for r in range(ny):
        for c in range(nx):
            boundary.append({
                "x": to_canvas_x(xs[c]),
                "y": to_canvas_y(ys[r]),
                "label": int(Z[r, c])
            })

    # class stats
    classes = sorted(set(y.tolist()))
    class_stats = []
    for cl in classes:
        mask = y_test == cl
        if mask.sum() == 0:
            class_stats.append({"class": cl, "precision": 0, "recall": 0, "support": 0})
            continue
        pred  = clf.predict(X_test[mask])
        prec  = float((pred == cl).mean())
        rec   = float((pred == cl).sum() / mask.sum())
        class_stats.append({
            "class": cl,
            "precision": round(prec, 3),
            "recall":    round(rec, 3),
            "support":   int(mask.sum())
        })

    return jsonify({
        "accuracy":    round(float(acc), 4),
        "losses":      losses,
        "boundary":    boundary,
        "class_stats": class_stats,
        "algorithm":   algorithm,
    })


# ─────────────────────────────────────────────────────────────
#  UNSUPERVISED
# ─────────────────────────────────────────────────────────────
@app.route('/api/unsupervised/generate', methods=['POST'])
def unsupervised_generate():
    body   = request.get_json(silent=True) or {}
    k      = max(2, int(body.get('k', 3)))
    n_samp = int(body.get('n_samples', 300))

    X, y_true = make_blobs(n_samples=n_samp, centers=k,
                            cluster_std=1.0, random_state=42)
    X[:, 0] = (X[:,0] - X[:,0].min()) / ((X[:,0].max()-X[:,0].min()) or 1) * 360 + 20
    X[:, 1] = (X[:,1] - X[:,1].min()) / ((X[:,1].max()-X[:,1].min()) or 1) * 260 + 20

    points = [{"x": float(X[i,0]), "y": float(X[i,1])} for i in range(len(X))]
    return jsonify({"points": points})


@app.route('/api/unsupervised/cluster', methods=['POST'])
def unsupervised_cluster():
    body      = request.get_json(silent=True) or {}
    points    = body.get('points', [])
    algorithm = body.get('algorithm', 'kmeans')
    k         = max(2, int(body.get('k', 3)))

    if not points:
        return jsonify({"error": "No data"}), 400

    X = np.array([[p['x'], p['y']] for p in points])
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    if algorithm == 'kmeans':
        model  = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_s)
        centers_s = model.cluster_centers_
        centers = scaler.inverse_transform(centers_s).tolist()
        inertia = float(model.inertia_)
    elif algorithm == 'dbscan':
        model  = DBSCAN(eps=0.5, min_samples=5)
        labels = model.fit_predict(X_s)
        centers = []
        inertia = 0.0
    elif algorithm == 'hierarchical':
        model  = AgglomerativeClustering(n_clusters=k)
        labels = model.fit_predict(X_s)
        centers = []
        inertia = 0.0
    elif algorithm == 'gmm':
        model  = GaussianMixture(n_components=k, random_state=42)
        model.fit(X_s)
        labels = model.predict(X_s)
        centers_s = model.means_
        centers = scaler.inverse_transform(centers_s).tolist()
        inertia = float(-model.score(X_s) * len(X_s))
    else:
        model  = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_s)
        centers = []
        inertia = 0.0

    labels = [int(l) for l in labels]
    unique = [l for l in set(labels) if l != -1]
    n_clusters = len(unique)

    sil = 0.0
    if n_clusters >= 2 and n_clusters < len(X):
        mask = np.array(labels) != -1
        if mask.sum() >= n_clusters:
            try:
                sil = float(silhouette_score(X_s[mask], np.array(labels)[mask]))
            except Exception:
                sil = 0.0

    # cluster counts
    cluster_counts = {}
    for l in labels:
        cluster_counts[l] = cluster_counts.get(l, 0) + 1

    # inertia iterations for chart
    np.random.seed(0)
    iters = []
    start = inertia * 3 if inertia > 0 else 1000
    for i in range(20):
        t = i / 19
        iters.append(round(float(start * (1 - t * 0.85) + np.random.normal(0, start * 0.02)), 2))

    return jsonify({
        "labels":         labels,
        "centers":        centers,
        "n_clusters":     n_clusters,
        "inertia":        round(inertia, 2),
        "silhouette":     round(sil, 4),
        "cluster_counts": cluster_counts,
        "inertia_iters":  iters,
    })


@app.route('/api/unsupervised/cluster_images', methods=['POST'])
def unsupervised_cluster_images():
    body   = request.get_json(silent=True) or {}
    images = body.get('images', [])
    k      = max(2, int(body.get('k', 3)))

    if len(images) < k:
        return jsonify({"error": f"Need at least {k} images (got {len(images)})"}), 400

    feats = []
    for b64 in images:
        if ',' in b64:
            b64 = b64.split(',', 1)[1]
        try:
            img = Image.open(io.BytesIO(base64.b64decode(b64))).convert('L').resize((24, 24))
        except Exception as e:
            return jsonify({"error": f"Bad image: {e}"}), 400
        feats.append(np.asarray(img, dtype=np.float32).flatten() / 255.0)

    X = np.stack(feats)

    model   = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels  = model.fit_predict(X)
    inertia = float(model.inertia_)

    pca = PCA(n_components=2, random_state=42)
    X2  = pca.fit_transform(X)
    X2[:, 0] = (X2[:,0] - X2[:,0].min()) / ((X2[:,0].max()-X2[:,0].min()) or 1) * 360 + 20
    X2[:, 1] = (X2[:,1] - X2[:,1].min()) / ((X2[:,1].max()-X2[:,1].min()) or 1) * 260 + 20
    points = [{"x": float(X2[i,0]), "y": float(X2[i,1]), "idx": i} for i in range(len(X))]

    sil = 0.0
    if 2 <= k < len(X):
        try:
            sil = float(silhouette_score(X, labels))
        except Exception:
            sil = 0.0

    cluster_counts = {}
    for l in labels:
        cluster_counts[int(l)] = cluster_counts.get(int(l), 0) + 1

    np.random.seed(0)
    iters = []
    start = inertia * 3 if inertia > 0 else 1.0
    for i in range(20):
        t = i / 19
        iters.append(round(float(start * (1 - t * 0.85) + np.random.normal(0, start * 0.02)), 4))

    return jsonify({
        "points":         points,
        "labels":         [int(l) for l in labels],
        "cluster_counts": cluster_counts,
        "n_clusters":     k,
        "inertia":        round(inertia, 4),
        "silhouette":     round(sil, 4),
        "inertia_iters":  iters,
    })


# ─────────────────────────────────────────────────────────────
#  REINFORCEMENT  (tabular Q-learning on grid)
# ─────────────────────────────────────────────────────────────
GRID = 5

def _encode(r, c): return r * GRID + c
def _decode(s):    return divmod(s, GRID)

def _next(r, c, a, walls, dangers):
    moves = [(-1,0),(0,1),(1,0),(0,-1)]
    nr, nc = r + moves[a][0], c + moves[a][1]
    if nr < 0 or nr >= GRID or nc < 0 or nc >= GRID:
        return r, c, -1.0, False
    if (nr, nc) in walls:
        return r, c, -1.0, False
    if (nr, nc) in dangers:
        return nr, nc, -10.0, True
    if nr == GRID-1 and nc == GRID-1:
        return nr, nc, +100.0, True
    return nr, nc, -0.1, False


@app.route('/api/rl/train', methods=['POST'])
def rl_train():
    body     = request.get_json(silent=True) or {}
    alpha    = float(body.get('alpha', 0.1))
    gamma    = float(body.get('gamma', 0.95))
    epsilon  = float(body.get('epsilon', 0.9))
    episodes = int(body.get('episodes', 300))
    walls    = [tuple(w) for w in body.get('walls', [[1,1],[2,3],[1,3]])]
    dangers  = [tuple(d) for d in body.get('dangers', [[3,1]])]

    walls_set   = set(map(tuple, walls))
    dangers_set = set(map(tuple, dangers))

    n_states  = GRID * GRID
    n_actions = 4
    Q = np.zeros((n_states, n_actions))
    eps = epsilon
    eps_decay = 0.995
    eps_min   = 0.05

    rewards_per_ep = []
    steps_per_ep   = []
    success_count  = 0

    for ep in range(episodes):
        r, c  = 0, 0
        total = 0.0
        steps = 0
        done  = False

        while not done and steps < 200:
            s = _encode(r, c)
            if np.random.rand() < eps:
                a = np.random.randint(n_actions)
            else:
                a = int(np.argmax(Q[s]))

            nr, nc, rew, done = _next(r, c, a, walls_set, dangers_set)
            ns = _encode(nr, nc)
            Q[s, a] += alpha * (rew + gamma * np.max(Q[ns]) - Q[s, a])

            r, c   = nr, nc
            total += rew
            steps += 1

        if r == GRID-1 and c == GRID-1:
            success_count += 1

        eps = max(eps_min, eps * eps_decay)
        rewards_per_ep.append(round(float(total), 2))
        steps_per_ep.append(steps)

    # policy: best action per cell
    ARROWS = ['↑', '→', '↓', '←']
    policy = []
    q_vals = []
    for row in range(GRID):
        for col in range(GRID):
            s  = _encode(row, col)
            ba = int(np.argmax(Q[s]))
            policy.append({
                "r": row, "c": col,
                "action": ba,
                "arrow":  ARROWS[ba],
                "q_max":  round(float(np.max(Q[s])), 2)
            })
            q_vals.append(round(float(np.max(Q[s])), 2))

    # smoothed reward
    window = max(1, episodes // 20)
    smoothed = []
    for i in range(len(rewards_per_ep)):
        sl = rewards_per_ep[max(0, i-window):i+1]
        smoothed.append(round(sum(sl)/len(sl), 2))

    avg_steps = round(sum(steps_per_ep[-50:]) / 50, 1) if episodes >= 50 else round(sum(steps_per_ep)/len(steps_per_ep),1)

    return jsonify({
        "rewards":       rewards_per_ep,
        "smoothed":      smoothed,
        "steps":         steps_per_ep,
        "policy":        policy,
        "q_vals":        q_vals,
        "success_count": success_count,
        "total_episodes":episodes,
        "avg_steps":     avg_steps,
        "success_rate":  round(success_count / episodes * 100, 1),
    })


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, port=5015, host='0.0.0.0')