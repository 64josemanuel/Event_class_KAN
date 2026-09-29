import os
import sys
import pickle
import re
import warnings
import numpy as np
import scipy.io
import sympy
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
from collections import Counter

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.covariance import GraphicalLasso
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from pygsp import graphs
from kan import KAN

# Global Settings & Directories

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore")
np.random.seed(42)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "axes.linewidth": 1.0,
    "legend.fontsize": 8,
    "legend.framealpha": 1.0,
    "legend.edgecolor": "black",
    "legend.fancybox": False,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "grid.alpha": 0.5,
    "grid.linestyle": "--",
    "figure.dpi": 600, 
})

DIR_DATOS = r"./data" # path
DIR_CHK = r"./checkpoints"
DIR_FIGS = r"./figures"
os.makedirs(DIR_CHK, exist_ok=True)
os.makedirs(DIR_FIGS, exist_ok=True)

archivo_dataset_crudo = os.path.join(DIR_CHK, "raw_dataset_master.pkl")
archivo_checkpoint_paso2 = os.path.join(DIR_CHK, "checkpoint_step2.pkl")
archivo_dataset_pt = os.path.join(DIR_CHK, "dataset_tensors_physics.pt")
archivo_modelo_binario = os.path.join(DIR_CHK, "model_kan_binary.pt")
archivo_historial = os.path.join(DIR_CHK, "optimization_history_binary.pkl")
archivo_dataset_mc = os.path.join(DIR_CHK, "dataset_tensors_mc_physics.pt")
archivo_modelo_multiclase = os.path.join(DIR_CHK, "model_kan_multiclass.pt")
ruta_ecuacion = os.path.join(DIR_CHK, "boundary_equations.txt")
archivo_stats_simbolicas = os.path.join(DIR_CHK, "symbolic_stats.pkl")


# STEP 1: Data Loading & Errors prevention

if os.path.exists(archivo_dataset_crudo):
    with open(archivo_dataset_crudo, "rb") as f:
        datos = pickle.load(f)
    dataset_crudo = datos['dataset_crudo']
    W_fisica = datos['W_fisica']
    mascara_gen = datos['mascara_gen']
    mascara_load = datos['mascara_load']
    indices_buses_validos = datos['indices_buses_validos']
else:
    frecuencia_base, limite_desviacion = 60.0, 2.0 
    
    archivos_mat = [os.path.join(r, a) for r, d, files in os.walk(DIR_DATOS) for a in files if a.endswith(".mat") and "Steady" not in a]
    if not archivos_mat:
        raise ValueError("No .mat files found in ./data directory.")
    
    mat_topo = scipy.io.loadmat(archivos_mat[0], squeeze_me=True, struct_as_record=False)
    matriz_bus_global = mat_topo['bus'] if isinstance(mat_topo, dict) else mat_topo.bus
    matriz_line_global = mat_topo['line'] if isinstance(mat_topo, dict) else mat_topo.line
    N_max = matriz_bus_global.shape[0] 

    dataset_crudo = []
    indices_buses_validos = None

    for archivo in tqdm(archivos_mat, desc="Processing MATs (Super-Graph)"):
        ruta = archivo.lower().replace('\\', '/')
        etiqueta = 0 if '/generators/' in ruta else (1 if '/lines/' in ruta else (2 if '/loads/' in ruta else -1))
        if etiqueta == -1: continue
        
        try:
            mat = scipy.io.loadmat(archivo, squeeze_me=True, struct_as_record=False)
            
            # Skip corrupt files
            if 'sstr' not in mat:
                continue
                
            sstr_obj = mat['sstr']
            bus_freq = sstr_obj.bus_freq * frecuencia_base
            bus_v = np.abs(sstr_obj.bus_v)
            t_val = sstr_obj.t
                
            # alignment correction
            if bus_freq.shape[0] < bus_v.shape[0]:
                padding_f = np.full((bus_v.shape[0] - bus_freq.shape[0], bus_freq.shape[1]), frecuencia_base)
                bus_freq = np.vstack((bus_freq, padding_f))
            elif bus_v.shape[0] < bus_freq.shape[0]:
                padding_v = np.ones((bus_freq.shape[0] - bus_v.shape[0], bus_v.shape[1]))
                bus_v = np.vstack((bus_v, padding_v))
                
            N_sim = bus_v.shape[0]
            if N_sim > N_max:
                bus_freq = bus_freq[:N_max, :]
                bus_v = bus_v[:N_max, :]
            elif N_sim < N_max:
                diferencia = N_max - N_sim
                padding_f_super = np.full((diferencia, bus_freq.shape[1]), frecuencia_base)
                bus_freq = np.vstack((bus_freq, padding_f_super))
                padding_v_super = np.ones((diferencia, bus_v.shape[1]))
                bus_v = np.vstack((bus_v, padding_v_super))
            
            if indices_buses_validos is None:
                indices_buses_validos = [i for i in range(N_max) if np.max(np.abs(bus_freq[i, :] - frecuencia_base)) < limite_desviacion]
            
            bf_f, bv_f = bus_freq[indices_buses_validos, :], bus_v[indices_buses_validos, :]
            
            dataset_crudo.append({
                'nombre': os.path.basename(archivo), 'etiqueta': etiqueta, 't': t_val,
                'delta_f': bf_f + np.random.normal(0, 0.0005, bf_f.shape) - frecuencia_base,
                'delta_v': bv_f + np.random.normal(0, 0.0001, bv_f.shape) - 1.0 
            })
            
        except Exception as e:
            print(f"\n[Warning] Skipped {os.path.basename(archivo)}: {e}")
            continue

    if not dataset_crudo or indices_buses_validos is None:
        raise ValueError("Error: No valid MAT files processed.")

    N_buses = len(indices_buses_validos)
    W_fisica, mascara_gen, mascara_load = np.zeros((N_buses, N_buses)), np.zeros(N_buses), np.zeros(N_buses)

    for fila in matriz_line_global:
        o, d = int(fila[0])-1, int(fila[1])-1
        if o in indices_buses_validos and d in indices_buses_validos:
            idx_o, idx_d = indices_buses_validos.index(o), indices_buses_validos.index(d)
            admitancia = 1.0 / np.sqrt(fila[2]**2 + fila[3]**2)
            W_fisica[idx_o, idx_d] = W_fisica[idx_d, idx_o] = admitancia

    for fila in matriz_bus_global:
        id_bus = int(fila[0])-1
        if id_bus in indices_buses_validos:
            idx = indices_buses_validos.index(id_bus)
            if int(fila[9]) in [1, 2] or id_bus < 46: mascara_gen[idx] = 1.0
            else: mascara_load[idx] = 1.0
                
    with open(archivo_dataset_crudo, "wb") as f:
        pickle.dump({'dataset_crudo': dataset_crudo, 'W_fisica': W_fisica, 'mascara_gen': mascara_gen,
                     'mascara_load': mascara_load, 'indices_buses_validos': indices_buses_validos}, f)
   
# STEP 2: Graphical Lasso & Topology Optimization

if os.path.exists(archivo_checkpoint_paso2):
    with open(archivo_checkpoint_paso2, "rb") as f:
        datos_paso2 = pickle.load(f)
        
    W_inferida = datos_paso2['W_inferida']
    W_final = datos_paso2['W_final']
    historial_alphas = datos_paso2['historial_alphas']
    historial_aristas = datos_paso2['historial_aristas']
    alpha_optimo = datos_paso2['alpha_optimo']
    aristas_optimas = datos_paso2['aristas_optimas']
    objetivo_aristas = datos_paso2['objetivo_aristas']
    G = graphs.Graph(W_final)

else:
    datos_estandarizados = StandardScaler().fit_transform(dataset_crudo[0]['delta_f'].T)
    rango_alphas = [0.8, 0.5, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001]
    objetivo_aristas = (datos_estandarizados.shape[1] * 1.2, datos_estandarizados.shape[1] * 6)
    
    W_inferida = np.zeros((datos_estandarizados.shape[1], datos_estandarizados.shape[1]))
    historial_alphas, historial_aristas = [], []
    alpha_optimo, aristas_optimas = None, None

    for alpha in rango_alphas:
        try:
            modelo = GraphicalLasso(alpha=alpha, max_iter=1000, tol=1e-3, assume_centered=True)
            modelo.fit(datos_estandarizados)
            
            W_temp = np.abs(modelo.precision_)
            np.fill_diagonal(W_temp, 0)
            if np.max(W_temp) > 0:
                W_temp = W_temp / np.max(W_temp)
                W_temp[W_temp < 0.05] = 0 
                
            G_temp = graphs.Graph(W_temp)
            historial_alphas.append(alpha)
            historial_aristas.append(G_temp.Ne)
            
            if objetivo_aristas[0] <= G_temp.Ne <= objetivo_aristas[1] and alpha_optimo is None:
                W_inferida = W_temp
                alpha_optimo = alpha
                aristas_optimas = G_temp.Ne
        except Exception:
            continue

    if alpha_optimo is None and historial_alphas: W_inferida = W_temp

    W_fisica_norm = W_fisica / np.max(W_fisica) if np.max(W_fisica) > 0 else W_fisica
    W_final = (0.5 * W_fisica_norm) + (0.5 * W_inferida)
    W_final[W_final < 0.05] = 0
    np.fill_diagonal(W_final, 0)
    
    G = graphs.Graph(W_final)
    
    with open(archivo_checkpoint_paso2, "wb") as f:
        pickle.dump({
            'W_inferida': W_inferida, 'W_final': W_final, 'historial_alphas': historial_alphas,
            'historial_aristas': historial_aristas, 'alpha_optimo': alpha_optimo,
            'aristas_optimas': aristas_optimas, 'objetivo_aristas': objetivo_aristas
        }, f)

G.compute_fourier_basis()

# Plot 2A: Graphical Lasso Optimization
fig_opt, ax_opt = plt.subplots(figsize=(8, 5))
ax_opt.plot(historial_alphas, historial_aristas, marker='o', markerfacecolor='none', markeredgecolor='blue', color='blue', linewidth=1.5, label="Inference")
ax_opt.axhspan(objetivo_aristas[0], objetivo_aristas[1], color='gray', alpha=0.2, label="Physical Target Range")

if alpha_optimo:
    ax_opt.scatter([alpha_optimo], [aristas_optimas], color='red', s=150, marker='*', edgecolors='black', zorder=5, label="Optimum")
    idx_texto = len(historial_alphas) // 2
    ax_opt.text(historial_alphas[idx_texto], objetivo_aristas[1] * 1.1, 
                f'Heuristic target: {int(objetivo_aristas[0])} - {int(objetivo_aristas[1])} edges', 
                color='black', fontweight='bold', fontsize=9)

ax_opt.set(xscale='log', xlabel='Alpha (Regularization)', ylabel='Inferred Edges', title='Graphical Lasso Optimization')
ax_opt.invert_xaxis()
ax_opt.grid(True, which='both', linestyle='--', linewidth=0.5)
ax_opt.legend()
fig_opt.savefig(os.path.join(DIR_FIGS, "Fig2_A_Optimization.pdf"), bbox_inches='tight')

# Plot 2B: Adjacency Matrices 
W_fisica_norm = W_fisica / np.max(W_fisica) if np.max(W_fisica) > 0 else W_fisica
fig_mat, axes_mat = plt.subplots(1, 3, figsize=(18, 5))

sns.heatmap(W_fisica_norm, cmap="viridis", mask=(W_fisica_norm == 0), ax=axes_mat[0]).set_title("Physical Topology")
sns.heatmap(W_inferida, cmap="viridis", mask=(W_inferida == 0), ax=axes_mat[1]).set_title("Inferred Topology (GLASSO)")
sns.heatmap(W_final, cmap="viridis", mask=(W_final == 0), ax=axes_mat[2]).set_title("Hybrid Topology")

for ax in axes_mat:
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.0)

fig_mat.savefig(os.path.join(DIR_FIGS, "Fig2_B_Matrices.pdf"), bbox_inches='tight')

# Plot 2C: Laplacian Spectrum
fig_spec, ax_spec = plt.subplots(figsize=(8, 5))
ax_spec.plot(range(len(G.e)), G.e, marker='s', markerfacecolor='none', markeredgecolor='black', color='black', linewidth=1.5)
ax_spec.axvline(x=1, color='red', linestyle='--', linewidth=1.5, label="Fiedler Value")
ax_spec.set(xlabel='Frequency (k)', ylabel='Eigenvalue', title='GFT Spectrum')
ax_spec.grid(True, linestyle='--', linewidth=0.5)
ax_spec.legend()
fig_spec.savefig(os.path.join(DIR_FIGS, "Fig2_C_Spectrum.pdf"), bbox_inches='tight')

# STEP 3: Spectral Data Generation

if os.path.exists(archivo_dataset_pt):
    dataset = torch.load(archivo_dataset_pt, weights_only=False)
    X_train_np = dataset['train_input'].numpy()
    Y_train_np = dataset['train_label'].numpy()
else:
    X_datos, Y_etiquetas = [], []
    
    for evento in dataset_crudo:
        t, delta_f, delta_v = evento['t'], evento['delta_f'], evento['delta_v']
        
        idx_normal = np.argmin(np.abs(t - 0.5))
        espectro_v_norm = np.abs(G.gft(delta_v[:, idx_normal]))
        espectro_f_gen_norm = np.abs(G.gft(delta_f[:, idx_normal] * mascara_gen))
        espectro_f_load_norm = np.abs(G.gft(delta_f[:, idx_normal] * mascara_load))
        
        X_datos.append(np.concatenate([espectro_v_norm, espectro_f_gen_norm, espectro_f_load_norm]))
        Y_etiquetas.append([0.0])
        
        mascara_transitorio = t > 1.1
        espectro_v_post = np.max(np.abs(G.gft(delta_v[:, mascara_transitorio])), axis=1)
        espectro_f_gen_post = np.max(np.abs(G.gft(delta_f[:, mascara_transitorio] * mascara_gen[:, np.newaxis])), axis=1)
        espectro_f_load_post = np.max(np.abs(G.gft(delta_f[:, mascara_transitorio] * mascara_load[:, np.newaxis])), axis=1)
        
        X_datos.append(np.concatenate([espectro_v_post, espectro_f_gen_post, espectro_f_load_post]))
        Y_etiquetas.append([1.0])

    X_datos_np, Y_etiquetas_np = np.array(X_datos), np.array(Y_etiquetas)
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_datos_np, Y_etiquetas_np, test_size=0.20, random_state=42, stratify=Y_etiquetas_np
    )
    
    dataset = {
        'train_input': torch.tensor(X_train, dtype=torch.float32),
        'train_label': torch.tensor(Y_train, dtype=torch.float32),
        'test_input': torch.tensor(X_test, dtype=torch.float32),
        'test_label': torch.tensor(Y_test, dtype=torch.float32),
    }
    torch.save(dataset, archivo_dataset_pt)
    X_train_np, Y_train_np = X_train, Y_train

# Plot 3B: Tensor Anatomy
N = G.N
idx_transitorio = np.where(Y_train_np == 1.0)[0][0]
tensor_ejemplo = X_train_np[idx_transitorio]
y_max = np.max(tensor_ejemplo)

fig_tensor, ax_tensor = plt.subplots(figsize=(12, 4))
markerline3, stemlines3, baseline3 = ax_tensor.stem(range(len(tensor_ejemplo)), tensor_ejemplo, linefmt='#9467bd', markerfmt='o', basefmt=' ')
plt.setp(markerline3, markerfacecolor='none', markeredgecolor='#9467bd', markeredgewidth=1.0, markersize=4)
plt.setp(stemlines3, linewidth=1.0)

ax_tensor.axvline(x=N-0.5, color='k', linestyle='--', alpha=0.8, linewidth=1.0)
ax_tensor.axvline(x=2*N-0.5, color='k', linestyle='--', alpha=0.8, linewidth=1.0)

bbox_style = dict(facecolor='white', alpha=1.0, edgecolor='black', linewidth=1.0)
ax_tensor.text(N/2, y_max*0.9, r'$\Delta V$ Spectrum', ha='center', fontweight='bold', bbox=bbox_style)
ax_tensor.text(N + N/2, y_max*0.9, r'$\Delta f$ Spectrum (Gen)', ha='center', fontweight='bold', bbox=bbox_style)
ax_tensor.text(2*N + N/2, y_max*0.9, r'$\Delta f$ Spectrum (Load)', ha='center', fontweight='bold', bbox=bbox_style)

ax_tensor.set(xlabel=f'Tensor Index (Dimension = {3*N})', ylabel='Magnitude', title='Concatenated Tensor Anatomy')
ax_tensor.grid(True, linestyle='--', linewidth=0.5)
fig_tensor.savefig(os.path.join(DIR_FIGS, "Fig3_B_Tensor_Anatomy.pdf"), bbox_inches='tight')

# STEP 4: Binary KAN Training
dimension_entrada = G.N * 3
modelo_kan = KAN(width=[dimension_entrada, 3, 1], grid=5, k=3, seed=42)
modelo_kan_inicial = KAN(width=[dimension_entrada, 3, 1], grid=5, k=3, seed=42)

if os.path.exists(archivo_modelo_binario) and os.path.exists(archivo_historial):
    modelo_kan.load_state_dict(torch.load(archivo_modelo_binario, weights_only=True))
    with open(archivo_historial, "rb") as f:
        resultados_entrenamiento = pickle.load(f)
else:
    resultados_entrenamiento = modelo_kan.fit(dataset, opt="LBFGS", steps=20, log=10)
    torch.save(modelo_kan.state_dict(), archivo_modelo_binario)
    with open(archivo_historial, "wb") as f:
        pickle.dump(resultados_entrenamiento, f)

# STEP 5: Multiclass Dataset Construction (Only for desbalanced sets)
if os.path.exists(archivo_dataset_mc):
    dataset_mc = torch.load(archivo_dataset_mc, weights_only=False)
    X_train_mc = dataset_mc['train_input'].numpy()
    Y_train_mc = dataset_mc['train_label'].numpy()
else:
    X_datos_mc, Y_etiquetas_mc = [], []
    
    for evento in dataset_crudo:
        t, delta_f, delta_v = evento['t'], evento['delta_f'], evento['delta_v']
        clase_original = evento['etiqueta'] 
        
        ventanas_norm = [(0.1, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        carac_norm = []
        for t_ini, t_fin in ventanas_norm:
            mascara_n = (t > t_ini) & (t <= t_fin)
            carac_norm.extend([
                np.max(np.abs(G.gft(delta_v[:, mascara_n])), axis=1),
                np.max(np.abs(G.gft(delta_f[:, mascara_n] * mascara_gen[:, np.newaxis])), axis=1),
                np.max(np.abs(G.gft(delta_f[:, mascara_n] * mascara_load[:, np.newaxis])), axis=1)
            ])
        X_datos_mc.append(np.concatenate(carac_norm))
        Y_etiquetas_mc.append([1.0, 0.0, 0.0, 0.0])
        
        ventanas_trans = [(1.1, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 15.0), (15.0, 20.0)]
        carac_post = []
        for t_ini, t_fin in ventanas_trans:
            mascara_t = (t > t_ini) & (t <= t_fin)
            carac_post.extend([
                np.max(np.abs(G.gft(delta_v[:, mascara_t])), axis=1),
                np.max(np.abs(G.gft(delta_f[:, mascara_t] * mascara_gen[:, np.newaxis])), axis=1),
                np.max(np.abs(G.gft(delta_f[:, mascara_t] * mascara_load[:, np.newaxis])), axis=1)
            ])
        X_datos_mc.append(np.concatenate(carac_post))
        
        etiqueta_onehot = [0.0, 0.0, 0.0, 0.0]
        etiqueta_onehot[clase_original + 1] = 1.0 
        Y_etiquetas_mc.append(etiqueta_onehot)

    # Data Augmentation
    X_extra, Y_extra = [], []
    for x, y in zip(X_datos_mc, Y_etiquetas_mc):
        if y[1] == 1.0: # Generator Faults
            X_extra.append(x + np.random.normal(0, np.max(x)*0.005, size=x.shape))
            Y_extra.append(y)
    
    X_datos_mc.extend(X_extra)
    Y_etiquetas_mc.extend(Y_extra)

    X_np, Y_np = np.array(X_datos_mc), np.array(Y_etiquetas_mc)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_np, Y_np, test_size=0.20, random_state=42, stratify=Y_np        
    )
    
    dataset_mc = {
        'train_input': torch.tensor(X_train, dtype=torch.float32),
        'train_label': torch.tensor(Y_train, dtype=torch.float32),
        'test_input': torch.tensor(X_test, dtype=torch.float32),
        'test_label': torch.tensor(Y_test, dtype=torch.float32),
    }
    torch.save(dataset_mc, archivo_dataset_mc)
    X_train_mc, Y_train_mc = X_train, Y_train

# Plot 5A: Kinematic Evolution
idx_linea = np.where(Y_train_mc[:, 2] == 1.0)[0][0]
tensor_linea = X_train_mc[idx_linea]
matriz_evolucion = np.column_stack([tensor_linea[(w*3*N) : (w*3*N)+N] for w in range(5)])

fig_cin, ax_cin = plt.subplots(figsize=(8, 6))
sns.heatmap(matriz_evolucion, cmap="viridis", ax=ax_cin, 
            xticklabels=["W1 (1-2s)", "W2 (2-5s)", "W3 (5-10s)", "W4 (10-15s)", "W5 (15-20s)"],
            cbar_kws={'label': r'Magnitude |GFT| of $\Delta V$'})

ax_cin.set_xlabel('Time Windows', fontweight='bold')
ax_cin.set_ylabel('Graph Frequency (k)', fontweight='bold')
ax_cin.set_title('Kinematic Evolution of the Spectrum', fontweight='bold')
fig_cin.savefig(os.path.join(DIR_FIGS, "Fig5_A_Kinematic_Evolution.pdf"), bbox_inches='tight')

# STEP 6: Multiclass KAN Analysis
dataset_mc = torch.load(archivo_dataset_mc, weights_only=False)
X_test_tensor = dataset_mc['test_input']
Y_test_tensor = dataset_mc['test_label']

dimension_entrada_mc = X_test_tensor.shape[1]
modelo_kan_mc = KAN(width=[dimension_entrada_mc, 8, 4], grid=15, k=3, seed=42)
modelo_kan_inicial = KAN(width=[dimension_entrada_mc, 8, 4], grid=15, k=3, seed=42)

if os.path.exists(archivo_modelo_multiclase):
    modelo_kan_mc.load_state_dict(torch.load(archivo_modelo_multiclase, weights_only=True))
else:
    modelo_kan_mc.fit(dataset_mc, opt="LBFGS", steps=20, log=10)
    torch.save(modelo_kan_mc.state_dict(), archivo_modelo_multiclase)

# STEP 7: Symbolic Regression Extraction
fases = ["W1", "W2", "W3", "W4", "W5"]
nombres_mc = ["Normal", "Generator Trip", "Line Trip", "Load Trip"]
dimension_entrada_total = G.N * 15

if os.path.exists(ruta_ecuacion) and os.path.exists(archivo_stats_simbolicas):
    with open(archivo_stats_simbolicas, "rb") as f:
        stats = pickle.load(f)
else:
    try:
        modelo_kan_mc = modelo_kan_mc.prune(node_th=1e-3, edge_th=1e-3)
        modelo_kan_mc.auto_symbolic(lib=['x', 'x^2', 'x^3', 'exp', 'sin', 'abs'])

        variables_simbolicas = []
        for fase in fases:
            for i in range(G.N): variables_simbolicas.append(sympy.Symbol(f"λV_{i}_{fase}"))
            for i in range(G.N): variables_simbolicas.append(sympy.Symbol(f"λf_gen_{i}_{fase}"))
            for i in range(G.N): variables_simbolicas.append(sympy.Symbol(f"λf_load_{i}_{fase}"))

        formulas_multiclase = modelo_kan_mc.symbolic_formula(var=variables_simbolicas)[0]
        formulas_str = [str(f) for f in formulas_multiclase]

        with open(ruta_ecuacion, "w", encoding="utf-8") as f:
            f.write("TOPOLOGICAL CLASSIFICATION SYSTEM (15N)\n" + "="*80 + "\n")
            for i in range(4):
                f.write(f"\n[{nombres_mc[i].upper()}]\n{formulas_str[i]}\n" + "-" * 80 + "\n")

        primitivas = {'sin(x)': 0, 'exp(x)': 0, 'x²': 0, 'x³': 0, '|x|': 0}
        matriz_supervivencia = np.zeros((4, 5))
        variables_unicas_sobrevivientes = set()

        for i, form_str in enumerate(formulas_str):
            primitivas['sin(x)'] += form_str.count('sin')
            primitivas['exp(x)'] += form_str.count('exp')
            primitivas['x²'] += form_str.count('**2')
            primitivas['x³'] += form_str.count('**3')
            primitivas['|x|'] += form_str.count('Abs')
            
            vars_en_formula = re.findall(r'λ[A-Za-z_0-9]+', form_str)
            for var in vars_en_formula:
                variables_unicas_sobrevivientes.add(var)
                for j, fase in enumerate(fases):
                    if fase in var:
                        matriz_supervivencia[i, j] += 1

        stats = {
            'inputs_iniciales': dimension_entrada_total,
            'inputs_finales': len(variables_unicas_sobrevivientes),
            'primitivas': primitivas,
            'matriz_supervivencia': matriz_supervivencia
        }
        
        with open(archivo_stats_simbolicas, "wb") as f:
            pickle.dump(stats, f)

    except Exception as e:
        print(f"Error in symbolic extraction: {e}")
        stats = None