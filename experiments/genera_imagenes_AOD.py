
"""
=======================================================================
DATOS 1D PARA LA RED NEURONAL
=======================================================================
Produce, para cada muestra, el par que se usa para entrenar:

  ENTRADA  ("holograma")  s(x) = sum_i sqrt(P_i) cos(2 pi f_i x / v_s + phi_i)

  SALIDA   ("imagen") la imagen de la camara en el plano focal

=======================================================================
"""

import json
from pathlib import Path

import numpy as np

# =====================================================================
# >>>>>>>>>>>>>>>>>>>>>>>>>>  PANEL  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# =====================================================================

# ---- Rutas ----
NB_PATH         = Path(__file__).resolve().parent / "PRUEBAS.ipynb"
CARPETA_SALIDA  = Path.home() / "Desktop" / "datos_AOD_TFG"

# ---- Peine de frecuencias (MHz) ----
N_TONOS         = 10
FREQ_INI        = 60.0
FREQ_FIN        = 85.0
ESPACIADO       = "uniforme"   # "uniforme" -> peine exacto (las intermodulaciones
                               #   caen EXACTAMENTE sobre otros tonos
                               # "arbitrario" -> espaciado no uniforme aleatorio
                               #   (caen ENTRE tonos)

# ---- Potencia acustica ----
P_TOTAL         = 1.2          # W, repartido entre los N tonos
G_MAX           = 3            # orden maximo de intermodulacion (3 = 2f_i-f_j, f_i+f_j-f_k)

# ---- Modelo ----
METODO          = "A"          # "A"      = Opcion A (Bragg ideal + banda por apodizacion sinc^2)
                               # "estela" = detuning analitico de Xu (eq. 2.134)
APLICAR_T       = False        # True = multiplicar por la transmitancia T(f) del transductor
TRANSDUCTOR     = "jaume"       # "raul" (AOD AAOptoelectronics) o "jaume" (phased array)

# ---- Optica del montaje (PARAMETROS DEL EXPERIMENTO) ----

W0_HAZ_MM       = 1.0          # radio 1/e^2 del haz laser en el cristal (mm)
F_LENTE_MM      = 100.0        # focal de la lente de Fourier (mm)
PIXEL_UM        = 5.2          # tamano de pixel de la camara (um)
N_PIX           = 1024         # pixeles del perfil de salida
PERFIL_PUNTO    = "gauss"      # "gauss" (haz gaussiano) o "sinc" (apertura dura)
CENTRO_CAMPO    = "peine"      # "peine"  -> campo centrado en el centro del peine
                               # "orden0" -> centrado en el orden 0 (x=0)
INCLUIR_ORDEN_0 = False        # el orden 0 es MUCHO mas intenso que los puntos difractados y se come el rango dinamico
                               
SOLO_D1         = True         # True = como si hubiera un polarizador que deja
                               # pasar solo la polarizacion difractada (D=1)

ALTO_FRANJA_PX  = 16           # alto (px) de la imagen de camara.
# ---- Holograma (entrada) ----
N_PIX_HOLO      = 4096         # muestras de s(x)
APERTURA_MM     = 4.0          # extension de x representada (mm)
ENVOLVENTE_HAZ  = False        # True = pesar s(x) por el perfil de intensidad del haz

# ---- Dataset ----
N_MUESTRAS      = 200          # pares entrada/salida a generar
SEMILLA         = 0
MODO_POTENCIAS  = "fijo"       # "aleatorio" = P_i aleatorias con sum(P_i) = P_TOTAL
                               # "fijo"      = P_i = P_TOTAL / N en todas las muestras
RANGO_AMP       = (0.3, 1.0)   # rango del sorteo de los pesos antes de normalizar
FASES_ALEATORIAS = True        # False = todas las fases a 0
GUARDAR_DATASET = True

# =====================================================================
# 1. Cargar el modelo desde el notebook
# =====================================================================

def cargar_modelo(nb_path=NB_PATH, verbose=True):
    """
    Carga el modelo desde el notebook. Ejecuta definiciones de funciones
    """
    nb = json.loads(Path(nb_path).read_text(encoding="utf-8"))
    codigo = [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]

    def buscar(marca):
        for src in codigo:
            if marca in src:
                return src
        raise RuntimeError(f"No encuentro la celda que contiene: {marca!r}")

    def cortar(src, marca):
        """Se queda con lo anterior a 'marca' (para no ejecutar los plots)."""
        i = src.find(marca)
        return src if i < 0 else src[:i]

    import matplotlib
    matplotlib.use("Agg")  # nada de ventanas

    g = {"__name__": "modelo_aod"}
    # a) funciones
    exec(cortar(buscar("def build_sparse_birref_system"), "PANEL DE CONTROL"), g)
    # b) parametros compartidos
    exec(buscar("PARAMETROS COMPARTIDOS"), g)
    # c) motor de ecualizacion 
    exec(cortar(buscar("def build_equalization_engine"), "# Ejecucion"), g)

    matplotlib.rcdefaults()
    matplotlib.rcParams.update({"axes.labelsize": 11, "axes.titlesize": 12,
                                "legend.fontsize": 9})

    for nombre in ("build_equalization_engine", "solve_birref_system_fast",
                   "generate_states_bragg", "simulate_birref_full"):
        if nombre not in g:
            raise RuntimeError(f"El notebook no ha definido {nombre}")
    if verbose:
        print(f"[modelo] cargado de {nb_path.name}")
    return g


def construir_dk_interp(g, fmin=5.0, fmax=260.0, paso=1.0):
  
    from scipy.interpolate import interp1d
    no, ne, rho = g["teo2_indices_and_rho"](g["LAMBDA_NM_MULTI"] * 1e-3)
    delta_oa = g["optical_activity_delta"](g["LAMBDA_NM_MULTI"] * 1e-9, rho, no)
    ta0 = np.deg2rad(g["THETA_A0_MULTI"])
    lam = g["LAMBDA_NM_MULTI"] * 1e-9
    L = g["L_CRYSTAL_MULTI"]
    f = np.arange(fmin, fmax + paso, paso)
    dk = np.zeros_like(f)
    for i, fi in enumerate(f):
        _eff, xi, _eta = g["xu_estela_single"](
            fi * 1e6, lam, g["ANGLE_INC_MULTI"], ta0, no, ne, delta_oa,
            L, g["H_CRYSTAL_MULTI"], P_TOTAL,
            vt=g["VT_MULTI"], vz=g["VZ_MULTI"],
            density=g["DENSITY_MULTI"], p_eff=g["P_EFF_MULTI"])
        dk[i] = -2.0 * xi / L
    return interp1d(f, dk, bounds_error=False, fill_value=(dk[0], dk[-1]))


def construir_engine(g, freqs_mhz, metodo=METODO, g_max=G_MAX,
                     aplicar_T=APLICAR_T, transductor=TRANSDUCTOR, verbose=True):
    
    kw = dict(g_max=g_max)
    if metodo == "estela":
        kw.update(use_estela_geometry=False, include_apodization=False,
                  delta_k_D1_interp=construir_dk_interp(g))
    elif metodo == "A":
        kw.update(use_estela_geometry=True, include_apodization=True,
                  apodization_mode="inside")
    else:
        raise ValueError("METODO debe ser 'A' o 'estela'")

    if aplicar_T:
        base = Path(NB_PATH).resolve().parent
        ficheros = {
            "raul": base / "FitxersSimulaPerToni" / "dades_transd"
                    / "Transmitancia_RF_AOD_optoelectronics.csv",
            "jaume": base / "dades_exp_Jaume" / "dades_transd" / "transducers_T.csv",
        }
        kw["t_interp"] = g["load_T_auto"](ficheros[transductor])

    engine = g["build_equalization_engine"](list(freqs_mhz), **kw)
    engine["_states"] = np.array(
        g["generate_states_bragg"](len(freqs_mhz), d_set=(0, 1), g_max=g_max), dtype=int)
    engine["_freqs"] = np.asarray(freqs_mhz, dtype=float)
    engine["_lambda_nm"] = float(g["LAMBDA_NM_MULTI"])
    engine["_solve"] = g["solve_birref_system_fast"]
    # v_s del canal central: fija la escala posicion <-> frecuencia
    canales = engine["r_base"]["channels"]
    engine["_v_s"] = float(canales[len(canales) // 2]["v_s"])
    if verbose:
        print(f"[engine] metodo={metodo}  N={len(freqs_mhz)}  "
              f"estados={engine['n_states']}  v_s={engine['_v_s']:.1f} m/s  "
              f"T={'si (' + transductor + ')' if aplicar_T else 'no'}")
    return engine


# =====================================================================
# 2. Resolver: amplitudes complejas de todos los ordenes
# =====================================================================

def amplitudes(engine, P_array, fases=None):
    """
    Resuelve las ecuaciones acopladas con potencias P_array (W por canal) y
    devuelve la amplitud compleja de cada estado, con las fases RF aplicadas.

    """
    P = np.maximum(np.asarray(P_array, dtype=float), 0.0)
    v = engine["v_base"] * np.sqrt(P / engine["P_BASE"])
    mat = engine["template"].copy()
    mat.data = (engine["sign_perm"] * v[engine["chan_perm"]]
                / (2.0 * engine["L"])).astype(complex)
    y = engine["_solve"](engine["phase_diag"], mat, engine["L"], engine["y0"])

    if engine["apod_mode"] == "outside":
        # legacy: el sinc^2 no viajo en la dinamica -> recortar los principales
        y = y.copy()
        y[engine["prin_idx"]] *= np.sqrt(engine["sinc2"])

    if fases is not None:
        y = y * np.exp(1j * (engine["_states"] @ np.asarray(fases, dtype=float)))

    # T(f) del transductor, FUERA de la formula (metodo Estela): intensidad x T
    # -> amplitud x sqrt(T), y solo a los estados que difractan (D=1).
    tw = engine.get("t_weight")
    if tw is not None and not np.allclose(tw, 1.0):
        f_net = engine["_states"] @ engine["_freqs"]
        D = engine["_states"].sum(axis=1)
        t_de_f = np.interp(f_net, engine["_freqs"], tw, left=tw[0], right=tw[-1])
        y = y * np.where(D == 1, np.sqrt(np.maximum(t_de_f, 0.0)), 1.0)
    return y


def agrupar_ordenes(engine, y, solo_D1=SOLO_D1, incluir_orden_0=INCLUIR_ORDEN_0):

    estados = engine["_states"]
    f_net = estados @ engine["_freqs"]
    D = estados.sum(axis=1)
    # Amplitud = |y| sin el peso (n cos) de la eq. 5.9: asi la intensidad es la
    # misma "eficiencia" I_raw que usa el notebook en todas sus graficas y en la
    # columna 'eta inicial' del Excel.
    a = y

    grupos = {}
    for k in range(len(estados)):
        if solo_D1 and D[k] != 1:
            continue
        if not incluir_orden_0 and D[k] == 0 and abs(f_net[k]) < 1e-9:
            continue
        clave = (int(D[k]), round(float(f_net[k]), 6))
        grupos.setdefault(clave, []).append(k)

    claves = sorted(grupos)
    f_out = np.array([c[1] for c in claves])
    D_out = np.array([c[0] for c in claves])
    I_out = np.array([abs(np.sum(a[grupos[c]])) ** 2 for c in claves])
    n_out = np.array([len(grupos[c]) for c in claves])
    return f_out, I_out, D_out, n_out


# =====================================================================
# 3. Renderizado 1D
# =====================================================================

def eje_camara(engine, freqs_mhz, n_pix=N_PIX, pixel_um=PIXEL_UM,
               f_lente_mm=F_LENTE_MM, centro=CENTRO_CAMPO):
    """Eje x de la camara (mm) y la escala mm/MHz."""
    lam_m = engine["_lambda_nm"] * 1e-9
    escala_mm_por_mhz = (f_lente_mm * lam_m * 1e6) / engine["_v_s"]  # F*lambda*f/v_s
    f_centro = 0.0 if centro == "orden0" else float(np.mean(freqs_mhz))
    x0 = f_centro * escala_mm_por_mhz
    ancho = n_pix * pixel_um * 1e-3
    x = x0 + (np.arange(n_pix) - (n_pix - 1) / 2.0) * pixel_um * 1e-3
    return x, escala_mm_por_mhz, ancho


def radio_punto_mm(engine, w0_haz_mm=W0_HAZ_MM, f_lente_mm=F_LENTE_MM):
    """Radio 1/e^2 del punto en el plano focal: w_f = lambda F / (pi w0)."""
    lam_mm = engine["_lambda_nm"] * 1e-6
    return lam_mm * f_lente_mm / (np.pi * w0_haz_mm)


def render_salida(engine, y, x_mm, escala_mm_por_mhz, w_punto_mm,
                  perfil=PERFIL_PUNTO, solo_D1=SOLO_D1,
                  incluir_orden_0=INCLUIR_ORDEN_0):
  
    f_net, I, D, _n = agrupar_ordenes(engine, y, solo_D1, incluir_orden_0)
    perfil_total = np.zeros_like(x_mm)
    for fi, Ii in zip(f_net, I):
        if Ii <= 0:
            continue
        u = (x_mm - fi * escala_mm_por_mhz) / w_punto_mm
        if perfil == "gauss":
            forma = np.exp(-2.0 * u ** 2)
        elif perfil == "sinc":
            forma = np.sinc(u / 1.2) ** 2   # 1.2: hace coincidir el ancho con el gauss
        else:
            raise ValueError("PERFIL_PUNTO debe ser 'gauss' o 'sinc'")
        perfil_total += Ii * forma
    return perfil_total, (f_net, I, D)


def franja_camara(perfil, x_mm, w_punto_mm, pixel_um=PIXEL_UM, alto_px=ALTO_FRANJA_PX):
   
    
    y = (np.arange(alto_px) - (alto_px - 1) / 2.0) * pixel_um * 1e-3
    return np.exp(-2.0 * (y / w_punto_mm) ** 2)[:, None] * perfil[None, :]


def render_holograma(freqs_mhz, P_array, fases, v_s,
                     n_pix=N_PIX_HOLO, apertura_mm=APERTURA_MM,
                     envolvente=ENVOLVENTE_HAZ, w0_haz_mm=W0_HAZ_MM):
    """
    s(x) = sum_i sqrt(P_i) cos(2 pi f_i x / v_s + phi_i)   (x en mm)

    """
    x = (np.arange(n_pix) - (n_pix - 1) / 2.0) * (apertura_mm / n_pix)
    f_hz = np.asarray(freqs_mhz, dtype=float) * 1e6
    k_por_mm = 2.0 * np.pi * f_hz / v_s * 1e-3        # rad/mm
    s = (np.sqrt(np.maximum(P_array, 0.0))[:, None]
         * np.cos(k_por_mm[:, None] * x[None, :] + np.asarray(fases)[:, None])).sum(axis=0)
    if envolvente:
        s = s * np.exp(-2.0 * (x / w0_haz_mm) ** 2)
    return x, s


def muestras_por_periodo(freqs_mhz, v_s, n_pix=N_PIX_HOLO, apertura_mm=APERTURA_MM):
    
    periodo_mm = v_s / (max(freqs_mhz) * 1e6) * 1e3
    return periodo_mm / (apertura_mm / n_pix)


# =====================================================================
# 4. Sorteo de una muestra
# =====================================================================

def peine(n=N_TONOS, f_ini=FREQ_INI, f_fin=FREQ_FIN, modo=ESPACIADO, rng=None):
    if modo == "uniforme" or n == 1:
        return np.linspace(f_ini, f_fin, n)
    if modo == "arbitrario":
        rng = rng or np.random.default_rng()
        # separaciones desiguales -> las intermodulaciones caen ENTRE tonos
        d = rng.uniform(0.6, 1.4, n - 1)
        f = np.concatenate([[0.0], np.cumsum(d)])
        return f_ini + (f_fin - f_ini) * f / f[-1]
    raise ValueError("ESPACIADO debe ser 'uniforme' o 'arbitrario'")


def sortear_potencias(n, rng, p_total=P_TOTAL, modo=MODO_POTENCIAS, rango=RANGO_AMP):
    if modo == "fijo":
        return np.full(n, p_total / n)
    w = rng.uniform(rango[0], rango[1], n)
    return p_total * w / w.sum()


def sortear_fases(n, rng, aleatorias=FASES_ALEATORIAS):
    return rng.uniform(0.0, 2.0 * np.pi, n) if aleatorias else np.zeros(n)


# =====================================================================
# 5. Programa principal
# =====================================================================

def main():
    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEMILLA)

    g = cargar_modelo()
    freqs = peine(rng=rng)
    engine = construir_engine(g, freqs)

    x_cam, escala, ancho_mm = eje_camara(engine, freqs)
    w_punto = radio_punto_mm(engine)
    spp = muestras_por_periodo(freqs, engine["_v_s"])

    print(f"[escala]  {escala * 1e3:.1f} um/MHz  |  campo = {ancho_mm:.2f} mm "
          f"({ancho_mm / escala:.1f} MHz)  |  punto w = {w_punto * 1e3:.1f} um "
          f"= {w_punto * 1e3 / PIXEL_UM:.1f} px")
    print(f"[holograma]  {spp:.1f} muestras por periodo acustico a {max(freqs):.0f} MHz"
          + ("" if spp >= 4 else "  <-- SUBE N_PIX_HOLO: hay aliasing"))
    sep_px = escala * (freqs[1] - freqs[0]) * 1e3 / PIXEL_UM if len(freqs) > 1 else 0
    print(f"[peine]  {len(freqs)} tonos, {freqs[0]:.2f} ... {freqs[-1]:.2f} MHz, "
          f"separacion = {sep_px:.1f} px entre puntos vecinos")

    # ---------- muestra de ejemplo ----------
    P = sortear_potencias(len(freqs), rng)
    phi = sortear_fases(len(freqs), rng)
    y = amplitudes(engine, P, phi)
    perfil, (f_net, I_grupo, _D) = render_salida(engine, y, x_cam, escala, w_punto)
    imagen = franja_camara(perfil, x_cam, w_punto)
    x_holo, s_holo = render_holograma(freqs, P, phi, engine["_v_s"])
    y_cam = (np.arange(ALTO_FRANJA_PX) - (ALTO_FRANJA_PX - 1) / 2.0) * PIXEL_UM * 1e-3

    picos = np.array([I_grupo[np.argmin(np.abs(f_net - f))] if len(f_net) else 0.0
                      for f in freqs])
    print(f"[ejemplo]  {len(f_net)} ordenes -> {len(freqs)} puntos  |  "
          f"intensidad de pico: media={picos.mean():.4f}  min={picos.min():.4f}  "
          f"max={picos.max():.4f}  CV={picos.std() / picos.mean() * 100:.2f}%")

    meta = json.dumps(dict(
        metodo=METODO, g_max=G_MAX, P_total=P_TOTAL, espaciado=ESPACIADO,
        aplicar_T=APLICAR_T, transductor=TRANSDUCTOR if APLICAR_T else None,
        w0_haz_mm=W0_HAZ_MM, f_lente_mm=F_LENTE_MM, pixel_um=PIXEL_UM,
        n_pix=N_PIX, alto_franja_px=ALTO_FRANJA_PX, perfil_punto=PERFIL_PUNTO,
        centro_campo=CENTRO_CAMPO, incluir_orden_0=INCLUIR_ORDEN_0, solo_D1=SOLO_D1,
        apertura_mm=APERTURA_MM, n_pix_holo=N_PIX_HOLO, envolvente_haz=ENVOLVENTE_HAZ,
        escala_mm_por_mhz=float(escala), w_punto_mm=float(w_punto),
        v_s=float(engine["_v_s"]), modo_potencias=MODO_POTENCIAS,
        rango_amp=list(RANGO_AMP), fases_aleatorias=FASES_ALEATORIAS, semilla=SEMILLA))

    np.savez_compressed(
        CARPETA_SALIDA / "ejemplo_muestra.npz",
        holograma=s_holo.astype(np.float32), x_holo_mm=x_holo,
        imagen=imagen.astype(np.float32), x_cam_mm=x_cam, y_cam_mm=y_cam,
        freqs_mhz=freqs, P_W=P, fases_rad=phi, meta=meta)

    # ---------- figura: entrada y salida, nada mas ----------
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(12, 6),
                           gridspec_kw={"height_ratios": [3.0, 1.0]})

    ax[0].plot(x_holo, s_holo, lw=0.7, color="#1C7293")
    ax[0].set_xlabel("x en la apertura del AOD (mm)")
    ax[0].set_ylabel(r"s(x)  [$\sqrt{W}$]")
    ax[0].set_title(f"ENTRADA: red acustica en el AOD  |  {len(freqs)} tonos, "
                    f"P_total = {P_TOTAL} W")
    ax[0].grid(alpha=0.3)

    # aspect="auto": la franja es de solo ALTO_FRANJA_PX pixeles, asi que se
    # estira en vertical para poder verla. Los datos guardados no se tocan.
    ax[1].imshow(imagen, aspect="auto", cmap="inferno", origin="lower",
                 extent=[x_cam[0], x_cam[-1], y_cam[0], y_cam[-1]])
    ax[1].set_xlabel("x en la camara (mm)")
    ax[1].set_ylabel("y (mm)")
    ax[1].set_title(f"SALIDA: imagen de la camara ({ALTO_FRANJA_PX} x {N_PIX} px)  |  {len(freqs)} puntos ")
    ax[1].set_yticks([y_cam[0], 0.0, y_cam[-1]])

    fig.tight_layout()
    fig.savefig(CARPETA_SALIDA / "ejemplo_holograma_y_salida.png", dpi=170,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {CARPETA_SALIDA / 'ejemplo_holograma_y_salida.png'}")

    # ---------- dataset ----------
    if GUARDAR_DATASET and N_MUESTRAS > 0:
        import time
        t0 = time.time()
        H = np.zeros((N_MUESTRAS, N_PIX_HOLO), dtype=np.float32)
        IMG = np.zeros((N_MUESTRAS, ALTO_FRANJA_PX, N_PIX), dtype=np.float32)
        PP = np.zeros((N_MUESTRAS, len(freqs)), dtype=np.float32)
        FI = np.zeros((N_MUESTRAS, len(freqs)), dtype=np.float32)
        for m in range(N_MUESTRAS):
            Pm = sortear_potencias(len(freqs), rng)
            phim = sortear_fases(len(freqs), rng)
            ym = amplitudes(engine, Pm, phim)
            perfm, _ = render_salida(engine, ym, x_cam, escala, w_punto)
            _xh, sh = render_holograma(freqs, Pm, phim, engine["_v_s"])
            H[m] = sh
            IMG[m] = franja_camara(perfm, x_cam, w_punto)
            PP[m], FI[m] = Pm, phim
            if (m + 1) % 50 == 0 or m == N_MUESTRAS - 1:
                print(f"  muestra {m + 1}/{N_MUESTRAS}  "
                      f"({(time.time() - t0) / (m + 1) * 1e3:.0f} ms/muestra)")
        ruta = CARPETA_SALIDA / "dataset_AOD_1D.npz"
        np.savez_compressed(
            ruta, holograma=H, imagen=IMG, P_W=PP, fases_rad=FI,
            freqs_mhz=freqs, x_holo_mm=x_holo, x_cam_mm=x_cam, y_cam_mm=y_cam,
            meta=meta)
        print(f"  -> {ruta}  ({ruta.stat().st_size / 1e6:.1f} MB, "
              f"{time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
