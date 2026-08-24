# ============================================================
# tools/training_monitor.py  --  Visualizador de entrenamientos
# Uso:  python tools/training_monitor.py
# ============================================================
import json
import os
import sys

# Ver la nota equivalente en main.py: evita que un print() con emoji crashee
# el programa si la consola no está en UTF-8.
if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Mismo criterio que el resto del proyecto: datasets/ junto a la raíz por
# defecto, salvo que COFFEE_BASE_DIR indique otra ubicación.
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR    = os.environ.get("COFFEE_BASE_DIR", os.path.join(_PROJECT_DIR, "datasets"))
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
LOGS_DIR    = os.path.join(RESULTS_DIR, "logs")


# ------------------------------------------------------------------
def load_summary():
    """Carga el historial de sesiones. Devuelve [] si aún no hay ninguna
    (no debe terminar el proceso: se usa tanto desde la consola como
    desde la GUI, y ambas deben poder seguir funcionando sin logs)."""
    path = os.path.join(LOGS_DIR, "summary.json")
    if not os.path.exists(path):
        print(f"No se encontro summary.json en {LOGS_DIR}")
        print("Entrena el modelo al menos una vez para generar logs.")
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_session(timestamp: str):
    path = os.path.join(LOGS_DIR, f"session_{timestamp}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------
def plot_summary(summary: list):
    """Grafica de progreso entre sesiones de entrenamiento."""
    if not summary:
        print("Sin datos de sesiones.")
        return

    sessions    = list(range(1, len(summary) + 1))
    val_accs    = [s["best_val_acc"]  for s in summary]
    val_losses  = [s["best_val_loss"] for s in summary]
    epochs_run  = [s["epochs_run"]    for s in summary]
    num_classes = [s["num_classes"]   for s in summary]
    timestamps  = [s["timestamp"]     for s in summary]
    # Etiqueta corta: solo hora
    labels = [t[9:15] if len(t) >= 15 else t for t in timestamps]

    fig = plt.figure(figsize=(16, 10), facecolor="#1a1a2e")
    fig.suptitle("Historial de Entrenamientos - Sistema de Cafe",
                 color="white", fontsize=15, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    #  1. Precision de validacion por sesion 
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor("#0f0f23")
    bars = ax1.bar(sessions, val_accs, color=[
        "#00ff88" if v == max(val_accs) else "#4488ff" for v in val_accs
    ], edgecolor="#333355", linewidth=0.5)
    for bar, val in zip(bars, val_accs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{val:.1f}%", ha="center", va="bottom",
                 color="white", fontsize=8)
    ax1.set_title("Mejor Precision de Validacion por Sesion", color="white", fontsize=11)
    ax1.set_xlabel("Sesion #", color="#aaaaaa")
    ax1.set_ylabel("Val Accuracy (%)", color="#aaaaaa")
    ax1.tick_params(colors="#aaaaaa")
    ax1.set_ylim(0, 105)
    ax1.spines[:].set_color("#333355")
    if len(sessions) <= 15:
        ax1.set_xticks(sessions)
        ax1.set_xticklabels([f"#{s}\n{l}" for s, l in zip(sessions, labels)],
                            fontsize=7, color="#aaaaaa")

    #  2. Val loss por sesion 
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor("#0f0f23")
    ax2.plot(sessions, val_losses, "o-", color="#ff6644", linewidth=2, markersize=5)
    ax2.fill_between(sessions, val_losses, alpha=0.15, color="#ff6644")
    ax2.set_title("Mejor Val Loss", color="white", fontsize=11)
    ax2.set_xlabel("Sesion #", color="#aaaaaa")
    ax2.set_ylabel("Loss", color="#aaaaaa")
    ax2.tick_params(colors="#aaaaaa")
    ax2.spines[:].set_color("#333355")

    #  3. Epochs por sesion 
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#0f0f23")
    ax3.bar(sessions, epochs_run, color="#ffcc44", edgecolor="#333355")
    ax3.set_title("Epochs Ejecutados", color="white", fontsize=11)
    ax3.set_xlabel("Sesion #", color="#aaaaaa")
    ax3.set_ylabel("Epochs", color="#aaaaaa")
    ax3.tick_params(colors="#aaaaaa")
    ax3.spines[:].set_color("#333355")

    #  4. Clases activas por sesion 
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#0f0f23")
    ax4.step(sessions, num_classes, where="mid", color="#88ddff", linewidth=2)
    ax4.scatter(sessions, num_classes, color="#88ddff", s=40, zorder=5)
    ax4.set_title("Clases Activas en el Modelo", color="white", fontsize=11)
    ax4.set_xlabel("Sesion #", color="#aaaaaa")
    ax4.set_ylabel("N clases", color="#aaaaaa")
    ax4.set_yticks(range(0, max(num_classes) + 2))
    ax4.tick_params(colors="#aaaaaa")
    ax4.spines[:].set_color("#333355")

    #  5. Tabla resumen ultimas 5 sesiones 
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor("#0f0f23")
    ax5.axis("off")
    last5 = summary[-5:]
    table_data = [
        [f"#{len(summary)-len(last5)+i+1}", s["timestamp"][9:15],
         f"{s['best_val_acc']:.1f}%", f"{s['epochs_run']}",
         f"{s['num_classes']}"]
        for i, s in enumerate(last5)
    ]
    tbl = ax5.table(
        cellText=table_data,
        colLabels=["#", "Hora", "Acc", "Ep.", "Cls"],
        cellLoc="center", loc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#1a1a3e" if r % 2 == 0 else "#0f0f23")
        cell.set_text_props(color="white")
        cell.set_edgecolor("#333355")
    ax5.set_title("Ultimas 5 Sesiones", color="white", fontsize=11)

    plt.tight_layout()
    plt.show()
    plt.close(fig)


# ------------------------------------------------------------------
def plot_session_detail(session: dict):
    """Curvas de loss y accuracy de una sesion especifica."""
    history = session.get("history", [])
    if not history:
        print("Sin historial de epochs para esta sesion.")
        return

    epochs     = [e["epoch"]      for e in history]
    tr_loss    = [e["train_loss"] for e in history]
    val_loss   = [e["val_loss"]   for e in history]
    tr_acc     = [e["train_acc"]  for e in history]
    val_acc    = [e["val_acc"]    for e in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#1a1a2e")
    title = (f"Sesion {session['timestamp']} | "
             f"Clases: {session['num_classes']} | "
             f"Mejor val_acc: {session['best_val_acc']:.1f}%")
    fig.suptitle(title, color="white", fontsize=12, fontweight="bold")

    for ax in axes:
        ax.set_facecolor("#0f0f23")
        ax.tick_params(colors="#aaaaaa")
        ax.spines[:].set_color("#333355")

    # Loss
    axes[0].plot(epochs, tr_loss,  "o-", color="#4488ff", label="Train Loss",  lw=2, ms=4)
    axes[0].plot(epochs, val_loss, "s-", color="#ff6644", label="Val Loss",    lw=2, ms=4)
    axes[0].set_title("Loss por Epoch", color="white")
    axes[0].set_xlabel("Epoch", color="#aaaaaa")
    axes[0].set_ylabel("Loss", color="#aaaaaa")
    axes[0].legend(facecolor="#1a1a3e", edgecolor="#333355",
                   labelcolor="white", fontsize=9)

    # Accuracy
    axes[1].plot(epochs, tr_acc,  "o-", color="#4488ff", label="Train Acc",  lw=2, ms=4)
    axes[1].plot(epochs, val_acc, "s-", color="#00ff88", label="Val Acc",    lw=2, ms=4)
    axes[1].axhline(y=max(val_acc), color="#00ff88", linestyle="--",
                    alpha=0.4, linewidth=1)
    axes[1].set_title("Accuracy por Epoch", color="white")
    axes[1].set_xlabel("Epoch", color="#aaaaaa")
    axes[1].set_ylabel("Accuracy (%)", color="#aaaaaa")
    axes[1].legend(facecolor="#1a1a3e", edgecolor="#333355",
                   labelcolor="white", fontsize=9)

    # Anotar mejor punto
    best_ep = max(range(len(val_acc)), key=lambda i: val_acc[i])
    axes[1].annotate(
        f"Mejor: {val_acc[best_ep]:.1f}%",
        xy=(epochs[best_ep], val_acc[best_ep]),
        xytext=(epochs[best_ep] + 0.5, val_acc[best_ep] - 5),
        color="white", fontsize=8,
        arrowprops=dict(arrowstyle="->", color="white", lw=1),
    )

    plt.tight_layout()
    plt.show()
    plt.close(fig)


# ------------------------------------------------------------------
def menu(summary: list):
    while True:
        print("\n" + "="*50)
        print("  MONITOR DE ENTRENAMIENTO")
        print("="*50)
        print(f"  Sesiones registradas: {len(summary)}")
        if summary:
            best = max(summary, key=lambda s: s["best_val_acc"])
            idx  = summary.index(best) + 1
            print(f"  Mejor sesion: #{idx} ({best['best_val_acc']:.1f}% val_acc)")
        print()
        print("  1. Ver progreso general entre sesiones")
        print("  2. Ver detalle de una sesion especifica")
        print("  3. Listar todas las sesiones")
        print("  4. Salir")

        choice = input("\nOpcion (1-4): ").strip()

        if choice == "1":
            plot_summary(summary)

        elif choice == "2":
            if not summary:
                print("Sin sesiones.")
                continue
            print("\nSesiones disponibles:")
            for i, s in enumerate(summary, 1):
                print(f"  {i:>3}. [{s['timestamp']}] "
                      f"acc={s['best_val_acc']:.1f}%  "
                      f"clases={s['num_classes']}  "
                      f"epochs={s['epochs_run']}")
            try:
                n = int(input("\nNumero de sesion (0=cancelar): "))
                if n == 0:
                    continue
                s = summary[n - 1]
                detail = load_session(s["timestamp"])
                if detail:
                    plot_session_detail(detail)
                else:
                    print(f"No se encontro el archivo de sesion {s['timestamp']}")
            except (ValueError, IndexError):
                print("Seleccion invalida.")

        elif choice == "3":
            if not summary:
                print("Sin sesiones.")
                continue
            print(f"\n{'#':>4}  {'Fecha/Hora':<16}  {'Val Acc':>8}  "
                  f"{'Val Loss':>9}  {'Epochs':>6}  {'Clases':>6}  Clases")
            print("-" * 75)
            for i, s in enumerate(summary, 1):
                classes = ", ".join(s.get("class_list", []))
                print(f"{i:>4}  {s['timestamp']:<16}  "
                      f"{s['best_val_acc']:>7.1f}%  "
                      f"{s['best_val_loss']:>9.4f}  "
                      f"{s['epochs_run']:>6}  "
                      f"{s['num_classes']:>6}  {classes}")

        elif choice == "4":
            break


# ------------------------------------------------------------------
if __name__ == "__main__":
    summary = load_summary()
    menu(summary)