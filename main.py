import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Constantes physiques
G = 6.67430e-11  # Constante gravitationnelle (m^3 kg^-1 s^-2)
M_terre = 5.972e24  # Masse de la Terre (kg)
R_terre = 6371e3  # Rayon de la Terre (m)


def équations_mouvement(t, y, G, M):
    """
    Équations différentielles pour le mouvement orbital.
    y = [x, y, vx, vy]
    """
    x, y_pos, vx, vy = y
    r = np.sqrt(x ** 2 + y_pos ** 2)
    ax = -G * M * x / r ** 3
    ay = -G * M * y_pos / r ** 3
    return [vx, vy, ax, ay]


def simulation_orbite(masse, altitude, vitesse_initiale, angle_deg, durée=86400):
    """
    Simule la trajectoire orbitale d'un satellite.

    Parameters:
    - masse: Masse du satellite (kg)
    - altitude: Altitude initiale au-dessus de la Terre (m)
    - vitesse_initiale: Vitesse initiale (m/s)
    - angle_deg: Angle d'injection par rapport à l'horizontale (degrés)
    - durée: Durée de la simulation (secondes)

    Returns:
    - Solution de l'intégration
    """
    # Conditions initiales
    angle_rad = np.radians(angle_deg)
    x0 = (R_terre + altitude)
    y0 = 0
    vx0 = vitesse_initiale * np.cos(angle_rad)
    vy0 = vitesse_initiale * np.sin(angle_rad)
    y_init = [x0, y0, vx0, vy0]

    # Temps de simulation
    t_span = (0, durée)
    t_eval = np.linspace(0, durée, 10000)

    # Résolution des équations différentielles
    sol = solve_ivp(équations_mouvement, t_span, y_init, args=(G, M_terre),
                    t_eval=t_eval, rtol=1e-8, atol=1e-10)
    return sol


def visualiser_orbite(sol):
    """
    Visualise la trajectoire orbitale.

    Parameters:
    - sol: Solution de l'intégration
    """
    x = sol.y[0]
    y = sol.y[1]

    plt.figure(figsize=(8, 8))
    # Tracé de la Terre
    cercle = plt.Circle((0, 0), R_terre, color='blue', label='Terre')
    plt.gca().add_artist(cercle)

    # Tracé de la trajectoire
    plt.plot(x, y, color='red', label='Trajectoire du satellite')

    plt.xlabel('Position X (m)')
    plt.ylabel('Position Y (m)')
    plt.title('Simulation de Trajectoire Orbitale')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()


def main():
    print("=== Simulateur de Trajectoires Orbitales ===")
    try:
        masse = 1000.0
        altitude = 400000.0
        vitesse_initiale = 7800
        angle_deg = 0.0
    except ValueError:
        print("Entrée invalide. Veuillez entrer des valeurs numériques.")
        return

    print("Simulation en cours...")
    sol = simulation_orbite(masse, altitude, vitesse_initiale, angle_deg)

    if not sol.success:
        print("La simulation a échoué.")
        return

    print("Simulation terminée. Affichage de la trajectoire.")
    visualiser_orbite(sol)


if __name__ == "__main__":
    main()
