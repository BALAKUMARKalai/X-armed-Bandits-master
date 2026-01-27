# -*- coding: utf-8 -*-
import Partitioner
import TestFunctions
import HOO
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import NOMA_Rayleigh
import NOMA_Gauss


def simple_test():

    # A. Initialisation de NOMA
    noma_wrapper = NOMA_Rayleigh.NOMA_Adapter()
    
    # B. Définition des bornes [0, 1] pour la puissance alpha
    bounds = [[0.0], [1]] 
    
    # C. Partitionneur (Découpage de l'espace)
    partitioner = Partitioner.Partitioner(min_values=bounds[0], max_values=bounds[1])
    
    # D. Configuration de l'Agent HOO
    # v1 : Paramètre de régularité.
    # ro : 0.5 est standard pour la dichotomie.
    x_armed_bandit = HOO.HOO(v1=0.6, ro=0.5, covering_generator_function=partitioner.halve_one_by_one)
    
    # Durée de la simulation
    x_armed_bandit.set_time_horizon(max_plays=5000)
    
    # Connexion : L'agent appelle 'noma_wrapper.get_reward' pour tester ses actions
    x_armed_bandit.set_environment(environment_function=noma_wrapper.get_reward)
    
    # E. Lancement
    x_armed_bandit.run_hoo()
    print("Dernière action choisie (Alpha) : {0}".format(x_armed_bandit.last_arm))
    '''
    # F. Récupération des données pour le Plot
    rewards = noma_wrapper.drawn_values
    best = noma_wrapper.bests

    # G. Affichage des courbes
    plt.figure(0)

    # Courbe 1 : Le meilleur théorique (Cumulé)
    cum_best = np.cumsum(np.array(best))
    plt.plot(cum_best, label="Maximum Theoretique (Oracle)")

    # Courbe 2 : Ce que l'agent a gagné (Cumulé)
    cum_rewards = np.cumsum(np.array(rewards))
    plt.plot(cum_rewards, label="Récompense Agent (HOO)")

    # Courbe 3 : Le Regret (La différence)
    cum_regret = cum_best - cum_rewards
    plt.plot(cum_regret, label="Regret")

    # Annotations finales
    plt.annotate('Reward: %0.2f' % cum_rewards[-1], xy=(1, cum_rewards[-1]), xytext=(8, 0), 
                 xycoords=('axes fraction', 'data'), textcoords='offset points')
    
    plt.xlabel("Rounds (Temps)")
    plt.ylabel("Valeur Cumulée")
    plt.title("Performance de HOO sur NOMA (Gauss-Markov)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    simple_test()
    '''

    rewards = np.array(noma_wrapper.drawn_values)
    bests = np.array(noma_wrapper.bests)

    # Eviter la division par zéro
    bests[bests == 0] = 1.0 
    
    # 1. Calcul du Ratio Instantané (Agent / Oracle)
    window = 50
    ratio_instantane = rewards / bests
    ratio_lisse = np.convolve(ratio_instantane, np.ones(window)/window, mode='valid')

    avg_reward_agent = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)
    avg_reward_oracle = np.cumsum(bests) / (np.arange(len(bests)) + 1)

    plt.figure(figsize=(10, 5))
    
    plt.plot(ratio_lisse, label="Efficacité (Agent/Oracle)", color='purple')
    
    plt.axhline(y=1.0, color='g', linestyle='--', label="Oracle")
    
    plt.ylim(0, 1.2)
    plt.xlabel("Rounds")
    plt.ylabel("Efficacité Normalisée (1.0 = Parfait)")
    plt.title("Performance Normalisée par l'Oracle avec le canal de Rayleigh")
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    simple_test()