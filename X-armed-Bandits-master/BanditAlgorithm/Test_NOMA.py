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
""""
def simple_test():

    # choose a test function.
    testFunction = TestFunctions.TestFunctions(functionName="hyper_ellipsoid", dimensions=10)
    #testFunction = TestFunctions.TestFunctions(functionName="analytical_g", g_params=np.array([0.1, 0.3, 1, 3, 10, 30, 90, 300]))
    #testFunction = TestFunctions.TestFunctions(functionName="SixHumpCamelback")
    
    # get the min and max bounds of the domain of the test function. 
    bounds = testFunction.get_bounds()
    
    #initialize a partitioner that will generate the tree covering sequence from the space-X defined by the above "bounds."
    partitioner = Partitioner.Partitioner(min_values=bounds[0], max_values=bounds[1])
    
    # initialize the bandit algorithm with the following.
    # ro = 0.5 is generally a good choice, for symmetric or near-symmetric X-spaces.
    # a good choice of v1 would be >= dimensions*6 for "hyper_ellipsoid function",
    # and >= 6 for "analytical_g" and "sixhumpcamelback"
    
    x_armed_bandit = HOO.HOO(v1=60, ro=0.5, covering_generator_function=partitioner.halve_one_by_one)
    x_armed_bandit.set_time_horizon(max_plays=3000)
    x_armed_bandit.set_environment(environment_function=testFunction.draw_value)
    x_armed_bandit.run_hoo()
    
    # this is the most rewarding point explored so far after "max_plays" rounds.
    print ("last selected arm was: {0}".format(x_armed_bandit.last_arm))
    
    # the rewards that are received by the bandit should be stored by the environment, as well as the best-fixed strategy.
    rewards = testFunction.drawn_values
    best = testFunction.bests

    # plotting the results.
    # -----------------------------------------------------------------------------------------
    plt.figure(0)

    cum_best = np.cumsum(np.array(best))
    plt.plot(cum_best, label="best strategy reward")
    plt.annotate('%0.2f' % cum_best[- 1], xy=(1, cum_best[- 1]), xytext=(8, 0),
                 xycoords=('axes fraction', 'data'), textcoords='offset points')

    cum_rewards = np.cumsum(np.array(rewards))
    plt.plot(cum_rewards, label="agent reward")

    cum_regret = cum_best - cum_rewards
    plt.plot(cum_regret, label="regret")

    plt.annotate('%0.2f' % cum_rewards[-1], xy=(1, cum_rewards[-1]), xytext=(8, 0),
                 xycoords=('axes fraction', 'data'), textcoords='offset points')
    plt.annotate('%0.2f' % cum_regret[- 1], xy=(1, cum_regret[- 1]), xytext=(8, 0),
                 xycoords=('axes fraction', 'data'), textcoords='offset points')

    plt.xlabel("rounds")
    plt.ylabel("cumulative rewards/regret")

    plt.legend()
    plt.show()
    
   
simple_test()
"""

def simple_test():

    # A. Initialisation de NOMA
    noma_wrapper = NOMA_Rayleigh.NOMA_Adapter()
    
    # B. Définition des bornes [0, 1] pour la puissance alpha
    bounds = [[0.0], [1]] 
    
    # C. Partitionneur (Découpage de l'espace)
    partitioner = Partitioner.Partitioner(min_values=bounds[0], max_values=bounds[1])
    
    # D. Configuration de l'Agent HOO
    # v1 : Paramètre de régularité. Max reward est 3.0, donc v1=3.0 ou 4.0 est bien.
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