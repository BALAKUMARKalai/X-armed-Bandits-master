import numpy as np
#U1 Utilisateur Faible (Loin)
#U2 Utilisateur Fort (Proche)
class NOMA_Simulator:
    
    def __init__(self):
        self.P_max = 1.0
        self.bruit = 0.1
        self.gain_moyen_U1 = 0.2
        self.gain_moyen_U2 = 1.0
        
        self.R_target_1 = 1.0
        self.R_target_2 = 2.0
        
        self.Gamma_1 = (2**self.R_target_1) - 1
        self.Gamma_2 = (2**self.R_target_2) - 1
        
    def generate_channels_gains(self):
        g1 = np.random.exponential(scale = self.gain_moyen_U1)
        g2 = np.random.exponential(scale = self.gain_moyen_U2)
        return g1,g2
            
    def step(self,alpha): 
        #alpha :action choisie par l'agent pour l'U1 (Faible) et 1-alpha : puissance allouée à U2 (Fort)
        g1,g2 = self.generate_channels_gains()
        #P1 = alpha * P_max et P2 = (1-alpha)*P_max
        #Utilisateur Faible
        Signal_U1 = alpha* self.P_max * g1
        Interference_U1 = (1- alpha)* self.P_max*g1
        SINR_1 = Signal_U1/ (Interference_U1 + self.bruit)
            
            #Utilisateur Fort
        Signal_U2_decoding_U1 = alpha*self.P_max*g2 
        Interference_U2_seeing_U2 = (1- alpha) * self.P_max *g2
            
        SINR_2_step1 = Signal_U2_decoding_U1/(Interference_U2_seeing_U2+ self.bruit)
            
        signal_U2_clean = (1 - alpha)* self.P_max *g2
        SINR_2_step2 = signal_U2_clean/ self.bruit
            
        success_U1 = (SINR_1 >= self.Gamma_1) #SINR suffisant pour le débit cible
            
        SIC_Condition = (SINR_2_step1 >= self.Gamma_1) #capable de lire U1?
        Decoding_Condition = (SINR_2_step2 >= self.Gamma_2) #capable de lire U2
        success_U2 = SIC_Condition and Decoding_Condition
        Reward = 0
        if success_U1:
            Reward += self.R_target_1
        if success_U2:
            Reward += self.R_target_2
        Feedbacks = [1 if success_U1 else 0, 1 if success_U2 else 0]
        return Reward, Feedbacks
                
class NOMA_Adapter:
    """
    Cette classe remplace 'TestFunctions'. 
    Elle stocke l'historique pour les plots et adapte la sortie.
    """
    def __init__(self):
        self.env = NOMA_Simulator()
        self.drawn_values = [] # Historique des récompenses reçues
        self.bests = []        # Historique du "meilleur théorique" (Oracle)
        
        # Le maximum théorique possible (si tout le monde réussit)
        # Utilisé pour calculer le Regret approximatif
        self.max_theoretical_reward = self.env.R_target_1 + self.env.R_target_2                 
            
    def get_reward(self, alpha_coordinate):
        """
        HOO appelle cette fonction avec une coordonnée (ex: un tableau numpy ou une liste).
        On doit extraire le float alpha.
        """
        # Gestion du format d'entrée (HOO envoie parfois un tableau numpy)
        if isinstance(alpha_coordinate, (list, np.ndarray)):
            alpha = float(alpha_coordinate[0])
        else:
            alpha = float(alpha_coordinate)

        # Simulation
        reward, feedbacks = self.env.step(alpha)
        
        # Stockage pour les graphiques
        self.drawn_values.append(reward)
        self.bests.append(self.max_theoretical_reward) 
        
        # HOO ne veut que la récompense, pas les feedbacks
        return reward