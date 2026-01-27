import numpy as np
#U1 Utilisateur Faible (Loin)
#U2 Utilisateur Fort (Proche)

class NOMA_Simulator:
    
    def __init__(self):
        self.P_max = 1.0
        self.bruit = 0.001
        self.gain_moyen_U1 = 0.2
        self.gain_moyen_U2 = 0.8
        self.R_target_1 = 1.5 #(utilisateur lointain)
        self.R_target_2 = 2.5 #(utilisateur proche)
        self.Gamma_1 = (2**self.R_target_1) - 1
        self.Gamma_2 = (2**self.R_target_2) - 1
        self.g1 = 0.0
        self.g2 = 0.0
        
    def generate_channels_gains(self):
        self.g1 = np.random.exponential(scale=self.gain_moyen_U1)
        self.g2 = np.random.exponential(scale=self.gain_moyen_U2)
        return self.g1, self.g2
    
    def step(self, alpha): 
        self.generate_channels_gains()
        g1 = self.g1
        g2 = self.g2
        
        Signal_U1 = alpha * self.P_max * g1
        Interference_U1 = (1 - alpha) * self.P_max * g1
        SINR_1 = Signal_U1 / (Interference_U1 + self.bruit)
            
        Signal_U2_decoding_U1 = alpha * self.P_max * g2 
        Interference_U2_seeing_U2 = (1 - alpha) * self.P_max * g2
        SINR_2_step1 = Signal_U2_decoding_U1 / (Interference_U2_seeing_U2 + self.bruit)
            
        signal_U2_clean = (1 - alpha) * self.P_max * g2
        SINR_2_step2 = signal_U2_clean / self.bruit
            
        success_U1 = (SINR_1 >= self.Gamma_1)
        SIC_Condition = (SINR_2_step1 >= self.Gamma_1)
        Decoding_Condition = (SINR_2_step2 >= self.Gamma_2)
        success_U2 = SIC_Condition and Decoding_Condition
        
        if success_U1 and success_U2:
            Reward = 1.0
        else:
            Reward = 0.0
            
        Feedbacks = [1 if success_U1 and success_U2 else 0]
        return Reward, Feedbacks

    def check_possibility(self, alpha, g1, g2):
        Signal_U1 = alpha * self.P_max * g1
        Interference_U1 = (1 - alpha) * self.P_max * g1
        SINR_1 = Signal_U1 / (Interference_U1 + self.bruit)
        
        Signal_U2_decoding_U1 = alpha * self.P_max * g2 
        Interference_U2_seeing_U2 = (1 - alpha) * self.P_max * g2
        SINR_2_step1 = Signal_U2_decoding_U1 / (Interference_U2_seeing_U2 + self.bruit)
        
        signal_U2_clean = (1 - alpha) * self.P_max * g2
        SINR_2_step2 = signal_U2_clean / self.bruit
        
        success_U1 = (SINR_1 >= self.Gamma_1)
        SIC_Condition = (SINR_2_step1 >= self.Gamma_1)
        Decoding_Condition = (SINR_2_step2 >= self.Gamma_2)
        success_U2 = SIC_Condition and Decoding_Condition
        
        if success_U1 and success_U2:
            return 1.0
        else:
            return 0.0

class NOMA_Adapter:
    def __init__(self):
        self.env = NOMA_Simulator()
        self.drawn_values = []
        self.bests = [] 
        self.max_theoretical_reward = 1.0 

    def get_reward(self, alpha_coordinate):
        if isinstance(alpha_coordinate, (list, np.ndarray)):
            alpha = float(alpha_coordinate[0])
        else:
            alpha = float(alpha_coordinate)

        reward, feedbacks = self.env.step(alpha)
        self.drawn_values.append(reward)

        g1_curr = self.env.g1
        g2_curr = self.env.g2
        best_outcome = 0.0
        
        test_alphas = np.linspace(0.01, 0.99, 50) 
        
        for a_test in test_alphas:
            r_test = self.env.check_possibility(a_test, g1_curr, g2_curr)
            if r_test == 1.0:
                best_outcome = 1.0
                break 
        
        self.bests.append(best_outcome)
        return reward