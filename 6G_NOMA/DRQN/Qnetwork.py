import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        :param input_size: 2 (Action + Reward)
        :param hidden_size: 64 ou 128 (Taille de la mémoire du LSTM)
        :param output_size: K_actions (Nombre d'actions discrètes, ex: 10)
        """
        super(QNetwork, self).__init__()
       
        # Transforme l'entrée brute (2 chiffres) en un vecteur(64 chiffres)
        self.fc1 = nn.Linear(input_size, hidden_size)
        # 2. Couche Récurrente (LSTM)
        # batch_first=True signifie que l'entrée doit être (Batch, Seq_Len, Features)
        self.lstm = nn.LSTM(
            input_size=hidden_size, 
            hidden_size=hidden_size, 
            batch_first=True)
        # 3. Couche de Décision (Q-Values): transforme la sortie du LSTM en une Q-value pour chaque action
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state=None):
        """
        Passe avant (Forward Pass)
        :param x: Tenseur d'entrée de forme (Batch_Size, Sequence_Length, Input_Size)
        :param hidden_state: Tuple (h_0, c_0) ou None (pour initialiser à 0)
        """
        
        # A. Passage dans la couche linéaire + Activation ReLU
        # x devient (Batch, Seq, Hidden_Size)
        x = F.relu(self.fc1(x))
        
        # B. Passage dans le LSTM
        # lstm_out contient les sorties pour CHAQUE pas de temps de la séquence
        # new_hidden_state contient la mémoire finale (h_n, c_n) à garder pour le tour suivant
        lstm_out, new_hidden_state = self.lstm(x, hidden_state)
        
        # C. Sélection du dernier pas de temps
        # Pour décider de l'action MAINTENANT, on regarde juste la dernière sortie du LSTM. lstm_out[:, -1, :] prend le dernier élément de la dimension 'Sequence'
        last_step_output = lstm_out[:, -1, :]
        
        # D. Calcul des Q-Values
        q_values = self.fc2(last_step_output)
        
        return q_values, new_hidden_state