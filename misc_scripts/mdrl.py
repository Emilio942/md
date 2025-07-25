import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from rdkit.Chem.rdMolDescriptors import CalcTPSA
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ========================================================================================
# SMILES VOCABULARY AND TOKENIZER
# ========================================================================================

class SMILESVocabulary:
    """Vocabulary für SMILES-Strings"""
    def __init__(self):
        # Häufige SMILES-Tokens
        self.tokens = [
            'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I',
            'c', 'n', 'o', 's', 'p',  # aromatische Varianten
            '(', ')', '[', ']', '=', '#', '-', '+',
            '1', '2', '3', '4', '5', '6', '7', '8', '9',
            '@', '@@', 'H',
            '<PAD>', '<START>', '<END>', '<UNK>'
        ]
        self.token_to_id = {token: i for i, token in enumerate(self.tokens)}
        self.id_to_token = {i: token for i, token in enumerate(self.tokens)}
        self.vocab_size = len(self.tokens)
        
    def tokenize(self, smiles):
        """Tokenisiert einen SMILES-String"""
        tokens = []
        i = 0
        while i < len(smiles):
            # Zweistellige Tokens prüfen
            if i < len(smiles) - 1:
                two_char = smiles[i:i+2]
                if two_char in self.token_to_id:
                    tokens.append(two_char)
                    i += 2
                    continue
            
            # Einzelne Zeichen
            char = smiles[i]
            if char in self.token_to_id:
                tokens.append(char)
            else:
                tokens.append('<UNK>')
            i += 1
        return tokens
    
    def encode(self, smiles, max_length=100):
        """Enkodiert SMILES zu Token-IDs"""
        tokens = ['<START>'] + self.tokenize(smiles) + ['<END>']
        ids = [self.token_to_id.get(token, self.token_to_id['<UNK>']) for token in tokens]
        
        # Padding
        if len(ids) < max_length:
            ids.extend([self.token_to_id['<PAD>']] * (max_length - len(ids)))
        else:
            ids = ids[:max_length]
        
        return ids
    
    def decode(self, ids):
        """Dekodiert Token-IDs zurück zu SMILES"""
        tokens = [self.id_to_token[id] for id in ids]
        # Entferne spezielle Tokens
        tokens = [t for t in tokens if t not in ['<PAD>', '<START>', '<END>']]
        return ''.join(tokens)

# ========================================================================================
# MOLECULAR PROPERTY PREDICTOR (SURROGATE MODEL)
# ========================================================================================

class MolecularPropertyPredictor(nn.Module):
    """Surrogatmodell für schnelle Eigenschaftsvorhersage"""
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 4)  # [MW, LogP, TPSA, QED]
        )
        
    def forward(self, x):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        lstm_out, (hidden, _) = self.lstm(embedded)
        
        # Global average pooling
        pooled = torch.mean(lstm_out, dim=1)  # [batch_size, hidden_dim*2]
        properties = self.fc_layers(pooled)
        
        return properties

# ========================================================================================
# RL POLICY NETWORK (SMILES GENERATOR)
# ========================================================================================

class SMILESGenerator(nn.Module):
    """Policy Network für SMILES-Generierung"""
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        lstm_out = self.dropout(lstm_out)
        output = self.output_layer(lstm_out)
        return output, hidden
    
    def generate_smiles(self, vocab, max_length=50, temperature=1.0):
        """Generiert einen SMILES-String"""
        self.eval()
        with torch.no_grad():
            # Start mit <START> token
            current_token = torch.tensor([[vocab.token_to_id['<START>']]])
            hidden = None
            generated_ids = []
            
            for _ in range(max_length):
                output, hidden = self.forward(current_token, hidden)
                logits = output[0, -1, :] / temperature
                
                # Verhindere PAD, START tokens in der Generierung
                logits[vocab.token_to_id['<PAD>']] = -float('inf')
                logits[vocab.token_to_id['<START>']] = -float('inf')
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                if next_token == vocab.token_to_id['<END>']:
                    break
                    
                generated_ids.append(next_token)
                current_token = torch.tensor([[next_token]])
            
            return vocab.decode(generated_ids)

# ========================================================================================
# MOLECULAR EVALUATOR
# ========================================================================================

class MolecularEvaluator:
    """Bewertet Moleküle basierend auf verschiedenen Eigenschaften"""
    
    @staticmethod
    def calculate_properties(smiles):
        """Berechnet molekulare Eigenschaften"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Grundlegende Eigenschaften
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = CalcTPSA(mol)
            
            # QED (Drug-likeness)
            try:
                qed = Descriptors.qed(mol)
            except:
                qed = 0.0
                
            # Lipinski Rule of Five
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            
            lipinski_violations = sum([
                mw > 500,
                logp > 5,
                hbd > 5,
                hba > 10
            ])
            
            return {
                'molecular_weight': mw,
                'logp': logp,
                'tpsa': tpsa,
                'qed': qed,
                'lipinski_violations': lipinski_violations,
                'valid': True
            }
        except:
            return None
    
    @staticmethod
    def calculate_reward(smiles, target_properties=None):
        """Berechnet Reward für RL-Training"""
        props = MolecularEvaluator.calculate_properties(smiles)
        
        if props is None or not props['valid']:
            return -10.0  # Strafe für ungültige Moleküle
        
        # Ziel: Drug-like Moleküle
        if target_properties is None:
            target_properties = {
                'molecular_weight': (200, 500),  # (min, max)
                'logp': (-1, 5),
                'tpsa': (20, 130),
                'qed': (0.5, 1.0),
                'lipinski_violations': (0, 1)
            }
        
        reward = 0.0
        
        # Molekulargewicht
        mw = props['molecular_weight']
        if target_properties['molecular_weight'][0] <= mw <= target_properties['molecular_weight'][1]:
            reward += 2.0
        else:
            reward -= abs(mw - np.mean(target_properties['molecular_weight'])) / 100
        
        # LogP
        logp = props['logp']
        if target_properties['logp'][0] <= logp <= target_properties['logp'][1]:
            reward += 2.0
        else:
            reward -= abs(logp - np.mean(target_properties['logp']))
        
        # TPSA
        tpsa = props['tpsa']
        if target_properties['tpsa'][0] <= tpsa <= target_properties['tpsa'][1]:
            reward += 1.5
        else:
            reward -= abs(tpsa - np.mean(target_properties['tpsa'])) / 50
        
        # QED
        qed = props['qed']
        reward += qed * 3.0  # QED ist bereits zwischen 0 und 1
        
        # Lipinski violations
        violations = props['lipinski_violations']
        reward -= violations * 2.0
        
        return reward

# ========================================================================================
# REINFORCEMENT LEARNING TRAINER
# ========================================================================================

class RLTrainer:
    """REINFORCE-basierter Trainer für Molekülgenerierung"""
    
    def __init__(self, generator, vocab, lr=1e-4):
        self.generator = generator
        self.vocab = vocab
        self.optimizer = optim.Adam(generator.parameters(), lr=lr)
        self.rewards_history = []
        self.losses_history = []
        
    def train_episode(self, num_molecules=32, baseline_alpha=0.9):
        """Trainiert eine Episode"""
        self.generator.train()
        
        # Generiere Moleküle und sammle Rewards
        log_probs = []
        rewards = []
        generated_smiles = []
        
        for _ in range(num_molecules):
            # Generiere SMILES und berechne log probabilities
            smiles, log_prob = self._generate_with_log_prob()
            reward = MolecularEvaluator.calculate_reward(smiles)
            
            log_probs.append(log_prob)
            rewards.append(reward)
            generated_smiles.append(smiles)
        
        # Baseline für Varianzreduktion
        if not hasattr(self, 'baseline'):
            self.baseline = np.mean(rewards)
        else:
            self.baseline = baseline_alpha * self.baseline + (1 - baseline_alpha) * np.mean(rewards)
        
        # Policy Gradient Update
        policy_loss = 0
        for log_prob, reward in zip(log_probs, rewards):
            advantage = reward - self.baseline
            policy_loss -= log_prob * advantage
        
        policy_loss = policy_loss / num_molecules
        
        # Backpropagation
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Statistiken speichern
        avg_reward = np.mean(rewards)
        self.rewards_history.append(avg_reward)
        self.losses_history.append(policy_loss.item())
        
        return {
            'avg_reward': avg_reward,
            'loss': policy_loss.item(),
            'generated_smiles': generated_smiles[:5],  # Erste 5 Beispiele
            'best_reward': max(rewards),
            'valid_molecules': sum([1 for s in generated_smiles if MolecularEvaluator.calculate_properties(s) is not None])
        }
    
    def _generate_with_log_prob(self, max_length=50, temperature=1.0):
        """Generiert SMILES und berechnet log probability"""
        self.generator.eval()
        
        current_token = torch.tensor([[self.vocab.token_to_id['<START>']]])
        hidden = None
        generated_ids = []
        log_prob_sum = 0
        
        for _ in range(max_length):
            output, hidden = self.generator(current_token, hidden)
            logits = output[0, -1, :] / temperature
            
            # Verhindere bestimmte Tokens
            logits[self.vocab.token_to_id['<PAD>']] = -float('inf')
            logits[self.vocab.token_to_id['<START>']] = -float('inf')
            
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            next_token = dist.sample()
            
            log_prob_sum += dist.log_prob(next_token)
            
            if next_token.item() == self.vocab.token_to_id['<END>']:
                break
                
            generated_ids.append(next_token.item())
            current_token = next_token.unsqueeze(0)
        
        smiles = self.vocab.decode(generated_ids)
        return smiles, log_prob_sum

# ========================================================================================
# MD SIMULATION MOCK (für Demonstration)
# ========================================================================================

class MDSimulationMock:
    """Mock für MD-Simulationen (vereinfacht für Demonstration)"""
    
    @staticmethod
    def simulate_binding_affinity(smiles, protein_pdb_id="dummy"):
        """Simuliert Bindungsaffinität (in Realität würde hier echte MD laufen)"""
        props = MolecularEvaluator.calculate_properties(smiles)
        if props is None:
            return {"binding_affinity": -999, "stability": 0}
        
        # Vereinfachte "Simulation" basierend auf molekularen Eigenschaften
        # In Realität: Docking + MD-Simulation mit GROMACS/AMBER
        binding_score = (
            props['qed'] * 5 +  # Drug-likeness
            max(0, 5 - props['logp']) +  # Nicht zu lipophil
            max(0, 400 - props['molecular_weight']) / 100 +  # Angemessene Größe
            (1 if props['lipinski_violations'] == 0 else -2)
        )
        
        stability = props['qed'] * 0.8 + random.uniform(-0.2, 0.2)
        
        return {
            "binding_affinity": binding_score,
            "stability": max(0, stability),
            "rmsd": random.uniform(1.0, 3.0),
            "sasa": props['tpsa'] * 2.5
        }

# ========================================================================================
# HAUPTTRAINING UND EVALUATION
# ========================================================================================

def main():
    print("🧪 Molekulares Design mit Reinforcement Learning")
    print("=" * 60)
    
    # Setup
    vocab = SMILESVocabulary()
    generator = SMILESGenerator(vocab.vocab_size)
    trainer = RLTrainer(generator, vocab, lr=1e-4)
    
    print(f"Vokabular Größe: {vocab.vocab_size}")
    print(f"Generator Parameter: {sum(p.numel() for p in generator.parameters()):,}")
    
    # Training Loop
    num_episodes = 50
    print(f"\n🚀 Starte Training für {num_episodes} Episoden...")
    
    best_molecules = []
    
    for episode in range(num_episodes):
        stats = trainer.train_episode(num_molecules=16)
        
        if episode % 10 == 0:
            print(f"\nEpisode {episode:3d}:")
            print(f"  Avg Reward: {stats['avg_reward']:6.2f}")
            print(f"  Best Reward: {stats['best_reward']:6.2f}")
            print(f"  Valid Mols: {stats['valid_molecules']:2d}/16")
            print(f"  Loss: {stats['loss']:8.4f}")
            
            # Zeige beste Beispiele
            for i, smiles in enumerate(stats['generated_smiles'][:3]):
                props = MolecularEvaluator.calculate_properties(smiles)
                if props:
                    print(f"    Molekül {i+1}: {smiles}")
                    print(f"      MW: {props['molecular_weight']:.1f}, LogP: {props['logp']:.2f}, QED: {props['qed']:.3f}")
        
        # Sammle beste Moleküle
        for smiles in stats['generated_smiles']:
            props = MolecularEvaluator.calculate_properties(smiles)
            if props and stats['best_reward'] > 5.0:  # Nur gute Moleküle
                best_molecules.append((smiles, stats['best_reward'], props))
    
    # Finale Evaluierung mit "MD-Simulation"
    print("\n🔬 Finale Evaluierung der besten Kandidaten mit MD-Simulation...")
    
    # Sortiere beste Moleküle
    best_molecules.sort(key=lambda x: x[1], reverse=True)
    top_candidates = best_molecules[:5]
    
    print("\n🏆 Top 5 Moleküle:")
    for i, (smiles, reward, props) in enumerate(top_candidates):
        print(f"\n{i+1}. SMILES: {smiles}")
        print(f"   RL Reward: {reward:.2f}")
        print(f"   MW: {props['molecular_weight']:.1f} Da")
        print(f"   LogP: {props['logp']:.2f}")
        print(f"   TPSA: {props['tpsa']:.1f} Ų")
        print(f"   QED: {props['qed']:.3f}")
        
        # "MD-Simulation"
        md_results = MDSimulationMock.simulate_binding_affinity(smiles)
        print(f"   MD Binding Affinity: {md_results['binding_affinity']:.2f}")
        print(f"   MD Stability: {md_results['stability']:.3f}")
    
    # Visualisierung der Trainingsfortschritte
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(trainer.rewards_history)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(trainer.losses_history)
    plt.title('Training Loss')
    plt.xlabel('Episode')
    plt.ylabel('Policy Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
        # Cleanup after animation
        plt.close('all')
        import gc
        gc.collect()
    
    print("\n✅ Training abgeschlossen!")
    print(f"📊 Finale Statistiken:")
    print(f"   Durchschnittlicher Reward: {np.mean(trainer.rewards_history[-10:]):.2f}")
    print(f"   Beste Moleküle gefunden: {len(best_molecules)}")
    print(f"   Verbesserung: {trainer.rewards_history[-1] - trainer.rewards_history[0]:.2f}")

# ========================================================================================
# ZUSÄTZLICHE ANALYSE-TOOLS
# ========================================================================================

def analyze_chemical_space(molecules):
    """Analysiert den chemischen Raum der generierten Moleküle"""
    valid_mols = []
    properties = []
    
    for smiles in molecules:
        props = MolecularEvaluator.calculate_properties(smiles)
        if props:
            valid_mols.append(smiles)
            properties.append([
                props['molecular_weight'],
                props['logp'],
                props['tpsa'],
                props['qed']
            ])
    
    if not properties:
        return None
    
    properties = np.array(properties)
    
    analysis = {
        'total_molecules': len(molecules),
        'valid_molecules': len(valid_mols),
        'validity_rate': len(valid_mols) / len(molecules),
        'property_stats': {
            'molecular_weight': {'mean': np.mean(properties[:, 0]), 'std': np.std(properties[:, 0])},
            'logp': {'mean': np.mean(properties[:, 1]), 'std': np.std(properties[:, 1])},
            'tpsa': {'mean': np.mean(properties[:, 2]), 'std': np.std(properties[:, 2])},
            'qed': {'mean': np.mean(properties[:, 3]), 'std': np.std(properties[:, 3])}
        }
    }
    
    return analysis

def demo_single_molecule_analysis():
    """Demonstriert die Analyse eines einzelnen Moleküls"""
    print("\n🔍 Einzelmolekül-Analyse Demo")
    print("=" * 40)
    
    # Beispiel-Moleküle (bekannte Medikamente)
    examples = [
        ("Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
        ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
        ("Generated", "CC(C)NC1=CC=C(C=C1)C(=O)N")  # Beispiel generiertes Molekül
    ]
    
    for name, smiles in examples:
        print(f"\n{name}: {smiles}")
        props = MolecularEvaluator.calculate_properties(smiles)
        if props:
            print(f"  Molekulargewicht: {props['molecular_weight']:.1f} Da")
            print(f"  LogP: {props['logp']:.2f}")
            print(f"  TPSA: {props['tpsa']:.1f} Ų")
            print(f"  QED: {props['qed']:.3f}")
            print(f"  Lipinski Violations: {props['lipinski_violations']}")
            
            reward = MolecularEvaluator.calculate_reward(smiles)
            print(f"  RL Reward: {reward:.2f}")
            
            # MD "Simulation"
            md_results = MDSimulationMock.simulate_binding_affinity(smiles)
            print(f"  Binding Affinity: {md_results['binding_affinity']:.2f}")

if __name__ == "__main__":
    # Hauptprogramm ausführen
    main()
    
    # Zusätzliche Demo
    demo_single_molecule_analysis()
    
    print("\n" + "="*60)
    print("🎯 Nächste Schritte für echte Anwendung:")
    print("1. Echte MD-Simulationen mit GROMACS/OpenMM integrieren")
    print("2. PDB-Strukturen für Target-spezifisches Design laden")
    print("3. Größere Trainingsdaten für Surrogatmodelle sammeln")
    print("4. Graph Neural Networks für bessere Molekülrepräsentation")
    print("5. Active Learning für effizientere MD-Nutzung")
    print("="*60)